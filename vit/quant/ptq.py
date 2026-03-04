import torch
from torch.autograd.function import Function
import torch.nn.functional as F
import torch.nn as nn


class FakeQuantize(Function):
    @staticmethod
    def forward(ctx, x, qparam):
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# asymmetric
def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    qmin = 0.0
    qmax = 2.0**num_bits - 1.0
    scale = (max_val - min_val) / (qmax - qmin)

    zero_point = qmax - max_val / scale

    zero_point.clamp_(min=qmin).to(torch.float32).to(min_val.dtype)
    zero_point.clamp_(max=qmax).to(torch.float32).to(max_val.dtype)

    zero_point.round_()

    return scale, zero_point


# symmetric
def calcScale(min_val, max_val, num_bits=8):
    _ = -(2.0 ** (num_bits - 1))  # ?
    qmax = 2.0 ** (num_bits - 1) - 1
    scale = torch.max(abs(min_val), abs(max_val)) / qmax

    return scale


def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False, mode="tensor"):
    """量化张量，支持2D和4D张量"""
    if mode == "tensor":
        if signed:
            q_x = x / scale
        else:
            q_x = zero_point + x / scale

    elif mode == "channel":
        # 修改这里：支持2D和4D张量
        if len(x.shape) == 2:  # 线性层权重 [out_features, in_features]
            if signed:
                q_x = x / scale.view(-1, 1)  # 每个输出特征使用不同的scale
            else:
                q_x = zero_point.view(-1, 1) + x / scale.view(-1, 1)
        elif len(x.shape) == 4:  # 卷积层权重 [out_channels, in_channels, H, W]
            if signed:
                q_x = x / scale.view(-1, 1, 1, 1)
            else:
                q_x = zero_point.view(-1, 1, 1, 1) + x / scale.view(-1, 1, 1, 1)
        else:
            raise ValueError(f"不支持的张量形状用于通道级量化: {x.shape}")

    if signed:
        qmin = -(2.0 ** (num_bits - 1))
        qmax = 2.0 ** (num_bits - 1) - 1
    else:
        qmin = 0.0
        qmax = 2.0**num_bits - 1.0

    q_x.clamp_(qmin, qmax).round_()

    return q_x


def dequantize_tensor(q_x, scale, zero_point, signed=False, mode="tensor"):
    """反量化张量，支持2D和4D张量"""
    if mode == "tensor":
        if signed:
            return scale * q_x
        else:
            return scale * (q_x - zero_point)
    elif mode == "channel":
        # 修改这里：支持2D和4D张量
        if len(q_x.shape) == 2:  # 线性层
            if signed:
                return scale.view(-1, 1) * q_x
            else:
                return scale.view(-1, 1) * (q_x - zero_point.view(-1, 1))
        elif len(q_x.shape) == 4:  # 卷积层
            if signed:
                return scale.view(-1, 1, 1, 1) * q_x
            else:
                return scale.view(-1, 1, 1, 1) * (q_x - zero_point.view(-1, 1, 1, 1))
        else:
            raise ValueError(f"不支持的张量形状用于通道级反量化: {q_x.shape}")


class QParam(nn.Module):
    def __init__(self, num_bits=8, mode="tensor", signed=False):
        super(QParam, self).__init__()
        self.num_bits = num_bits
        scale = torch.tensor([], requires_grad=False)
        zero_point = torch.tensor([], requires_grad=False)
        min_val = torch.tensor([], requires_grad=False)
        max_val = torch.tensor([], requires_grad=False)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        self.register_buffer("min", min_val)
        self.register_buffer("max", max_val)
        self.mode = mode
        self.signed = signed
        self.original_shape = None  # 记录原始形状

    def update(self, tensor):
        """更新量化参数，支持2D和4D张量"""
        self.original_shape = tensor.shape
        
        if self.mode == "tensor":
            if self.max.nelement() == 0 or self.max.data < tensor.max().data:
                self.max.data = tensor.max().data
            self.max.clamp_(min=0)

            if self.min.nelement() == 0 or self.min.data > tensor.min().data:
                self.min.data = tensor.min().data
            self.min.clamp_(max=0)

            if self.signed:
                self.scale, self.zero_point = calcScale(
                    self.min, self.max, self.num_bits
                ), torch.tensor([0]).to(torch.float32).to(self.scale.device)
            else:
                self.scale, self.zero_point = calcScaleZeroPoint(
                    self.min, self.max, self.num_bits
                )
                
        elif self.mode == "channel":
            # 修改这里：支持2D和4D张量
            if len(tensor.shape) == 2:  # 线性层权重 [out_features, in_features]
                # 按输出特征统计
                max_values = torch.max(tensor, dim=1)[0]  # 每个输出特征的最大值
                min_values = torch.min(tensor, dim=1)[0]  # 每个输出特征的最小值
                
            elif len(tensor.shape) == 4:  # 卷积层权重 [out_channels, in_channels, H, W]
                # 重塑为 [out_channels, -1] 然后按通道统计
                tensor_flat = tensor.view(tensor.size(0), -1)
                max_values = torch.max(tensor_flat, dim=1)[0]
                min_values = torch.min(tensor_flat, dim=1)[0]
                
            else:
                raise ValueError(f"不支持的张量形状用于通道级量化: {tensor.shape}")
            
            # 更新统计
            if self.max.nelement() == 0:
                self.max.data = max_values.data
            else:
                # 逐元素比较，保持每个通道的最大值
                self.max = torch.max(self.max, max_values)
            self.max.clamp_(min=0)

            if self.min.nelement() == 0:
                self.min.data = min_values.data
            else:
                self.min = torch.min(self.min, min_values)
            self.min.clamp_(max=0)

            # 计算量化参数
            if self.signed:
                self.scale = calcScale(self.min, self.max, self.num_bits)
                self.zero_point = torch.zeros_like(self.scale)
            else:
                # 对于非对称量化，需要为每个通道计算zero_point
                qmin = 0.0
                qmax = 2.0**self.num_bits - 1.0
                self.scale = (self.max - self.min) / (qmax - qmin)
                self.zero_point = qmax - self.max / self.scale
                self.zero_point.clamp_(qmin, qmax).round_()


    def quantize_tensor(self, tensor):
        return quantize_tensor(
            tensor,
            self.scale,
            self.zero_point,
            num_bits=self.num_bits,
            signed=self.signed,
            mode=self.mode,
        )

    def dequantize_tensor(self, q_x):
        return dequantize_tensor(
            q_x, self.scale, self.zero_point, signed=self.signed, mode=self.mode
        )


class Qconv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        weight_bits=4,
        activation_bits=4,
        mode="quantize_sym",
    ):
        super(Qconv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.mode = mode
        self.quantize = False
        self.first_pass = False
        self.calibration = True

    def start_quantize(self):
        if self.mode == "quantize_sym":
            signed = True
        self.qi = QParam(
            num_bits=self.activation_bits, mode="tensor", signed=signed
        )  # input quantizer
        self.qw = QParam(
            num_bits=self.weight_bits, mode="channel", signed=signed
        )  # weight quantizer
        self.quantize = True
        print("start quantize")

    def end_calibration(self):
        self.calibration = False

    def forward(self, x):
        if self.quantize:
            if self.calibration:
                self.qi.update(x)
                self.qw.update(self.weight.data)


            x = F.conv2d(
                FakeQuantize.apply(x, self.qi),
                FakeQuantize.apply(self.weight, self.qw),
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        else:
            x = F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        return x

    def __repr__(self):
        s = super().__repr__()
        s = f"(layer={s}, qmode={self.mode}, wbit={self.weight_bits}, abit={self.activation_bits}, enabled={self.quantize})"
        return s

class QLinear(nn.Linear):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: str = "quantize_sym",
        weight_bits: int = 32,
        activation_bits: int = 32,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.mode = mode
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.quantize = False
        self.first_pass = False
        self.calibration = True

    def __repr__(self):
        s = super().__repr__()
        s = f"(layer={s}, qmode={self.mode}, wbit={self.weight_bits}, abit={self.activation_bits}, enabled={self.quantize})"
        return s

    def start_quantize(self):
        if self.mode == "quantize_sym":
            signed = True
        self.qi = QParam(
            num_bits=self.activation_bits, mode="tensor", signed=signed
        )  # input quantizer
        self.qw = QParam(
            num_bits=self.weight_bits, mode="channel", signed=signed
        )  # weight quantizer
        self.quantize = True
        print("start quantize")

    def end_calibration(self):
        self.calibration = False


    def forward(self, x: torch.Tensor):
        if self.quantize:
            if self.calibration:
                self.qi.update(x)
                self.qw.update(self.weight.data)

            x = F.linear(
                FakeQuantize.apply(x, self.qi),
                FakeQuantize.apply(self.weight, self.qw),
                self.bias,
            )
        else:
            x = F.linear(
            input=x,
            weight=self.weight,
            bias=self.bias,
        )

        return x