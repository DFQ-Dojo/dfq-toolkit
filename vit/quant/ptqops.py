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


def quantize_tensor(
    x, scale, zero_point, num_bits=8, signed=False, mode="tensor"
):
    if mode == "tensor":
        if signed:
            q_x = x / scale
        else:
            q_x = zero_point + x / scale

    elif mode == "channel":
        assert len(x.shape) == 4 and signed
        q_x = x / scale.view(-1, 1, 1, 1)

    if signed:
        qmin = -(2.0 ** (num_bits - 1))
        qmax = 2.0 ** (num_bits - 1) - 1
    else:
        qmin = 0.0
        qmax = 2.0**num_bits - 1.0

    q_x.clamp_(qmin, qmax).round_()

    return q_x


def dequantize_tensor(q_x, scale, zero_point, signed=False, mode="tensor"):
    if mode == "tensor":
        if signed:
            return scale * q_x
        else:
            return scale * (q_x - zero_point)
    elif mode == "channel":
        assert len(q_x.shape) == 4 and signed
        return scale.view(-1, 1, 1, 1) * (q_x)


class QParam(nn.Module):
    def __init__(self, num_bits=8, mode="tensor", signed=False):
        super(QParam, self).__init__()
        self.num_bits = num_bits
        scale = torch.tensor([], requires_grad=False)
        zero_point = torch.tensor([], requires_grad=False)
        min = torch.tensor([], requires_grad=False)
        max = torch.tensor([], requires_grad=False)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        self.register_buffer("min", min)
        self.register_buffer("max", max)
        self.mode = mode
        self.signed = signed

    def update(self, tensor):
        if self.mode == "tensor":
            if self.max.nelement() == 0 or self.max.data < tensor.max().data:
                self.max.data = tensor.max().data
            self.max.clamp_(min=0)

            if self.min.nelement() == 0 or self.min.data > tensor.min().data:
                self.min.data = tensor.min().data
            self.min.clamp_(max=0)

            if self.signed:
                self.scale, self.zero_point = (
                    calcScale(self.min, self.max, self.num_bits),
                    torch.tensor([0]).to(torch.float32).to(self.scale.device),
                )
            else:
                self.scale, self.zero_point = calcScaleZeroPoint(
                    self.min, self.max, self.num_bits
                )
        elif self.mode == "channel":
            assert len(tensor.shape) == 4
            max_values = torch.max(tensor.view(tensor.size(0), -1), dim=1)[0]
            if self.max.nelement() == 0:
                self.max.data = max_values.data
            else:
                self.max = torch.max(self.max, max_values)
            self.max.clamp_(min=0)

            min_values = torch.min(tensor.view(tensor.size(0), -1), dim=1)[0]
            if self.min.nelement() == 0:
                self.min.data = min_values.data
            else:
                self.min = torch.min(self.min, min_values)
            self.min.clamp_(max=0)

            assert len(self.max.shape) == 1

            self.scale, self.zero_point = (
                calcScale(self.min, self.max, self.num_bits),
                torch.tensor([0]).to(torch.float32).to(self.scale.device),
            )

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
        num_bits=8,
        mode="quantize",
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
        self.num_bits = num_bits
        self.mode = mode
        self.quantize = False
        self.first_pass = False
        self.calibration = True

    def start_quantize(self):
        self.qi = QParam(
            num_bits=self.num_bits, mode="tensor", signed=False
        )  # input quantizer
        self.qw = QParam(
            num_bits=self.num_bits, mode="channel", signed=True
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
