import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class ALSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, beta):
        # assert alpha > 0, "alpha={}".format(alpha)
        ctx.save_for_backward(weight, alpha, beta)
        ctx.other = g, Qn, Qp
        w_q = Round.apply(torch.div((weight - beta), alpha).clamp(Qn, Qp))
        w_q = w_q * alpha + beta
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha, beta = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = (weight - beta) / alpha
        smaller = (q_w < Qn).float()  # bool值转浮点值，1.0或者0.0
        bigger = (q_w > Qp).float()  # bool值转浮点值，1.0或者0.0
        between = 1.0 - smaller - bigger  # 得到位于量化区间的index
        grad_alpha = (
            (
                (
                    smaller * Qn
                    + bigger * Qp
                    + between * Round.apply(q_w)
                    - between * q_w
                )
                * grad_weight
                * g
            )
            .sum()
            .unsqueeze(dim=0)
        )
        grad_beta = (
            ((smaller + bigger) * grad_weight * g).sum().unsqueeze(dim=0)
        )
        # 在量化区间之外的值都是常数，故导数也是0
        grad_weight = between * grad_weight
        # 返回的梯度要和forward的参数对应起来
        return grad_weight, grad_alpha, None, None, None, grad_beta


class WLSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, per_channel):
        # assert alpha > 0, "alpha={}".format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp, per_channel
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            w_q = Round.apply(torch.div(weight, alpha).clamp(Qn, Qp))
            w_q = w_q * alpha
            w_q = torch.transpose(w_q, 0, 1)
            w_q = w_q.contiguous().view(sizes)
        else:
            w_q = Round.apply(torch.div(weight, alpha).clamp(Qn, Qp))
            w_q = w_q * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp, per_channel = ctx.other
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            q_w = weight / alpha
            q_w = torch.transpose(q_w, 0, 1)
            q_w = q_w.contiguous().view(sizes)
        else:
            q_w = weight / alpha
        smaller = (q_w < Qn).float()  # bool值转浮点值，1.0或者0.0
        bigger = (q_w > Qp).float()  # bool值转浮点值，1.0或者0.0
        between = 1.0 - smaller - bigger  # 得到位于量化区间的index
        if per_channel:
            grad_alpha = (
                (
                    smaller * Qn
                    + bigger * Qp
                    + between * Round.apply(q_w)
                    - between * q_w
                )
                * grad_weight
                * g
            )
            grad_alpha = (
                grad_alpha.contiguous()
                .view(grad_alpha.size()[0], -1)
                .sum(dim=1)
            )
        else:
            grad_alpha = (
                (
                    (
                        smaller * Qn
                        + bigger * Qp
                        + between * Round.apply(q_w)
                        - between * q_w
                    )
                    * grad_weight
                    * g
                )
                .sum()
                .unsqueeze(dim=0)
            )
        # 在量化区间之外的值都是常数，故导数也是0
        grad_weight = between * grad_weight
        return grad_weight, grad_alpha, None, None, None, None


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


# A(特征)量化
class LSQPlusActivationQuantizer(nn.Module):
    def __init__(self, a_bits, all_positive=False, batch_init=20):
        # activations 没有per-channel这个选项的
        super(LSQPlusActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2**self.a_bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = -(2 ** (self.a_bits - 1))
            self.Qp = 2 ** (self.a_bits - 1) - 1
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.g = 1
        # self.beta = torch.nn.Parameter(torch.tensor([float(0)]))
        self.beta = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.init_state = 0
        self.quantize = False

    def start_quantize(self):
        self.quantize = True

    # 量化/反量化
    def forward(self, activation):
        if self.quantize:
            assert self.batch_init == 1, "bacth init should be 1"
            with torch.no_grad():
                if self.init_state == 0:
                    self.g = 1.0 / math.sqrt(activation.numel() * self.Qp)
                    new_s_data = (
                        torch.mean(torch.abs(activation.detach()))
                        * 2
                        / (math.sqrt(self.Qp))
                    )
                    self.s.data.copy_(new_s_data)
                    self.init_state += 1
                    print(f"initial activation scale to {self.s.data}", end=" ")
                    print(f"initial activation beta to {self.beta.data}")
                else:
                    self.s.data.copy_(self.s.data.abs())
        # V1
        # print(self.a_bits, self.batch_init)
        if self.a_bits == 32:
            q_a = activation
        elif self.a_bits == 1:
            print("！Binary quantization is not supported ！")
            assert self.a_bits != 1
        else:
            q_a = ALSQPlus.apply(
                activation, self.s, self.g, self.Qn, self.Qp, self.beta
            )
            # print(self.s, self.beta)
        return q_a


# W(权重)量化
class LSQPlusWeightQuantizer(nn.Module):
    def __init__(
        self, w_bits, all_positive=False, per_channel=False, batch_init=20
    ):
        super(LSQPlusWeightQuantizer, self).__init__()
        self.w_bits = w_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2**w_bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = -(2 ** (w_bits - 1))
            self.Qp = 2 ** (w_bits - 1) - 1
        self.per_channel = per_channel
        self.init_state = 0
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.g = 1
        self.quantize = False

    def start_quantize(self):
        self.quantize = True

    # 量化/反量化
    def forward(self, weight):
        assert self.batch_init == 1 and not self.per_channel, (
            "error occurs in weight setting"
        )
        if self.quantize:
            with torch.no_grad():
                if self.init_state == 0:
                    self.g = 1.0 / math.sqrt(weight.numel() * self.Qp)
                    new_s_data = (
                        torch.mean(torch.abs(weight.detach()))
                        * 2
                        / (math.sqrt(self.Qp))
                    )
                    self.s.data.copy_(new_s_data)
                    self.init_state += 1
                    print(f"initial weight scale to {self.s.data}")
                elif self.init_state < self.batch_init:
                    self.div = 2**self.w_bits - 1
                    if self.per_channel:
                        weight_tmp = (
                            weight.detach()
                            .contiguous()
                            .view(weight.size()[0], -1)
                        )
                        mean = torch.mean(weight_tmp, dim=1)
                        std = torch.std(weight_tmp, dim=1)
                        self.s.data, _ = torch.max(
                            torch.stack(
                                [
                                    torch.abs(mean - 3 * std),
                                    torch.abs(mean + 3 * std),
                                ]
                            ),
                            dim=0,
                        )
                        self.s.data = (
                            self.s.data * 0.9 + 0.1 * self.s.data / self.div
                        )
                    else:
                        mean = torch.mean(weight.detach())
                        std = torch.std(weight.detach())
                        self.s.data = (
                            self.s.data * 0.9
                            + 0.1
                            * max(
                                [
                                    torch.abs(mean - 3 * std),
                                    torch.abs(mean + 3 * std),
                                ]
                            )
                            / self.div
                        )
                    self.init_state += 1
                elif self.init_state == self.batch_init:
                    # self.s = torch.nn.Parameter(self.s)
                    self.s.data.copy_(self.s.data.abs())

        if self.w_bits == 32:
            output = weight
        elif self.w_bits == 1:
            print("！Binary quantization is not supported ！")
            assert self.w_bits != 1
        else:
            w_q = WLSQPlus.apply(
                weight, self.s, self.g, self.Qn, self.Qp, self.per_channel
            )

            # alpha = grad_scale(self.s, g)
            # w_q = Round.apply((weight/alpha).clamp(Qn, Qp)) * alpha
        return w_q


class QuantConv2dPlus(nn.Conv2d):
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
        a_bits=8,
        w_bits=8,
        quant_inference=False,
        all_positive=False,
        per_channel=False,
        batch_init=1,
    ):
        super(QuantConv2dPlus, self).__init__(
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
        self.quant_inference = quant_inference
        self.activation_quantizer = LSQPlusActivationQuantizer(
            a_bits=a_bits, all_positive=all_positive, batch_init=batch_init
        )
        self.weight_quantizer = LSQPlusWeightQuantizer(
            w_bits=w_bits,
            all_positive=all_positive,
            per_channel=per_channel,
            batch_init=batch_init,
        )
        self.train_batch = 0

    def start_quantize(self):
        self.activation_quantizer.start_quantize()
        self.weight_quantizer.start_quantize()

    def forward(self, input):
        if self.activation_quantizer.quantize and self.train_batch == 0:
            self.train_batch = input.shape[0]
            self.activation_quantizer.batch_init = 1
            self.weight_quantizer.batch_init = 1
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight

        output = F.conv2d(
            quant_input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output


class QuantLinearPlus(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        a_bits=8,
        w_bits=8,
        quant_inference=False,
        all_positive=False,
        per_channel=False,
        batch_init=1,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.quant_inference = quant_inference
        self.activation_quantizer = LSQPlusActivationQuantizer(
            a_bits=a_bits, all_positive=all_positive, batch_init=batch_init
        )
        self.weight_quantizer = LSQPlusWeightQuantizer(
            w_bits=w_bits,
            all_positive=all_positive,
            per_channel=per_channel,
            batch_init=batch_init,
        )
        self.train_batch = 0

    def __repr__(self):
        s = super().__repr__()
        s = f"(layer={s}, qmode=lsqplus, wbit={self.weight_quantizer.w_bits}, abit={self.activation_quantizer.a_bits}, enabled={self.activation_quantizer.quantize and self.weight_quantizer.quantize})"
        return s

    def start_quantize(self):
        self.activation_quantizer.start_quantize()
        self.weight_quantizer.start_quantize()

    def forward(self, input: torch.Tensor):
        if self.activation_quantizer.quantize and self.train_batch == 0:
            self.train_batch = input.shape[0]
            self.activation_quantizer.batch_init = 1
            self.weight_quantizer.batch_init = 1
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight

        return F.linear(input=quant_input, weight=quant_weight, bias=self.bias)
