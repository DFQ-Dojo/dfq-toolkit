import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class lsq_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, bits):
        num_bins = 2**bits - 1
        bias = -num_bins // 2
        num_features = input.numel()
        grad_scale = 1.0 / np.sqrt(num_features)

        # Forward
        eps = 1e-7
        scale += eps
        transformed = input / scale - bias
        vbar = torch.clamp(transformed, 0.0, num_bins).round()
        quantized = (vbar + bias) * scale

        # Step size gradient
        error = vbar - transformed
        mask = torch.logical_and(transformed >= 0, transformed <= num_bins)
        case1 = (transformed < 0).float() * bias
        case2 = mask.float() * error
        case3 = (transformed > num_bins).float() * (bias + num_bins)
        # TODO gradient scale might be too small, so optimizing without AdaGrad might be problematic...
        ss_gradient = (case1 + case2 + case3) * grad_scale  # * 100 * scale
        ctx.save_for_backward(mask, ss_gradient)
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        mask, ss_gradient = ctx.saved_tensors
        return grad_output * mask.float(), (grad_output * ss_gradient).sum(), None


class lsq_quantize_perchannel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, bits):
        scale = scale.view(-1, 1, 1, 1)
        num_bins = 2**bits - 1
        bias = -num_bins // 2
        num_features = input.numel() / input.shape[0]
        grad_scale = 1.0 / np.sqrt(num_features * num_bins)

        # Forward
        eps = 1e-7
        scale = scale + eps
        transformed = input / scale - bias
        vbar = torch.clamp(transformed, 0.0, num_bins).round()
        quantized = (vbar + bias) * scale

        # Step size gradient
        error = vbar - transformed
        mask = torch.logical_and(transformed >= 0, transformed <= num_bins)
        case1 = (transformed < 0).float() * bias
        case2 = mask.float() * error
        case3 = (transformed > num_bins).float() * (bias + num_bins)
        ss_gradient = (case1 + case2 + case3) * grad_scale  # * 100 * scale
        ctx.save_for_backward(mask, ss_gradient)
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        mask, ss_gradient = ctx.saved_tensors
        return (
            grad_output * mask.float(),
            (grad_output * ss_gradient).sum([1, 2, 3]),
            None,
        )


class LSQ(nn.Module):
    def __init__(self, bits):
        super(LSQ, self).__init__()
        self.bits = bits
        self.step_size = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                num_bins = 2**self.bits - 1
                self.step_size.copy_(2 * x.abs().mean() / np.sqrt(num_bins))
                self.initialized = True
                print("Initializing step size to ", self.step_size)

        return lsq_quantize().apply(x, self.step_size, self.bits)


class LSQPerChannel(nn.Module):
    def __init__(self, num_channels, bits):
        super(LSQPerChannel, self).__init__()
        self.bits = bits
        self.step_size = nn.Parameter(torch.ones(num_channels), requires_grad=True)
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                num_bins = 2**self.bits - 1
                self.step_size.copy_(2 * x.abs().mean([1, 2, 3]) / np.sqrt(num_bins))
                self.initialized = True
                print("Initializing step size to ", self.step_size.mean())

        return lsq_quantize_perchannel().apply(x, self.step_size.abs(), self.bits)


class LSQConv(nn.Conv2d):
    def __init__(
        self,
        in_chn,
        out_chn,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        num_bits=1,
        mode="exact",
    ):
        super(LSQConv, self).__init__(in_chn, out_chn, kernel_size, stride, padding, dilation, groups)
        if not bias:
            self.bias = None
        self.quantizer_tensor = LSQ(num_bits)
        self.quantizer_channel = LSQPerChannel(out_chn, num_bits)
        self.quantizer_input = LSQ(num_bits)
        self.mode = mode
        self.quantize = False

    def start_quantize(self):
        self.quantize = True

    def forward(self, x):
        # scaling_factor = self.weight.abs().mean((1, 2, 3)).view(-1, 1, 1, 1)
        # quant_weights = self.quantizer(self.weight / scaling_factor) * scaling_factor
        if self.quantize:
            if self.mode == "channel":
                quant_weights = self.quantizer_channel(self.weight)
            elif self.mode == "tensor":
                quant_weights = self.quantizer_tensor(self.weight)
            elif self.mode == "quantize_channel":
                quant_weights = self.quantizer_channel(self.weight)
                x = self.quantizer_input(x)
            elif self.mode == "quantize_tensor":
                quant_weights = self.quantizer_tensor(self.weight)
                x = self.quantizer_input(x)
            elif self.mode == "exact":
                quant_weights = self.weight

            y = F.conv2d(
                x,
                quant_weights,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        else:
            y = F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        return y
