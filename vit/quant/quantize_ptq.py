import torch.nn as nn

from quant.quant_mode import QuantizeMode

from quant.qatops import LSQconv2d, LSQLinear

from quant.ptq import Qconv2d, QLinear   

from quant.lsqplus import QuantLinearPlus, QuantConv2dPlus


def _build_inner_conv(c1, c2, k, s, p, g, d, bias, mode: QuantizeMode):
    if mode.kind == "exact" or (mode.weight_bits == 32 and mode.activation_bits == 32):
        return nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            dilation=d,
            bias=bias,
        )
    if mode.kind == "quantize_sym":
        return Qconv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            dilation=d,
            bias=bias,
            weight_bits=mode.weight_bits,
            activation_bits=mode.activation_bits,
            mode="quantize_sym",
        )
    return QuantConv2dPlus(
        in_channels=c1,
        out_channels=c2,
        kernel_size=k,
        stride=s,
        padding=p,
        groups=g,
        dilation=d,
        bias=bias,
        a_bits=mode.activation_bits,
        w_bits=mode.weight_bits,
    )


def convert_conv(conv: nn.Conv2d, mode: QuantizeMode):
    has_bias = conv.bias is not None

    lsq_conv = _build_inner_conv(
        c1=conv.in_channels,
        c2=conv.out_channels,
        k=conv.kernel_size,
        s=conv.stride,
        p=conv.padding,
        g=conv.groups,
        d=conv.dilation,
        bias=has_bias,
        mode=mode,
    )
    lsq_conv.state_dict()["weight"].copy_(conv.state_dict()["weight"])
    if has_bias:
        lsq_conv.state_dict()["bias"].copy_(conv.state_dict()["bias"])
    return lsq_conv


def _build_inner_linear(in_features, out_features, bias, mode: QuantizeMode):
    if mode.kind == "exact" or (mode.weight_bits == 32 and mode.activation_bits == 32):
        return nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
    if mode.kind == "quantize_sym":
        return QLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            mode="quantize_sym",
            weight_bits=mode.weight_bits,
            activation_bits=mode.activation_bits,
        )
    return QuantLinearPlus(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        a_bits=mode.activation_bits,
        w_bits=mode.weight_bits,
    )


def convert_linear(linear: nn.Linear, mode: QuantizeMode):
    has_bias = linear.bias is not None

    lsq_linear = _build_inner_linear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=has_bias,
        mode=mode,
    )
    lsq_linear.state_dict()["weight"].copy_(linear.state_dict()["weight"])
    if has_bias:
        lsq_linear.state_dict()["bias"].copy_(linear.state_dict()["bias"])
    return lsq_linear
