"""Quantizers and the convolution and linear layers that use them.

A `LearnedStepQuantizer` sets its step size from the first tensor it sees:
`scale = 2 * mean(|x|) / sqrt(qp) + 1e-10`, as both papers recommend. Afterwards it keeps the step
size positive by taking its absolute value before each use. Whether it has started, and the
LSQ+ gradient factor fixed on the first batch, are saved with the state dict, so a loaded
quantization-aware checkpoint keeps its trained step sizes.

`QuantConv2d` and `QuantLinear` quantize their input and their weight, then run the ordinary
operation. They subclass the float layer, so the weight and bias keep their state-dict names.
"""

from enum import StrEnum
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from dfq.quant.functions import (
    Rounding,
    integer_range,
    learned_step_quantize,
    lsq_grad_factor,
)


class Method(StrEnum):
    LSQ = "lsq"
    LSQ_PLUS = "lsq+"


class QuantConfig(BaseModel):
    """How to quantize: the method and the bit widths. 32 bits leaves a tensor in float."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Method
    weight_bits: int = Field(ge=2, le=32)
    activation_bits: int = Field(ge=2, le=32)


ROUNDING = {Method.LSQ: Rounding.HALF_EVEN, Method.LSQ_PLUS: Rounding.HALF_AWAY}


class LearnedStepQuantizer(nn.Module):
    """Fake-quantize a tensor to signed `bits`-bit integers with a learned step size.

    With `learn_offset`, the quantizer also learns the LSQ+ offset; otherwise the offset is zero.
    """

    def __init__(self, bits: int, method: Method, learn_offset: bool) -> None:
        super().__init__()
        self.bits = bits
        self.method = method
        self.qn, self.qp = integer_range(bits, signed=True)
        self.scale = nn.Parameter(torch.ones(1))
        offset = torch.zeros(1)
        if learn_offset:
            self.offset: Tensor = nn.Parameter(offset)
        else:
            self.register_buffer("offset", offset)
        self.started = False
        self.fixed_grad_factor = 0.0

    def get_extra_state(self) -> dict[str, Any]:
        return {"started": self.started, "fixed_grad_factor": self.fixed_grad_factor}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.started = state["started"]
        self.fixed_grad_factor = state["fixed_grad_factor"]

    def start(self, x: Tensor) -> None:
        initial = 2 * x.detach().abs().mean() / self.qp**0.5 + 1e-10
        self.scale.data.copy_(initial)
        self.fixed_grad_factor = lsq_grad_factor(x, self.qp)
        self.started = True

    def grad_factor(self, x: Tensor) -> float:
        if self.method is Method.LSQ:
            return lsq_grad_factor(x, self.qp)
        return self.fixed_grad_factor

    def forward(self, x: Tensor) -> Tensor:
        if self.bits == 32:
            return x
        if self.started:
            self.scale.data.abs_()
        else:
            self.start(x)
        grad_factor = self.grad_factor(x)
        rounding = ROUNDING[self.method]
        return learned_step_quantize(
            x, self.scale, self.offset, self.qn, self.qp, grad_factor, rounding
        )

    def extra_repr(self) -> str:
        return f"bits={self.bits}, method={self.method}"


def input_quantizer(config: QuantConfig) -> LearnedStepQuantizer:
    learn_offset = config.method is Method.LSQ_PLUS
    return LearnedStepQuantizer(config.activation_bits, config.method, learn_offset)


def weight_quantizer(config: QuantConfig) -> LearnedStepQuantizer:
    return LearnedStepQuantizer(config.weight_bits, config.method, learn_offset=False)


class QuantConv2d(nn.Conv2d):
    input_quantizer: LearnedStepQuantizer
    weight_quantizer: LearnedStepQuantizer

    def forward(self, x: Tensor) -> Tensor:
        quantized_input = self.input_quantizer(x)
        quantized_weight = self.weight_quantizer(self.weight)
        return self._conv_forward(quantized_input, quantized_weight, self.bias)


class QuantLinear(nn.Linear):
    input_quantizer: LearnedStepQuantizer
    weight_quantizer: LearnedStepQuantizer

    def forward(self, x: Tensor) -> Tensor:
        quantized_input = self.input_quantizer(x)
        quantized_weight = self.weight_quantizer(self.weight)
        return F.linear(quantized_input, quantized_weight, self.bias)


def pair(values: tuple[int, ...]) -> tuple[int, int]:
    return values[0], values[1]


def quantized_conv(conv: nn.Conv2d, config: QuantConfig) -> QuantConv2d:
    """Return a quantized convolution that shares `conv`'s weight and bias."""
    padding = conv.padding if isinstance(conv.padding, str) else pair(conv.padding)
    quantized = QuantConv2d(
        conv.in_channels,
        conv.out_channels,
        pair(conv.kernel_size),
        stride=pair(conv.stride),
        padding=padding,
        dilation=pair(conv.dilation),
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
    )
    quantized.weight = conv.weight
    quantized.bias = conv.bias
    quantized.input_quantizer = input_quantizer(config).to(conv.weight.device)
    quantized.weight_quantizer = weight_quantizer(config).to(conv.weight.device)
    return quantized


def quantized_linear(linear: nn.Linear, config: QuantConfig) -> QuantLinear:
    """Return a quantized linear layer that shares `linear`'s weight and bias."""
    quantized = QuantLinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
    )
    quantized.weight = linear.weight
    quantized.bias = linear.bias
    quantized.input_quantizer = input_quantizer(config).to(linear.weight.device)
    quantized.weight_quantizer = weight_quantizer(config).to(linear.weight.device)
    return quantized
