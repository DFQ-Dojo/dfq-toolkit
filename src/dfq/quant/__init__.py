"""Fake quantization: LSQ and LSQ+ quantizers, quantized layers, and the scope that applies them."""

from dfq.quant.layers import Method, QuantConfig, QuantConv2d, QuantLinear
from dfq.quant.scope import QuantScope, quantize

__all__ = ["Method", "QuantConfig", "QuantConv2d", "QuantLinear", "QuantScope", "quantize"]
