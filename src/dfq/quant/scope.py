"""Which layers of a model to quantize, and the function that quantizes them.

A `QuantScope` names layers by their qualified module names (`backbone.body.layer1.0.conv1`)
with shell-style patterns. A layer is quantized when its name matches an `include` pattern and no
`exclude` pattern. Only `nn.Conv2d` and `nn.Linear` layers are quantized. Each detector adapter
owns its scope: for example, YOLO keeps its first convolution and its detection head in float.
"""

from dataclasses import dataclass
from fnmatch import fnmatchcase

from torch import nn

from dfq.quant.layers import (
    QuantConfig,
    QuantConv2d,
    QuantLinear,
    quantized_conv,
    quantized_linear,
)


@dataclass(frozen=True)
class QuantScope:
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()

    def contains(self, name: str) -> bool:
        included = any(fnmatchcase(name, pattern) for pattern in self.include)
        excluded = any(fnmatchcase(name, pattern) for pattern in self.exclude)
        return included and not excluded


def quantizable_names(model: nn.Module, scope: QuantScope) -> list[str]:
    """Return the names of the float convolutions and linear layers inside `scope`."""
    names = []
    for name, module in model.named_modules():
        is_float_layer = isinstance(module, nn.Conv2d | nn.Linear) and not isinstance(
            module, QuantConv2d | QuantLinear
        )
        if is_float_layer and scope.contains(name):
            names.append(name)
    return names


def quantized_layer(layer: nn.Module, config: QuantConfig) -> nn.Module:
    if isinstance(layer, nn.Conv2d):
        return quantized_conv(layer, config)
    if isinstance(layer, nn.Linear):
        return quantized_linear(layer, config)
    raise TypeError(f"cannot quantize {type(layer).__name__}")


def quantize(model: nn.Module, config: QuantConfig, scope: QuantScope) -> list[str]:
    """Replace each layer inside `scope` with its quantized copy, in place.

    Returns the names of the replaced layers. The copies share the float layers' parameters, so
    quantize a model after loading its float weights.
    """
    names = quantizable_names(model, scope)
    for name in names:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        layer = parent.get_submodule(child_name)
        replacement = quantized_layer(layer, config)
        setattr(parent, child_name, replacement)
    return names
