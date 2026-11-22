"""An exponential moving average (EMA) of a network's weights, as YOLO training keeps.

After update `n`, `ema = d * ema + (1 - d) * weights` with `d = decay * (1 - exp(-n / tau))`, so
the average follows the weights closely at first. Integer buffers and extra state, such as
whether a quantizer has started, are copied.
"""

import copy
import math

import torch
from torch import nn


def averaged_copy(network: nn.Module) -> nn.Module:
    average = copy.deepcopy(network).eval()
    for parameter in average.parameters():
        parameter.requires_grad_(False)
    return average


def update_average(
    average: nn.Module, network: nn.Module, updates: int, decay: float, tau: float
) -> None:
    weight = decay * (1 - math.exp(-updates / tau))
    current = network.state_dict()
    copied = {}
    with torch.no_grad():
        for name, value in average.state_dict().items():
            if not isinstance(value, torch.Tensor):
                copied[name] = current[name]
            elif value.dtype.is_floating_point:
                value.mul_(weight).add_(current[name].detach(), alpha=1 - weight)
            else:
                value.copy_(current[name])
    average.load_state_dict(copied, strict=False)
