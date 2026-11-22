# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules of YOLO11, from ultralytics 8.3.53 (`ultralytics/nn/modules/conv.py`).

Attribute names (`conv`, `bn`, `act`) match Ultralytics checkpoints; constructor arguments follow
the order of the model YAML: input channels, output channels, kernel, stride, ...
"""

import math

import torch
from torch import Tensor, nn


def same_padding(kernel: int, dilation: int = 1) -> int:
    """The padding that keeps the spatial size at stride 1."""
    effective = dilation * (kernel - 1) + 1
    return effective // 2


class Conv(nn.Module):
    """Convolution without bias, batch norm, then SiLU (or `act`, or no activation)."""

    default_act: nn.Module = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        padding = same_padding(k, d) if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, padding, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if act is True:
            self.act = self.default_act
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """Depth-wise convolution: one group per channel pair."""

    def __init__(
        self, c1: int, c2: int, k: int = 1, s: int = 1, d: int = 1, act: bool | nn.Module = True
    ) -> None:
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Concat(nn.Module):
    """Concatenate a list of tensors along `dimension`."""

    def __init__(self, dimension: int = 1) -> None:
        super().__init__()
        self.d = dimension

    def forward(self, x: list[Tensor]) -> Tensor:
        return torch.cat(x, self.d)
