# Ultralytics YOLOv5, AGPL-3.0 license
"""YOLOv5's blocks, from YOLOv5 v7.0 (`models/common.py`).

`Conv`, `Concat` and `SPPF` compute exactly what the vendored Ultralytics ones do, with the same
attribute names, so YOLOv5 uses those. Checkpoints name them `models.common.*`; see `MODULE_PATHS`.
"""

import torch
from torch import Tensor, nn

from dfq.vendor.ultralytics.nn.modules.block import SPPF
from dfq.vendor.ultralytics.nn.modules.conv import Concat, Conv

__all__ = ["C3", "SPPF", "Bottleneck", "Concat", "Conv"]


class Bottleneck(nn.Module):
    """A 1x1 then a 3x3 convolution, with a residual connection when shapes allow."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5) -> None:
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(hidden, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: Tensor) -> Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    """CSP bottleneck with three convolutions and `n` bottlenecks."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5
    ) -> None:
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(hidden, hidden, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: Tensor) -> Tensor:
        branch = self.m(self.cv1(x))
        return self.cv3(torch.cat((branch, self.cv2(x)), 1))
