# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""The YOLO11 detection head, from ultralytics 8.3.53 (`ultralytics/nn/modules/head.py`).

Kept: training and inference. Removed: export formats and the end-to-end (YOLOv10) branch. In
training mode the head returns the raw output of each level, `[B, 4 * reg_max + nc, H, W]`; in
evaluation mode it also returns the decoded boxes and class scores, `[B, 4 + nc, anchors]`.
"""

import math

import torch
from torch import Tensor, nn

from ...utils.tal import dist2bbox, make_anchors
from .block import DFL
from .conv import Conv, DWConv


def box_branch(channels: int, hidden: int, reg_max: int) -> nn.Sequential:
    return nn.Sequential(
        Conv(channels, hidden, 3), Conv(hidden, hidden, 3), nn.Conv2d(hidden, 4 * reg_max, 1)
    )


def class_branch(channels: int, hidden: int, nc: int, legacy: bool) -> nn.Sequential:
    """Two convolutions (depth-wise and point-wise since YOLO11, plain before), then the classes."""
    if legacy:
        return nn.Sequential(
            Conv(channels, hidden, 3), Conv(hidden, hidden, 3), nn.Conv2d(hidden, nc, 1)
        )
    first = nn.Sequential(DWConv(channels, channels, 3), Conv(channels, hidden, 1))
    second = nn.Sequential(DWConv(hidden, hidden, 3), Conv(hidden, hidden, 1))
    return nn.Sequential(first, second, nn.Conv2d(hidden, nc, 1))


class Detect(nn.Module):
    # The anchors are cached for the last input shape; checkpoints pickle instances without them.
    anchors = torch.empty(0)
    strides = torch.empty(0)
    shape: torch.Size | None = None

    def __init__(self, nc: int = 80, ch: tuple[int, ...] = (), legacy: bool = False) -> None:
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        box_hidden = max(16, ch[0] // 4, self.reg_max * 4)
        class_hidden = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(box_branch(x, box_hidden, self.reg_max) for x in ch)
        self.cv3 = nn.ModuleList(class_branch(x, class_hidden, nc, legacy) for x in ch)
        self.dfl = DFL(self.reg_max)

    def forward(self, x: list[Tensor]) -> list[Tensor] | tuple[Tensor, list[Tensor]]:
        raw = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)]
        if self.training:
            return raw
        return self.inference(raw), raw

    def inference(self, raw: list[Tensor]) -> Tensor:
        """Decode boxes (centre, width, height in pixels) and sigmoid class scores."""
        shape = raw[0].shape
        joined = torch.cat([level.view(shape[0], self.no, -1) for level in raw], 2)
        if self.shape != shape:
            anchors, strides = make_anchors(raw, self.stride, 0.5)
            self.anchors, self.strides = anchors.transpose(0, 1), strides.transpose(0, 1)
            self.shape = shape
        box, cls = joined.split((self.reg_max * 4, self.nc), 1)
        distances = self.dfl(box)
        boxes = dist2bbox(distances, self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        return torch.cat((boxes, cls.sigmoid()), 1)

    def bias_init(self) -> None:
        """Start with box sides of 1 and a class prior of 0.01 objects per 640-pixel image."""
        for box, cls, stride in zip(self.cv2, self.cv3, self.stride, strict=True):
            assert isinstance(box, nn.Sequential) and isinstance(cls, nn.Sequential)
            box_output, class_output = box[-1], cls[-1]
            assert isinstance(box_output, nn.Conv2d) and isinstance(class_output, nn.Conv2d)
            assert box_output.bias is not None and class_output.bias is not None
            box_output.bias.data[:] = 1.0
            class_output.bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)
