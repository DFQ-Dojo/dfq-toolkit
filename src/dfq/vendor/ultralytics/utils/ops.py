# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Box conversions and NMS, from ultralytics 8.3.53 (`ultralytics/utils/ops.py`).

`non_max_suppression` keeps the path YOLO detection takes: the best class per box, class-aware
NMS, no masks. It does not change its input, unlike upstream's default.
"""

import math

import torch
import torchvision
from torch import Tensor


def make_divisible(x: float, divisor: int) -> int:
    return math.ceil(x / divisor) * divisor


def xywh2xyxy(x: Tensor) -> Tensor:
    """Centre boxes `[..., 4]` to corner boxes."""
    corners = torch.empty_like(x)
    half = x[..., 2:] / 2
    corners[..., :2] = x[..., :2] - half
    corners[..., 2:] = x[..., :2] + half
    return corners


def non_max_suppression(
    prediction: Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    max_nms: int = 30000,
    max_wh: int = 7680,
) -> list[Tensor]:
    """Per image, `[N, 6]` rows of (x1, y1, x2, y2, score, class) from `[B, 4 + nc, anchors]`."""
    classes = prediction.shape[1] - 4
    candidates = prediction[:, 4:].amax(1) > conf_thres
    rows = prediction.transpose(-1, -2)
    boxes = xywh2xyxy(rows[..., :4])
    rows = torch.cat((boxes, rows[..., 4:]), dim=-1)
    output = []
    for image, kept in enumerate(candidates):
        x = rows[image][kept]
        box, scores = x.split((4, classes), 1)
        confidence, label = scores.max(1, keepdim=True)
        x = torch.cat((box, confidence, label.float()), 1)[confidence.view(-1) > conf_thres]
        if len(x) > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        offset = x[:, 5:6] * max_wh
        keep = torchvision.ops.nms(x[:, :4] + offset, x[:, 4], iou_thres)[:max_det]
        output.append(x[keep])
    return output
