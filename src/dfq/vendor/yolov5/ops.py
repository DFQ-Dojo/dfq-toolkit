# Ultralytics YOLOv5, AGPL-3.0 license
"""YOLOv5's NMS, from YOLOv5 v7.0 (`utils/general.py`): class scores times objectness, best class
per box, class-aware NMS. Merging, label inputs and the time limit are removed."""

import torch
import torchvision
from torch import Tensor

from dfq.vendor.ultralytics.utils.ops import xywh2xyxy


def non_max_suppression(
    prediction: Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    max_nms: int = 30000,
    max_wh: int = 7680,
) -> list[Tensor]:
    """Per image, `[N, 6]` rows of (x1, y1, x2, y2, score, class) from `[B, anchors, 5 + nc]`."""
    candidates = prediction[..., 4] > conf_thres
    output = []
    for image, kept in enumerate(candidates):
        x = prediction[image][kept]  # boolean indexing copies, so scaling below is safe
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        confidence, label = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, confidence, label.float()), 1)[confidence.view(-1) > conf_thres]
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        offset = x[:, 5:6] * max_wh
        keep = torchvision.ops.nms(x[:, :4] + offset, x[:, 4], iou_thres)[:max_det]
        output.append(x[keep])
    return output
