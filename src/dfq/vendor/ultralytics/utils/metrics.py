# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Complete IoU, from ultralytics 8.3.53 (`bbox_iou` in `ultralytics/utils/metrics.py`)."""

import math

import torch
from torch import Tensor


def complete_iou(box1: Tensor, box2: Tensor, eps: float = 1e-7) -> Tensor:
    """CIoU (https://arxiv.org/abs/1911.08287) of corner boxes `[..., 4]`; returns `[..., 1]`."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    inter_w = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0)
    inter_h = (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    inter = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    enclosing_w = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    enclosing_h = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    diagonal = enclosing_w.pow(2) + enclosing_h.pow(2) + eps
    centre_distance = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
    ) / 4
    aspect = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():
        alpha = aspect / (aspect - iou + (1 + eps))
    return iou - (centre_distance / diagonal + aspect * alpha)
