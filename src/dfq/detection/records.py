"""What a detector consumes and produces, for one image.

Boxes are `[x1, y1, x2, y2]` in pixels of the image the detector sees. Labels are contiguous
class indices `0 .. num_classes - 1`; for COCO, `coco_classes.json` maps them to category ids.
These are runtime values that hold tensors, so they are dataclasses, not Pydantic records.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Targets:
    """The ground truth of one image: `boxes` `[N, 4]`, `labels` `[N]`, `masks` `[N, H, W]`."""

    boxes: Tensor
    labels: Tensor
    masks: Tensor | None = None


@dataclass(frozen=True)
class Detections:
    """The predictions for one image: `boxes` `[N, 4]`, `scores` `[N]`, `labels` `[N]`."""

    boxes: Tensor
    scores: Tensor
    labels: Tensor
    masks: Tensor | None = None


def targets_to(targets: Targets, device: torch.device) -> Targets:
    masks = targets.masks.to(device) if targets.masks is not None else None
    return Targets(targets.boxes.to(device), targets.labels.to(device), masks)
