"""Adaptive box sampling: let the boxes of a generated batch follow what the detector sees.

Generation starts from sampled labels, often one box per image. During the sampling window, a
verifier detects objects in the current images. Each confident detection (`score > confidence`)
that overlaps no current box (IoU below `overlap_iou`) becomes a new box, if its area is between
`min_area` and `max_area` of the image. When an image has more than one box, a box that no
confident detection overlaps is dropped, but the best-overlapping box always stays.
"""

from collections.abc import Sequence

import torch
from pydantic import BaseModel, ConfigDict
from torchvision.ops import box_iou

from dfq.detection import Detections, Targets


class BoxSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: int = 800
    stop: int = 2800
    confidence: float = 0.2
    overlap_iou: float = 0.35
    min_area: float = 0.01
    max_area: float = 0.85

    def active(self, iteration: int) -> bool:
        return self.start <= iteration <= self.stop


def kept_boxes(overlap_with_existing: torch.Tensor, threshold: float) -> torch.Tensor:
    """Which existing boxes keep: those some detection overlaps, and at least the best one."""
    best_overlap = overlap_with_existing.max(dim=0).values
    keep = best_overlap >= threshold
    if not keep.any():
        keep[best_overlap.argmax()] = True
    return keep


def resample_image(
    targets: Targets, detections: Detections, image_area: float, config: BoxSamplingConfig
) -> Targets:
    confident = detections.scores > config.confidence
    boxes = detections.boxes[confident].clamp(min=0.0)
    labels = detections.labels[confident]
    if len(boxes) == 0 or len(targets.boxes) == 0:
        return targets
    overlap = box_iou(boxes, targets.boxes)
    is_new = overlap.max(dim=1).values < config.overlap_iou
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights / image_area
    fits = (areas > config.min_area) & (areas < config.max_area)
    added = is_new & fits
    keep = torch.ones(len(targets.boxes), dtype=torch.bool, device=boxes.device)
    if len(targets.boxes) > 1:
        keep = kept_boxes(overlap, config.overlap_iou)
    new_boxes = torch.cat([targets.boxes[keep], boxes[added]])
    new_labels = torch.cat([targets.labels[keep], labels[added].to(targets.labels.dtype)])
    return Targets(boxes=new_boxes, labels=new_labels)


def resample_boxes(
    targets: Sequence[Targets],
    detections: Sequence[Detections],
    image_area: float,
    config: BoxSamplingConfig,
) -> list[Targets]:
    pairs = zip(targets, detections, strict=True)
    return [resample_image(t, d, image_area, config) for t, d in pairs]
