"""Random changes to the images on each generation step, with matching changes to the boxes.

All are off by default. Jitter and flip use one random draw for the whole batch; brightness,
contrast and erasing draw per image. They act on the optimized tensor, before the detector sees it.
"""

import random
from collections.abc import Sequence

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torchvision.transforms import RandomErasing

from dfq.detection import Targets


class AugmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    jitter: int = 0
    flip: bool = False
    brightness: bool = False
    contrast: bool = False
    erase: bool = False


def shift_targets(targets: Targets, dx: int, dy: int, height: int, width: int) -> Targets:
    """Shift boxes, clip them to the image, and drop those left with no area."""
    shift = torch.tensor([dx, dy, dx, dy], dtype=targets.boxes.dtype, device=targets.boxes.device)
    boxes = targets.boxes + shift
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return Targets(boxes=boxes[keep], labels=targets.labels[keep])


def flip_targets(targets: Targets, width: int) -> Targets:
    boxes = targets.boxes.clone()
    boxes[:, 0] = width - targets.boxes[:, 2]
    boxes[:, 2] = width - targets.boxes[:, 0]
    return Targets(boxes=boxes, labels=targets.labels)


def erase_masks(images: Tensor) -> Tensor:
    """Ones with a random rectangle of zeros in half of the images."""
    eraser = RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
    ones = torch.ones(images.shape[1:], device=images.device)
    masks = [eraser(ones) for _ in range(images.shape[0])]
    return torch.stack(masks)


def augment(
    images: Tensor, targets: Sequence[Targets], config: AugmentConfig
) -> tuple[Tensor, list[Targets]]:
    height, width = images.shape[-2:]
    augmented = list(targets)
    if config.jitter:
        dy = random.randint(-config.jitter, config.jitter)
        dx = random.randint(-config.jitter, config.jitter)
        images = torch.roll(images, shifts=(dy, dx), dims=(2, 3))
        augmented = [shift_targets(t, dx, dy, height, width) for t in augmented]
    if config.flip and random.random() > 0.5:
        images = torch.flip(images, dims=(3,))
        augmented = [flip_targets(t, width) for t in augmented]
    if config.brightness:
        images = images + torch.randn(images.shape[0], 1, 1, 1, device=images.device) * 0.2
    if config.contrast:
        images = images * (1.0 + torch.randn(images.shape[0], 1, 1, 1, device=images.device) * 0.1)
    if config.erase:
        images = images * erase_masks(images)
    return images, augmented
