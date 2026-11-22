"""COCO instances for Mask R-CNN: loading, resizing, flipping and batching.

`CocoInstances` reads images and annotations with pycocotools. `resize_keep_ratio` scales an
image so that it fits `short x long` (800 x 1333 for COCO), as MMDetection's `Resize` does.
`pad_batch` pads images to a multiple of 32 with a fill colour; pass the network's pixel mean so
that padding normalizes to zero, as MMDetection's `Normalize` then `Pad` did.
"""

import contextlib
import io
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor
from torch.utils.data import Dataset

from dfq.detection import Targets
from dfq.detection.coco import coco_classes


@dataclass(frozen=True)
class ImageInfo:
    """Where an image came from: its COCO id, original size, the scale it was resized by, and
    the padding added on its left and top (letterboxing)."""

    image_id: int
    height: int
    width: int
    scale: float = 1.0
    left: float = 0.0
    top: float = 0.0


Sample = tuple[Tensor, Targets, ImageInfo]


def annotation_targets(coco: COCO, image_id: int, height: int, width: int, masks: bool) -> Targets:
    index_of = {item.coco_id: item.index for item in coco_classes()}
    annotation_ids = coco.getAnnIds(imgIds=image_id, iscrowd=False)
    annotations = [
        a for a in coco.loadAnns(annotation_ids) if a["bbox"][2] > 0 and a["bbox"][3] > 0
    ]
    corners = [[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in annotations)]
    boxes = torch.tensor(corners, dtype=torch.float32).reshape(-1, 4)
    labels = torch.tensor([index_of[a["category_id"]] for a in annotations], dtype=torch.int64)
    if not masks:
        return Targets(boxes=boxes, labels=labels)
    bitmaps = [coco.annToMask(a) for a in annotations]
    stacked = np.stack(bitmaps) if bitmaps else np.zeros((0, height, width), dtype=np.uint8)
    return Targets(boxes=boxes, labels=labels, masks=torch.from_numpy(stacked))


class CocoInstances(Dataset[Sample]):
    """COCO images as float RGB in `[0, 1]`, with their boxes (and masks).

    For training, `skip_empty` drops images without boxes and `limit` keeps the first images, as
    the 2k-image baselines need.
    """

    def __init__(
        self,
        images: Path,
        annotations: Path,
        masks: bool,
        skip_empty: bool,
        limit: int | None = None,
    ) -> None:
        self.images = images
        with contextlib.redirect_stdout(io.StringIO()):
            self.coco = COCO(str(annotations))
        self.masks = masks
        ids = sorted(self.coco.getImgIds())
        if skip_empty:
            ids = [i for i in ids if self.coco.getAnnIds(imgIds=i, iscrowd=False)]
        self.ids = ids[:limit]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Sample:
        image_id = self.ids[index]
        record = self.coco.loadImgs(image_id)[0]
        with Image.open(self.images / record["file_name"]) as file:
            rgb = np.asarray(file.convert("RGB"), dtype=np.float32) / 255.0
        image = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        height, width = image.shape[1:]
        targets = annotation_targets(self.coco, image_id, height, width, self.masks)
        return image, targets, ImageInfo(image_id=image_id, height=height, width=width)


def resize_keep_ratio(sample: Sample, short: int, long: int) -> Sample:
    image, targets, info = sample
    height, width = image.shape[1:]
    scale = min(short / min(height, width), long / max(height, width))
    new_height = int(height * scale + 0.5)
    new_width = int(width * scale + 0.5)
    resized = F.interpolate(
        image[None], size=(new_height, new_width), mode="bilinear", align_corners=False
    )[0]
    factors = torch.tensor([new_width / width, new_height / height] * 2)
    masks = targets.masks
    if masks is not None and len(masks):
        masks = F.interpolate(masks[None].float(), size=(new_height, new_width), mode="nearest")[
            0
        ].to(torch.uint8)
    elif masks is not None:
        masks = masks.new_zeros((0, new_height, new_width))
    scaled = Targets(boxes=targets.boxes * factors, labels=targets.labels, masks=masks)
    resized_info = ImageInfo(info.image_id, info.height, info.width, scale=scale)
    return resized, scaled, resized_info


def letterbox(sample: Sample, size: int, stride: int = 32, fill: float = 114 / 255) -> Sample:
    """Fit the image into `size x size` keeping its ratio, then pad it evenly on both sides up to
    a multiple of `stride`, as YOLO validation does for one image per batch."""
    image, targets, info = sample
    height, width = image.shape[1:]
    scale = min(size / height, size / width)
    new_height = round(height * scale)
    new_width = round(width * scale)
    resized = F.interpolate(
        image[None], size=(new_height, new_width), mode="bilinear", align_corners=False
    )[0]
    padded_height = math.ceil(new_height / stride) * stride
    padded_width = math.ceil(new_width / stride) * stride
    top = (padded_height - new_height) / 2
    left = (padded_width - new_width) / 2
    canvas = torch.full((3, padded_height, padded_width), fill)
    row, column = round(top - 0.1), round(left - 0.1)
    canvas[:, row : row + new_height, column : column + new_width] = resized
    shift = torch.tensor([column, row, column, row], dtype=torch.float32)
    boxes = targets.boxes * scale + shift
    moved = Targets(boxes=boxes, labels=targets.labels)
    return (
        canvas,
        moved,
        ImageInfo(info.image_id, info.height, info.width, scale, float(column), float(row)),
    )


def flip_horizontally(sample: Sample, probability: float) -> Sample:
    image, targets, info = sample
    if random.random() >= probability:
        return sample
    width = image.shape[-1]
    boxes = targets.boxes.clone()
    boxes[:, 0] = width - targets.boxes[:, 2]
    boxes[:, 2] = width - targets.boxes[:, 0]
    masks = targets.masks.flip(-1) if targets.masks is not None else None
    return image.flip(-1), Targets(boxes=boxes, labels=targets.labels, masks=masks), info


def pad_batch(
    samples: Sequence[Sample], fill: tuple[float, float, float], divisor: int = 32
) -> tuple[Tensor, list[Targets], list[ImageInfo]]:
    """Pad images to a shared size, a multiple of `divisor`; masks are padded with zeros."""
    height = max(image.shape[1] for image, _, _ in samples)
    width = max(image.shape[2] for image, _, _ in samples)
    height = (height + divisor - 1) // divisor * divisor
    width = (width + divisor - 1) // divisor * divisor
    colour = torch.tensor(fill).view(3, 1, 1)
    batch = colour.expand(len(samples), 3, height, width).clone()
    targets = []
    for index, (image, target, _) in enumerate(samples):
        batch[index, :, : image.shape[1], : image.shape[2]] = image
        masks = target.masks
        if masks is not None:
            masks = F.pad(masks, (0, width - masks.shape[2], 0, height - masks.shape[1]))
        targets.append(Targets(boxes=target.boxes, labels=target.labels, masks=masks))
    infos = [info for _, _, info in samples]
    return batch, targets, infos
