"""Label sets: the boxes that image generation starts from, saved as JSON Lines.

A label set holds one `ImageLabels` record per image. Boxes are normalized to the image size,
`[x1, y1, x2, y2]` in `[0, 1]`, so one label set serves any generation size. There are three
sources:

- `one_box_labels` samples one random box per image, as the paper's adaptive sampling starts;
- `coco_labels` reads the boxes of a COCO annotation file ("real labels");
- `from_targets` keeps the boxes a generation run ended with ("sampled labels").
"""

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from dfq.detection import Targets
from dfq.detection.coco import coco_classes


class Box(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: int
    x1: float
    y1: float
    x2: float
    y2: float


class ImageLabels(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    image_id: str
    boxes: list[Box]


def write_label_set(path: Path, items: Iterable[ImageLabels]) -> None:
    lines = [item.model_dump_json() for item in items]
    path.write_text("\n".join(lines) + "\n")


def read_label_set(path: Path) -> list[ImageLabels]:
    lines = path.read_text().splitlines()
    return [ImageLabels.model_validate_json(line) for line in lines if line]


def to_targets(labels: ImageLabels, height: int, width: int) -> Targets:
    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    corners = [[box.x1, box.y1, box.x2, box.y2] for box in labels.boxes]
    boxes = torch.tensor(corners, dtype=torch.float32).reshape(-1, 4) * scale
    classes = torch.tensor([box.label for box in labels.boxes], dtype=torch.int64)
    return Targets(boxes=boxes, labels=classes)


def from_targets(targets: Targets, height: int, width: int, image_id: str) -> ImageLabels:
    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    normalized = (targets.boxes.detach().cpu().float() / scale).tolist()
    classes = targets.labels.tolist()
    boxes = [
        Box(label=label, x1=x1, y1=y1, x2=x2, y2=y2)
        for label, (x1, y1, x2, y2) in zip(classes, normalized, strict=True)
    ]
    return ImageLabels(image_id=image_id, boxes=boxes)


def one_box(rng: np.random.RandomState, num_classes: int, min_side: float, max_side: float) -> Box:
    """One box whose sides are uniform in `[min_side, max_side]`, shifted inside the image."""
    width = min_side + (max_side - min_side) * rng.rand()
    height = min_side + (max_side - min_side) * rng.rand()
    center_x = 0.5 + (0.5 - width / 2.0) * rng.rand() * rng.choice([1, -1])
    center_y = 0.5 + (0.5 - height / 2.0) * rng.rand() * rng.choice([1, -1])
    label = int(rng.choice(np.arange(num_classes)))
    return Box(
        label=label,
        x1=center_x - width / 2,
        y1=center_y - height / 2,
        x2=center_x + width / 2,
        y2=center_y + height / 2,
    )


def one_box_labels(
    count: int, num_classes: int, seed: int, min_side: float = 0.2, max_side: float = 0.8
) -> list[ImageLabels]:
    """`count` images with one random box each.

    The draws match the old `generate_one_label.py --seed seed`, which seeded NumPy with
    `seed + 1`.
    """
    rng = np.random.RandomState(seed + 1)
    return [
        ImageLabels(
            image_id=f"{index + 1:012d}", boxes=[one_box(rng, num_classes, min_side, max_side)]
        )
        for index in range(count)
    ]


def coco_labels(annotation_file: Path) -> list[ImageLabels]:
    """The non-crowd boxes of each image in a COCO annotation file that has any, in file order."""
    data = json.loads(annotation_file.read_text())
    index_of = {item.coco_id: item.index for item in coco_classes()}
    sizes = {image["id"]: (image["width"], image["height"]) for image in data["images"]}
    boxes: dict[int, list[Box]] = {image_id: [] for image_id in sizes}
    for annotation in data["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        width, height = sizes[annotation["image_id"]]
        x, y, w, h = annotation["bbox"]
        if w <= 0 or h <= 0:
            continue
        box = Box(
            label=index_of[annotation["category_id"]],
            x1=x / width,
            y1=y / height,
            x2=(x + w) / width,
            y2=(y + h) / height,
        )
        boxes[annotation["image_id"]].append(box)
    return [
        ImageLabels(image_id=str(image_id), boxes=found)
        for image_id, found in boxes.items()
        if found
    ]
