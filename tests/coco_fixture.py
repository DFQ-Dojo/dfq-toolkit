"""A tiny COCO dataset written to a temporary directory: two images with three objects."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

# (annotation id, image id, category id, [x, y, width, height])
ANNOTATIONS = [
    (1, 1, 1, (10, 20, 30, 40)),
    (2, 1, 3, (50, 10, 20, 20)),
    (3, 2, 90, (0, 0, 40, 30)),
]


def write_coco(root: Path) -> tuple[Path, Path]:
    images = root / "images"
    images.mkdir()
    sizes = {1: (100, 80), 2: (60, 120)}
    for image_id, (width, height) in sizes.items():
        pixels = np.full((height, width, 3), 40 * image_id, dtype=np.uint8)
        Image.fromarray(pixels).save(images / f"{image_id:012d}.png")
    annotations = []
    for annotation_id, image_id, category_id, (x, y, w, h) in ANNOTATIONS:
        polygon = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "iscrowd": 0,
            "area": w * h,
            "segmentation": polygon,
        }
        annotations.append(annotation)
    data = {
        "images": [
            {"id": i, "file_name": f"{i:012d}.png", "width": w, "height": h}
            for i, (w, h) in sizes.items()
        ],
        "annotations": annotations,
        "categories": [{"id": i, "name": str(i)} for i in (1, 3, 90)],
    }
    annotation_file = root / "instances.json"
    annotation_file.write_text(json.dumps(data))
    return images, annotation_file
