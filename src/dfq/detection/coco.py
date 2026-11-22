"""The 80 COCO classes, read from `coco_classes.json`, the one place that lists them."""

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources


@dataclass(frozen=True)
class CocoClass:
    index: int
    coco_id: int
    name: str


@cache
def coco_classes() -> tuple[CocoClass, ...]:
    text = resources.files("dfq.detection").joinpath("coco_classes.json").read_text()
    records = json.loads(text)["classes"]
    return tuple(CocoClass(**record) for record in records)


def coco_names() -> list[str]:
    return [item.name for item in coco_classes()]
