"""Synthetic image sets: the output of image generation and the input of distillation.

A set is a directory:

    manifest.json        SyntheticManifest: image size, batch files, image count
    batch_00000.pt       {"images": float [B, 3, H, W] in [0, 1], "boxes": [...], "labels": [...]}
    labels.jsonl         the final boxes as a label set, to start another generation from

Boxes in batch files are in pixels; `labels.jsonl` holds them normalized.
"""

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.utils.data import Dataset

from dfq.data.labels import ImageLabels, from_targets, read_label_set, write_label_set
from dfq.detection import Targets

MANIFEST = "manifest.json"


class SyntheticManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    height: int
    width: int
    batches: list[str] = []
    counts: list[int] = []


def read_manifest(directory: Path) -> SyntheticManifest:
    return SyntheticManifest.model_validate_json((directory / MANIFEST).read_text())


def write_manifest(directory: Path, manifest: SyntheticManifest) -> None:
    (directory / MANIFEST).write_text(manifest.model_dump_json(indent=2))


def append_batch(directory: Path, images: Tensor, targets: Sequence[Targets]) -> SyntheticManifest:
    """Save one generated batch and record it in the manifest, which must exist."""
    manifest = read_manifest(directory)
    name = f"batch_{len(manifest.batches):05d}.pt"
    record = {
        "images": images.detach().cpu(),
        "boxes": [t.boxes.detach().cpu() for t in targets],
        "labels": [t.labels.detach().cpu() for t in targets],
    }
    torch.save(record, directory / name)
    updated = manifest.model_copy(
        update={"batches": [*manifest.batches, name], "counts": [*manifest.counts, len(targets)]}
    )
    write_manifest(directory, updated)
    start = sum(manifest.counts)
    labels = [
        from_targets(t, manifest.height, manifest.width, image_id=f"{start + i:012d}")
        for i, t in enumerate(targets)
    ]
    label_file = directory / "labels.jsonl"
    previous = read_label_set(label_file) if label_file.exists() else []
    write_label_set(label_file, [*previous, *labels])
    return updated


def start_set(directory: Path, height: int, width: int) -> SyntheticManifest:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = SyntheticManifest(height=height, width=width)
    write_manifest(directory, manifest)
    return manifest


@lru_cache(maxsize=2)
def load_batch(path: Path) -> dict[str, object]:
    record: dict[str, object] = torch.load(path, map_location="cpu", weights_only=True)
    return record


class SyntheticImages(Dataset[tuple[Tensor, Targets]]):
    """The images and boxes of a synthetic set; `limit` keeps only the first images."""

    def __init__(self, directory: Path, limit: int | None = None) -> None:
        self.directory = directory
        manifest = read_manifest(directory)
        self.locations = [
            (name, offset)
            for name, count in zip(manifest.batches, manifest.counts, strict=True)
            for offset in range(count)
        ][:limit]

    def __len__(self) -> int:
        return len(self.locations)

    def __getitem__(self, index: int) -> tuple[Tensor, Targets]:
        name, offset = self.locations[index]
        record = load_batch(self.directory / name)
        images, boxes, labels = record["images"], record["boxes"], record["labels"]
        assert isinstance(images, Tensor) and isinstance(boxes, list) and isinstance(labels, list)
        return images[offset], Targets(boxes=boxes[offset], labels=labels[offset])


def stack_batch(samples: Sequence[tuple[Tensor, Targets]]) -> tuple[Tensor, list[Targets]]:
    """Collate same-sized images into one tensor and keep the targets as a list."""
    images = torch.stack([image for image, _ in samples])
    targets = [target for _, target in samples]
    return images, targets


def label_set_of(directory: Path) -> list[ImageLabels]:
    return read_label_set(directory / "labels.jsonl")
