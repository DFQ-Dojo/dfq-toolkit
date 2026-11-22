"""The config records that the TOML files in `configs/` hold, one per script."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from dfq.distill import DistillConfig
from dfq.quant import QuantConfig
from dfq.synthesis import GenerationConfig
from dfq.training import TrainConfig


class Family(StrEnum):
    RESNET50 = "resnet50"


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(Record):
    family: Family
    weights: Path


class CocoData(Record):
    """COCO images and an annotation file; `limit` keeps the first images, as baselines need."""

    kind: Literal["coco"] = "coco"
    images: Path
    annotations: Path
    limit: int | None = None


class SyntheticData(Record):
    """One or more synthetic sets, such as the parts that several GPUs generated."""

    kind: Literal["synthetic"] = "synthetic"
    directories: tuple[Path, ...]
    limit: int | None = None


class ResizeConfig(Record):
    """Mask R-CNN resizing: a random shorter side from `short_sides`, at most `long_side`."""

    short_sides: tuple[int, ...] = (800,)
    long_side: int = 1333
    flip: float = 0.0


class LabelsConfig(Record):
    source: Literal["one_box", "coco"]
    output: Path
    count: int = 0
    num_classes: int = 80
    seed: int = 0
    annotations: Path | None = None


class GenerateConfig(Record):
    """Generate `count` images of `size` (height, width) from a label set.

    To use several GPUs, start one process per GPU with its own `shard_index` and `output`; each
    takes every `shard_count`-th label. A rerun skips the batches its output already holds.
    """

    model: ModelConfig
    labels: Path
    output: Path
    size: tuple[int, int]
    batch_size: int
    count: int
    shard_index: int = 0
    shard_count: int = 1
    seed: int = 0
    device: str = "cuda"
    verify: bool = True
    generation: GenerationConfig


class QatConfig(Record):
    model: ModelConfig
    quant: QuantConfig
    data: Annotated[CocoData | SyntheticData, Field(discriminator="kind")]
    resize: ResizeConfig = ResizeConfig()
    validation: CocoData
    output: Path
    batch_size: int
    workers: int = 4
    device: str = "cuda"
    seed: int = 0
    distill: DistillConfig | None = None
    train: TrainConfig


class EvaluateConfig(Record):
    model: ModelConfig
    quant: QuantConfig | None = None
    checkpoint: Path | None = None
    validation: CocoData
    batch_size: int = 1
    workers: int = 4
    device: str = "cuda"
