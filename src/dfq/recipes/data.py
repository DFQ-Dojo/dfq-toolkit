"""Data loaders for training and validation, with each family's preprocessing.

Mask R-CNN resizes keeping the ratio (a random shorter side when training) and pads with the
pixel mean; YOLO letterboxes to `image_size` and pads with grey (114). Synthetic sets are used as
generated, since generation already made them the detector's input size.
"""

import functools
import random
from collections.abc import Callable, Iterator, Sequence

from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler, Subset

from dfq.common import distributed
from dfq.data.coco import (
    CocoInstances,
    Sample,
    flip_horizontally,
    letterbox,
    pad_batch,
    resize_keep_ratio,
)
from dfq.data.synthetic import SyntheticImages, stack_batch
from dfq.detection import Targets
from dfq.models.maskrcnn.transform import IMAGENET_MEAN
from dfq.recipes.configs import CocoData, ResizeConfig, SyntheticData

Transform = Callable[[Sample], Sample]
YOLO_GREY = (114 / 255, 114 / 255, 114 / 255)


def mask_rcnn_transform(sample: Sample, resize: ResizeConfig) -> Sample:
    short = random.choice(resize.short_sides)
    resized = resize_keep_ratio(sample, short, resize.long_side)
    return flip_horizontally(resized, resize.flip)


def yolo_transform(sample: Sample, resize: ResizeConfig) -> Sample:
    boxed = letterbox(sample, resize.image_size)
    return flip_horizontally(boxed, resize.flip)


class Transformed(Dataset[Sample]):
    def __init__(self, samples: CocoInstances, transform: Transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.transform(self.samples[index])


def collate_pairs(
    samples: Sequence[Sample], fill: tuple[float, float, float]
) -> tuple[Tensor, list[Targets]]:
    images, targets, _ = pad_batch(samples, fill=fill)
    return images, targets


def collate_with_infos(samples: Sequence[Sample], fill: tuple[float, float, float]) -> tuple:
    return pad_batch(samples, fill=fill)


class Epochs:
    """Re-iterable batches that reshuffle a distributed sampler every epoch."""

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self.epoch = 0

    def __iter__(self) -> Iterator:
        if isinstance(self.loader.sampler, DistributedSampler):
            self.loader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)


def loader(
    dataset: Dataset, batch_size: int, workers: int, shuffle: bool, collate: Callable
) -> DataLoader:
    sampler: DistributedSampler | None = None
    if distributed.world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate,
        drop_last=shuffle,
    )


def synthetic_dataset(data: SyntheticData) -> Dataset:
    parts = [SyntheticImages(directory) for directory in data.directories]
    joined: ConcatDataset[tuple[Tensor, Targets]] = ConcatDataset(parts)
    count = len(joined) if data.limit is None else min(data.limit, len(joined))
    return Subset(joined, range(count))


def preprocessing(yolo: bool, resize: ResizeConfig) -> tuple[Transform, tuple[float, float, float]]:
    if yolo:
        return functools.partial(yolo_transform, resize=resize), YOLO_GREY
    return functools.partial(mask_rcnn_transform, resize=resize), IMAGENET_MEAN


def train_batches(
    data: CocoData | SyntheticData, resize: ResizeConfig, yolo: bool, batch_size: int, workers: int
) -> Epochs:
    if isinstance(data, SyntheticData):
        dataset = synthetic_dataset(data)
        return Epochs(loader(dataset, batch_size, workers, shuffle=True, collate=stack_batch))
    samples = CocoInstances(
        data.images, data.annotations, masks=not yolo, skip_empty=True, limit=data.limit
    )
    transform, fill = preprocessing(yolo, resize)
    collate = functools.partial(collate_pairs, fill=fill)
    return Epochs(
        loader(Transformed(samples, transform), batch_size, workers, shuffle=True, collate=collate)
    )


def validation_batches(
    data: CocoData, yolo: bool, batch_size: int, workers: int
) -> tuple[DataLoader, CocoInstances]:
    samples = CocoInstances(
        data.images, data.annotations, masks=False, skip_empty=False, limit=data.limit
    )
    transform, fill = preprocessing(yolo, ResizeConfig())
    collate = functools.partial(collate_with_infos, fill=fill)
    batches = loader(
        Transformed(samples, transform), batch_size, workers, shuffle=False, collate=collate
    )
    return batches, samples
