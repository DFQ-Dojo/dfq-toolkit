"""Data loaders for Mask R-CNN training and validation."""

import random
from collections.abc import Iterator, Sequence

from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler, Subset

from dfq.common import distributed
from dfq.data.coco import CocoInstances, Sample, flip_horizontally, pad_batch, resize_keep_ratio
from dfq.data.synthetic import SyntheticImages, stack_batch
from dfq.detection import Targets
from dfq.models.maskrcnn.transform import IMAGENET_MEAN
from dfq.recipes.configs import CocoData, ResizeConfig, SyntheticData


class Resized(Dataset[Sample]):
    def __init__(self, samples: CocoInstances, resize: ResizeConfig) -> None:
        self.samples = samples
        self.resize = resize

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        short = random.choice(self.resize.short_sides)
        resized = resize_keep_ratio(self.samples[index], short, self.resize.long_side)
        return flip_horizontally(resized, self.resize.flip)


def train_pairs(samples: Sequence[Sample]) -> tuple[Tensor, list[Targets]]:
    images, targets, _ = pad_batch(samples, fill=IMAGENET_MEAN)
    return images, targets


def validation_batch(samples: Sequence[Sample]) -> tuple[Tensor, list[Targets], list]:
    return pad_batch(samples, fill=IMAGENET_MEAN)


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


def loader(dataset: Dataset, batch_size: int, workers: int, shuffle: bool, collate) -> DataLoader:
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


def train_batches(
    data: CocoData | SyntheticData, resize: ResizeConfig, batch_size: int, workers: int
) -> Epochs:
    if isinstance(data, SyntheticData):
        dataset = synthetic_dataset(data)
        return Epochs(loader(dataset, batch_size, workers, shuffle=True, collate=stack_batch))
    samples = CocoInstances(
        data.images, data.annotations, masks=True, skip_empty=True, limit=data.limit
    )
    resized = Resized(samples, resize)
    return Epochs(loader(resized, batch_size, workers, shuffle=True, collate=train_pairs))


def validation_batches(
    data: CocoData, batch_size: int, workers: int
) -> tuple[DataLoader, CocoInstances]:
    samples = CocoInstances(
        data.images, data.annotations, masks=False, skip_empty=False, limit=data.limit
    )
    resize = ResizeConfig()
    resized = Resized(samples, resize)
    batches = loader(resized, batch_size, workers, shuffle=False, collate=validation_batch)
    return batches, samples
