import torch
from torch.utils.data import distributed

from ultralytics.utils import RANK
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.dataset import YOLODataset

from pathlib import Path
from external.data import get_generated_dataset


def build_dataloader(dataset, batch_size, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    sampler = (
        None
        if rank == -1
        else distributed.DistributedSampler(dataset, shuffle=shuffle)
    )
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=0,  # TODO: check this
        sampler=sampler,
        pin_memory=True,
        collate_fn=YOLODataset.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_dataloader_from_gimg(
    pseudo_data_path: Path,
    calibration_size: int,
    batch_size: int,
    shuffle: bool,
    rank: int,
):
    dataset = get_generated_dataset(pseudo_data_path, calibration_size)
    return build_dataloader(dataset, batch_size, shuffle, rank)
