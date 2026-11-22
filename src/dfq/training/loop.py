"""The quantization-aware training loop.

`train` runs `epochs` passes over `batches`. Each step calls `objective(images, targets)`, which
returns loss terms including their sum, `"loss"`, and backpropagates the sum. Gradients
accumulate until the nominal batch size is reached, then the optimizer steps, after averaging
gradients over processes and clipping their norm. After each epoch, `evaluate` scores the EMA
network (or the network, without EMA); the best and last weights are saved in `run_dir`, and
training stops once the score has not improved for `patience` epochs.
"""

import math
from collections.abc import Callable, Iterable, Sequence, Sized
from enum import StrEnum
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn

from dfq.common import distributed
from dfq.detection import Detector, Targets, targets_to
from dfq.training.ema import averaged_copy, update_average
from dfq.training.optim import (
    OptimizerConfig,
    ScheduleConfig,
    WarmupConfig,
    build_optimizer,
    epoch_factor,
    set_rates,
)

Objective = Callable[[Tensor, Sequence[Targets]], dict[str, Tensor]]
Evaluate = Callable[[nn.Module], float]
Batch = tuple[Tensor, list[Targets]]


class ProcessReduction(StrEnum):
    """How the gradients of several processes combine: `mean`, or `sum` as YOLO does."""

    MEAN = "mean"
    SUM = "sum"


class EmaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decay: float = 0.9999
    tau: float = 2000.0


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int
    optimizer: OptimizerConfig
    schedule: ScheduleConfig = ScheduleConfig()
    warmup: WarmupConfig = WarmupConfig()
    nominal_batch_size: int | None = None
    grad_clip: float | None = None
    ema: EmaConfig | None = None
    amp: bool = False
    patience: int | None = None
    process_reduction: ProcessReduction = ProcessReduction.MEAN


class EpochRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epoch: int
    lr: float
    losses: dict[str, float]
    score: float | None


def detection_objective(detector: Detector) -> Objective:
    """The detector's own losses and their sum, for training without distillation."""

    def objective(images: Tensor, targets: Sequence[Targets]) -> dict[str, Tensor]:
        losses = detector.losses(images, targets)
        total = torch.stack(list(losses.values())).sum()
        return {**losses, "loss": total}

    return objective


def accumulation(step: int, batch_size: int, config: TrainConfig) -> int:
    """How many batches to accumulate before an optimizer step; it grows during warm-up."""
    if config.nominal_batch_size is None:
        return 1
    full = config.nominal_batch_size / (batch_size * distributed.world_size())
    if step < config.warmup.iterations:
        full = 1 + (full - 1) * step / config.warmup.iterations
    return max(1, round(full))


def optimizer_step(
    network: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
) -> None:
    distributed.average_gradients(network)
    if config.process_reduction is ProcessReduction.SUM:
        for parameter in network.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(distributed.world_size())
    scaler.unscale_(optimizer)
    if config.grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()


def save(network: nn.Module, path: Path) -> None:
    if distributed.is_main():
        torch.save(network.state_dict(), path)


def train(
    network: nn.Module,
    objective: Objective,
    batches: Iterable[Batch],
    config: TrainConfig,
    evaluate: Evaluate | None,
    run_dir: Path,
) -> list[EpochRecord]:
    """Train `network` in place; return one record per epoch. `batches` is iterated per epoch."""
    per_epoch = len(batches) if isinstance(batches, Sized) else 0
    warmup_iterations = max(config.warmup.iterations, round(config.warmup.epochs * per_epoch))
    warmup = config.warmup.model_copy(update={"iterations": warmup_iterations})
    config = config.model_copy(update={"warmup": warmup})
    device = next(network.parameters()).device
    optimizer = build_optimizer(network, config.optimizer)
    scaler = torch.amp.GradScaler(device.type, enabled=config.amp)
    average = averaged_copy(network) if config.ema is not None else None
    records: list[EpochRecord] = []
    best_score = -math.inf
    best_epoch = 0
    step = 0
    updates = 0
    since_update = 0
    for epoch in range(config.epochs):
        target_lr = config.optimizer.lr * epoch_factor(config.schedule, epoch, config.epochs)
        network.train()
        sums: dict[str, float] = {}
        count = 0
        for images, targets in batches:
            set_rates(optimizer, config.optimizer, config.warmup, target_lr, step)
            on_device = [targets_to(t, device) for t in targets]
            with torch.autocast(device.type, enabled=config.amp):
                terms = objective(images.to(device), on_device)
            scaler.scale(terms["loss"]).backward()
            since_update += 1
            if since_update >= accumulation(step, images.shape[0], config):
                optimizer_step(network, optimizer, scaler, config)
                since_update = 0
                updates += 1
                if average is not None and config.ema is not None:
                    update_average(average, network, updates, config.ema.decay, config.ema.tau)
            for name, value in terms.items():
                sums[name] = sums.get(name, 0.0) + value.item()
            count += 1
            step += 1
        evaluated = average if average is not None else network
        score = evaluate(evaluated) if evaluate is not None else None
        means = {name: total / max(count, 1) for name, total in sums.items()}
        records.append(EpochRecord(epoch=epoch, lr=target_lr, losses=means, score=score))
        save(evaluated, run_dir / "last.pt")
        if score is not None and score > best_score:
            best_score, best_epoch = score, epoch
            save(evaluated, run_dir / "best.pt")
        if config.patience is not None and epoch - best_epoch >= config.patience:
            break
    return records
