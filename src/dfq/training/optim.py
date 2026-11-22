"""The optimizer, its parameter groups, and the learning-rate and momentum schedule.

Parameters fall into three groups: weights with weight decay; normalization weights and weights
whose name matches a `no_decay` pattern, without decay; and biases, without decay. With
`decay_biases_and_norms`, biases and normalization weights decay too, as plain SGD does. YOLO
warms the bias group up from a high learning rate, so the bias group comes first.

The learning rate of iteration `i` in epoch `e` is `lr * epoch_factor(e)`. During the warm-up
iterations it rises linearly from `start_factor * lr` (or from `bias_start_lr`
for biases) to that value, and the momentum rises from `start_momentum` to `momentum`.
"""

import math
from enum import StrEnum
from fnmatch import fnmatchcase

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.nn.modules.batchnorm import _NormBase


class OptimizerName(StrEnum):
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: OptimizerName
    lr: float
    momentum: float = 0.9
    nesterov: bool = False
    weight_decay: float = 0.0
    no_decay: tuple[str, ...] = ()
    decay_biases_and_norms: bool = False


class ScheduleKind(StrEnum):
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    STEP = "step"


class ScheduleConfig(BaseModel):
    """`linear` and `cosine` go from 1 to `final_factor` over the run; `step` multiplies by
    `gamma` at each epoch in `milestones`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ScheduleKind = ScheduleKind.CONSTANT
    final_factor: float = 0.01
    milestones: tuple[int, ...] = ()
    gamma: float = 0.1


class WarmupConfig(BaseModel):
    """Warm up for `max(iterations, epochs x batches per epoch)` iterations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    iterations: int = 0
    epochs: float = 0.0
    start_factor: float = 0.0
    bias_start_lr: float | None = None
    start_momentum: float | None = None


def is_norm(module: nn.Module) -> bool:
    return isinstance(module, _NormBase | nn.LayerNorm | nn.GroupNorm)


def parameter_groups(
    network: nn.Module, no_decay: tuple[str, ...], decay_biases_and_norms: bool = False
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    """Split trainable parameters into (biases, decayed weights, other weights)."""
    biases, decayed, undecayed = [], [], []
    for module_name, module in network.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            exempt = any(fnmatchcase(name, pattern) for pattern in no_decay)
            if parameter_name == "bias":
                biases.append(parameter)
            elif exempt or (is_norm(module) and not decay_biases_and_norms):
                undecayed.append(parameter)
            else:
                decayed.append(parameter)
    return biases, decayed, undecayed


def build_optimizer(network: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    biases, decayed, undecayed = parameter_groups(
        network, config.no_decay, config.decay_biases_and_norms
    )
    bias_decay = config.weight_decay if config.decay_biases_and_norms else 0.0
    groups = [
        {"params": biases, "weight_decay": bias_decay},
        {"params": decayed, "weight_decay": config.weight_decay},
        {"params": undecayed, "weight_decay": 0.0},
    ]
    if config.name is OptimizerName.SGD:
        return torch.optim.SGD(
            groups, lr=config.lr, momentum=config.momentum, nesterov=config.nesterov
        )
    betas = (config.momentum, 0.999)
    if config.name is OptimizerName.ADAM:
        return torch.optim.Adam(groups, lr=config.lr, betas=betas)
    return torch.optim.AdamW(groups, lr=config.lr, betas=betas)


def epoch_factor(config: ScheduleConfig, epoch: int, epochs: int) -> float:
    """The learning-rate factor of 0-based `epoch`."""
    if config.kind is ScheduleKind.LINEAR:
        remaining = max(1 - epoch / epochs, 0)
        return remaining * (1.0 - config.final_factor) + config.final_factor
    if config.kind is ScheduleKind.COSINE:
        rise = (1 - math.cos(epoch * math.pi / epochs)) / 2
        return rise * (config.final_factor - 1) + 1
    if config.kind is ScheduleKind.STEP:
        passed = sum(1 for milestone in config.milestones if epoch >= milestone)
        return config.gamma**passed
    return 1.0


def interpolate(step: int, steps: int, start: float, end: float) -> float:
    fraction = min(step / steps, 1.0)
    return start + (end - start) * fraction


def set_rates(
    optimizer: torch.optim.Optimizer,
    optimizer_config: OptimizerConfig,
    warmup: WarmupConfig,
    target_lr: float,
    step: int,
) -> None:
    """Set each group's learning rate (and momentum while warming up) for global `step`."""
    warming = step < warmup.iterations
    for index, group in enumerate(optimizer.param_groups):
        start = warmup.start_factor * target_lr
        if index == 0 and warmup.bias_start_lr is not None:
            start = warmup.bias_start_lr
        group["lr"] = (
            interpolate(step, warmup.iterations, start, target_lr) if warming else target_lr
        )
        if warmup.start_momentum is None:
            continue
        momentum = optimizer_config.momentum
        if warming:
            momentum = interpolate(step, warmup.iterations, warmup.start_momentum, momentum)
        if "betas" in group:
            group["betas"] = (momentum, group["betas"][1])
        else:
            group["momentum"] = momentum
