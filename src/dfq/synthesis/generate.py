"""Generate a batch of calibration images for given boxes, by optimizing the pixels.

The loop optimizes a tensor `z` with Adam. The detector sees `pixels = z * std + mean` per channel,
so `LatentSpace` decides the space the optimizer works in: pixels themselves for YOLO, or
ImageNet-normalized values for Mask R-CNN. Each step minimizes

    detection * (the teacher's losses on the boxes)
    + tv_l1 * TV_L1(z) + tv_l2 * TV_L2(z)
    + batch_norm * sum(BN distances) + first_batch_norm * (first BN distance)
    - attention_entropy * (patch similarity entropy)

on an augmented copy of `z`. Every `verify_every` steps, a verifier (usually the teacher itself)
scores the un-augmented images: the images with the lowest verifier loss are kept, and, inside
the box-sampling window, the boxes are resampled from its detections. The result holds the kept
images with the boxes as they are at the end.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from dfq.detection import Detector, Targets, freeze, generation_mode
from dfq.detection.taps import tap
from dfq.synthesis.augment import AugmentConfig, augment
from dfq.synthesis.box_sampling import BoxSamplingConfig, resample_boxes
from dfq.synthesis.priors import batch_norm_statistics, patch_similarity_entropy, total_variation


class Init(StrEnum):
    UNIFORM_PIXELS = "uniform_pixels"
    NORMAL_PIXELS = "normal_pixels"
    NORMAL_LATENT = "normal_latent"


class Schedule(StrEnum):
    """Cosine decay to zero over the run.

    `cosine` is already decayed by one step at the first step (the scheduler stepped before the
    optimizer, as in YOLOv5 and ResNet); `cosine_after_step` starts at the full rate (YOLO11);
    `warmup_cosine` first rises linearly over `warmup` steps (Swin, DeiT).
    """

    COSINE = "cosine"
    COSINE_AFTER_STEP = "cosine_after_step"
    WARMUP_COSINE = "warmup_cosine"


class LatentSpace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)


class PriorWeights(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detection: float = 1.0
    tv_l1: float = 0.0
    tv_l2: float = 0.0
    batch_norm: float = 0.0
    first_batch_norm: float = 0.0
    attention_entropy: float = 0.0


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    iterations: int
    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    schedule: Schedule = Schedule.COSINE
    warmup: int = 0
    latent: LatentSpace = LatentSpace()
    init: Init = Init.UNIFORM_PIXELS
    weights: PriorWeights = PriorWeights()
    augment: AugmentConfig = AugmentConfig()
    clamp: bool = True
    verify_every: int = 50
    box_sampling: BoxSamplingConfig | None = None


class StepRecord(BaseModel):
    """What one step logged: the weighted terms and, on verifying steps, the verifier loss."""

    model_config = ConfigDict(extra="forbid")

    iteration: int
    lr: float
    terms: dict[str, float]
    verifier_loss: float | None = None
    boxes: int


@dataclass(frozen=True)
class GeneratedBatch:
    images: Tensor
    targets: list[Targets]
    history: list[StepRecord]


def learning_rate(config: GenerationConfig, iteration: int) -> float:
    """The learning rate of 1-based `iteration`."""
    if config.schedule is Schedule.COSINE:
        return 0.5 * config.lr * (1 + math.cos(math.pi * iteration / config.iterations))
    step = iteration - 1
    if config.schedule is Schedule.COSINE_AFTER_STEP:
        return 0.5 * config.lr * (1 + math.cos(math.pi * step / config.iterations))
    if step < config.warmup:
        return config.lr * (step + 1) / config.warmup
    progress = (step - config.warmup) / (config.iterations - config.warmup)
    return 0.5 * config.lr * (1 + math.cos(math.pi * progress))


def channel_tensor(values: tuple[float, float, float], device: torch.device) -> Tensor:
    return torch.tensor(values, device=device).view(1, 3, 1, 1)


def to_pixels(latent: Tensor, space: LatentSpace) -> Tensor:
    mean = channel_tensor(space.mean, latent.device)
    std = channel_tensor(space.std, latent.device)
    return latent * std + mean


def to_latent(pixels: Tensor, space: LatentSpace) -> Tensor:
    mean = channel_tensor(space.mean, pixels.device)
    std = channel_tensor(space.std, pixels.device)
    return (pixels - mean) / std


def initial_latent(
    shape: tuple[int, ...], config: GenerationConfig, device: torch.device
) -> Tensor:
    if config.init is Init.NORMAL_LATENT:
        return torch.randn(shape, device=device)
    if config.init is Init.NORMAL_PIXELS:
        pixels = torch.randn(shape, device=device)
    else:
        pixels = torch.randint(high=255, size=shape, device=device).float() / 255
    return to_latent(pixels, config.latent)


def clamp_to_image(latent: Tensor, space: LatentSpace) -> None:
    """Clamp `latent` in place so that its pixels lie in `[0, 1]`."""
    zeros = torch.zeros(1, 3, 1, 1, device=latent.device)
    low = to_latent(zeros, space)
    high = to_latent(zeros + 1, space)
    latent.data = torch.maximum(torch.minimum(latent.data, high), low)


def step_terms(
    teacher: Detector,
    pixels: Tensor,
    latent: Tensor,
    targets: Sequence[Targets],
    weights: PriorWeights,
) -> dict[str, Tensor]:
    """The weighted loss terms of one step and their sum, `"loss"`."""
    attention_names = teacher.feature_layers("attention") if weights.attention_entropy else []
    with (
        batch_norm_statistics(teacher.network) as distances,
        tap(teacher.network, attention_names) as attention,
    ):
        losses = teacher.losses(pixels, targets)
    detection = torch.stack(list(losses.values())).sum()
    tv_l1, tv_l2 = total_variation(latent)
    terms = {"detection": weights.detection * detection}
    terms["tv_l1"] = weights.tv_l1 * tv_l1
    terms["tv_l2"] = weights.tv_l2 * tv_l2
    if distances:
        terms["batch_norm"] = weights.batch_norm * torch.stack(distances).sum()
        terms["first_batch_norm"] = weights.first_batch_norm * distances[0]
    if attention_names:
        outputs = [attention[name] for name in attention_names]
        tensors = [output for output in outputs if isinstance(output, Tensor)]
        entropy = patch_similarity_entropy(tensors, batch_size=pixels.shape[0])
        terms["attention_entropy"] = -weights.attention_entropy * entropy
    terms["loss"] = torch.stack(list(terms.values())).sum()
    return terms


def verifier_loss(verifier: Detector, pixels: Tensor, targets: Sequence[Targets]) -> float:
    generation_mode(verifier.network)
    with torch.no_grad():
        losses = verifier.losses(pixels, targets)
    total = torch.stack(list(losses.values())).sum()
    return total.item()


def sampled_boxes(
    verifier: Detector, pixels: Tensor, targets: list[Targets], config: BoxSamplingConfig
) -> list[Targets]:
    verifier.network.eval()
    with torch.no_grad():
        detections = verifier.predict(pixels)
    image_area = float(pixels.shape[-1] * pixels.shape[-2])
    return resample_boxes(targets, detections, image_area, config)


def generate_batch(
    teacher: Detector,
    targets: Sequence[Targets],
    size: tuple[int, int],
    config: GenerationConfig,
    verifier: Detector | None = None,
) -> GeneratedBatch:
    """Generate `len(targets)` images of `size` (height, width) for `targets`, in pixels."""
    device = next(teacher.network.parameters()).device
    freeze(teacher.network)
    shape = (len(targets), 3, *size)
    latent = initial_latent(shape, config, device).requires_grad_()
    optimizer = torch.optim.Adam([latent], lr=config.lr, betas=config.betas, eps=1e-8)
    current = list(targets)
    best = latent.detach().clone()
    best_loss = math.inf
    history = []
    for iteration in range(1, config.iterations + 1):
        lr = learning_rate(config, iteration)
        for group in optimizer.param_groups:
            group["lr"] = lr
        generation_mode(teacher.network)
        augmented, augmented_targets = augment(latent, current, config.augment)
        pixels = to_pixels(augmented, config.latent)
        terms = step_terms(teacher, pixels, augmented, augmented_targets, config.weights)
        checked = None
        if verifier is not None and iteration % config.verify_every == 0:
            plain = to_pixels(latent.detach(), config.latent)
            checked = verifier_loss(verifier, plain, current)
            if config.box_sampling is not None and config.box_sampling.active(iteration):
                current = sampled_boxes(verifier, plain, current, config.box_sampling)
            if checked < best_loss or iteration < 100:
                best = latent.detach().clone()
                best_loss = checked
        optimizer.zero_grad()
        terms["loss"].backward()
        optimizer.step()
        if config.clamp:
            clamp_to_image(latent, config.latent)
        logged = {name: value.item() for name, value in terms.items()}
        boxes = sum(len(t.boxes) for t in current)
        history.append(
            StepRecord(iteration=iteration, lr=lr, terms=logged, verifier_loss=checked, boxes=boxes)
        )
    kept = best if verifier is not None else latent.detach()
    images = to_pixels(kept, config.latent)
    return GeneratedBatch(images=images, targets=current, history=history)
