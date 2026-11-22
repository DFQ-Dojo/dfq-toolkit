"""Image priors: losses that pull generated images towards what the detector saw in training.

- `total_variation` penalizes differences between neighbouring pixels.
- `batch_norm_statistics` compares each batch-norm layer's input statistics with its running
  statistics (DeepInversion, https://arxiv.org/abs/1912.08795).
- `patch_similarity_entropy` rewards diverse attention outputs in vision transformers
  (PSAQ-ViT, https://arxiv.org/abs/2203.02250).
"""

import math
from collections.abc import Generator
from contextlib import contextmanager

import torch
from torch import Tensor, nn


def total_variation(images: Tensor) -> tuple[Tensor, Tensor]:
    """Return the L1 and L2 total variation over four neighbour directions."""
    diffs = [
        images[:, :, :, :-1] - images[:, :, :, 1:],
        images[:, :, :-1, :] - images[:, :, 1:, :],
        images[:, :, 1:, :-1] - images[:, :, :-1, 1:],
        images[:, :, :-1, :-1] - images[:, :, 1:, 1:],
    ]
    l1 = torch.stack([diff.abs().mean() for diff in diffs]).sum()
    l2 = torch.stack([torch.norm(diff) for diff in diffs]).sum()
    return l1, l2


def statistics_distance(layer: nn.BatchNorm2d, features: Tensor) -> Tensor:
    """`||running_var - var||_2 + ||running_mean - mean||_2` of the layer's input."""
    assert layer.running_mean is not None and layer.running_var is not None
    mean = features.mean(dim=(0, 2, 3))
    per_channel = features.transpose(0, 1).reshape(features.shape[1], -1)
    variance = per_channel.var(dim=1, unbiased=False)
    var_distance = torch.norm(layer.running_var - variance, 2)
    mean_distance = torch.norm(layer.running_mean - mean, 2)
    return var_distance + mean_distance


@contextmanager
def batch_norm_statistics(network: nn.Module) -> Generator[list[Tensor]]:
    """Collect `statistics_distance` for every `BatchNorm2d`, in the order they run."""
    distances: list[Tensor] = []
    handles = []
    for module in network.modules():
        if isinstance(module, nn.BatchNorm2d):

            def record(layer: nn.Module, inputs: tuple[Tensor, ...], _output: Tensor) -> None:
                assert isinstance(layer, nn.BatchNorm2d)
                distances.append(statistics_distance(layer, inputs[0]))

            handles.append(module.register_forward_hook(record))
    try:
        yield distances
    finally:
        for handle in handles:
            handle.remove()


def gaussian_density(samples: Tensor, points: Tensor, bandwidth: float = 0.01) -> Tensor:
    """Kernel density estimate with a Gaussian kernel: `[B, n]` samples at `[B, m]` points."""
    diffs = points.unsqueeze(2) - samples.unsqueeze(1)
    variance = bandwidth**2
    kernel = torch.exp(-diffs.pow(2) / (2 * variance)) / math.sqrt(2 * math.pi * variance)
    return kernel.mean(dim=-1)


def differential_entropy(density: Tensor, points: Tensor) -> Tensor:
    """Integrate `-p log p` over `points` with the trapezoidal rule, averaged over the batch."""
    shifted = density + 1e-4
    integrand = -shifted * torch.log(shifted)
    return torch.trapezoid(integrand, points, dim=-1).mean()


def patch_similarity_entropy(attention_outputs: list[Tensor], batch_size: int) -> Tensor:
    """The summed entropy of the patch-to-patch cosine similarities of each attention output.

    Each output is `[B', heads, N, D]`; for windowed attention `B'` counts windows. The heads are
    averaged and the first token is dropped, as in PSAQ-ViT. Maximize this to diversify patches.
    """
    entropies = []
    for output in attention_outputs:
        patches = output.mean(dim=1)[:, 1:, :]
        similarity = torch.cosine_similarity(patches.unsqueeze(1), patches.unsqueeze(2), dim=3)
        samples = similarity.reshape(batch_size, -1)
        low = similarity.min().item()
        high = similarity.max().item()
        line = torch.linspace(low, high, steps=10, device=output.device)
        points = line.repeat(batch_size, 1)
        density = gaussian_density(samples, points)
        entropies.append(differential_entropy(density, points))
    return torch.stack(entropies).sum()
