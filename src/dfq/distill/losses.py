"""Distillation losses between a student's and a teacher's outputs.

Each loss takes two lists of tensors of the same shapes, one per output, each `[B, ...]`.
The teacher's tensors are treated as constants.
"""

from enum import StrEnum

import torch
import torch.nn.functional as F
from torch import Tensor


class SoftmaxOver(StrEnum):
    """Which values one softmax runs over, per image."""

    EACH_OUTPUT = "each_output"
    ALL_OUTPUTS = "all_outputs"


def flatten_per_image(tensors: list[Tensor]) -> list[Tensor]:
    return [tensor.reshape(tensor.shape[0], -1) for tensor in tensors]


def kl_divergence(
    student: list[Tensor], teacher: list[Tensor], tau: float, softmax_over: SoftmaxOver
) -> Tensor:
    """KL(teacher || student) of softened distributions, scaled by `tau ** 2`, batch-averaged.

    With `EACH_OUTPUT`, each output gets its own softmax and the losses are summed. With
    `ALL_OUTPUTS`, the flattened outputs are joined into one distribution per image.
    """
    student_flat = flatten_per_image(student)
    teacher_flat = flatten_per_image(teacher)
    if softmax_over is SoftmaxOver.ALL_OUTPUTS:
        student_flat = [torch.cat(student_flat, dim=1)]
        teacher_flat = [torch.cat(teacher_flat, dim=1)]
    losses = []
    for student_logits, teacher_logits in zip(student_flat, teacher_flat, strict=True):
        teacher_probs = F.softmax(teacher_logits.detach() / tau, dim=1)
        student_log_probs = F.log_softmax(student_logits / tau, dim=1)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        losses.append(tau**2 * loss)
    return torch.stack(losses).sum()


def pearson_distance(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """Return `1 - mean Pearson correlation` between the rows of `a` and `b`."""
    centered_a = a - a.mean(dim=1, keepdim=True)
    centered_b = b - b.mean(dim=1, keepdim=True)
    dot = (centered_a * centered_b).sum(dim=1)
    norms = centered_a.norm(dim=1) * centered_b.norm(dim=1) + eps
    return 1 - (dot / norms).mean()


def dist_loss(
    student: list[Tensor], teacher: list[Tensor], beta: float, gamma: float, tau: float
) -> Tensor:
    """DIST (https://arxiv.org/abs/2205.10536): inter- and intra-class relation, output-averaged."""
    losses = []
    pairs = zip(flatten_per_image(student), flatten_per_image(teacher), strict=True)
    for student_logits, teacher_logits in pairs:
        student_probs = (student_logits / tau).softmax(dim=1)
        teacher_probs = (teacher_logits.detach() / tau).softmax(dim=1)
        inter = pearson_distance(student_probs, teacher_probs)
        intra = pearson_distance(student_probs.T, teacher_probs.T)
        losses.append(tau**2 * (beta * inter + gamma * intra))
    return torch.stack(losses).mean()


def feature_mse(student: list[Tensor], teacher: list[Tensor]) -> Tensor:
    """The mean over outputs of each output's mean squared difference."""
    losses = [torch.mean((s - t.detach()) ** 2) for s, t in zip(student, teacher, strict=True)]
    return torch.stack(losses).mean()
