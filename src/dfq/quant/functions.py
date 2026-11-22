"""Fake quantization with a learned step size.

Two methods share one autograd function:

- LSQ (Learned Step-size Quantization, https://arxiv.org/abs/1902.08153) quantizes
  `x` to `round(clamp(x / s, qn, qp)) * s`. Rounding is PyTorch's round-half-to-even.
- LSQ+ (https://arxiv.org/abs/2004.09576) adds a learned offset `beta`:
  `round(clamp((x - beta) / s, qn, qp)) * s + beta`. Rounding is half away from zero.

The backward pass follows both papers. The input gets a straight-through gradient inside the
clamp range and zero outside it. The step size gets `g * (round(q) - q)` inside the range and
`g * qn` or `g * qp` outside, where `q = (x - beta) / s` and `g = 1 / sqrt(numel(x) * qp)`.
The offset gets `g` outside the range and zero inside. The caller passes `g`: LSQ computes it on
every call, while LSQ+ fixes it on the first batch.
"""

from enum import StrEnum

import torch
from torch import Tensor
from torch.autograd import Function


class Rounding(StrEnum):
    HALF_EVEN = "half_even"
    HALF_AWAY = "half_away"


def round_values(x: Tensor, rounding: Rounding) -> Tensor:
    if rounding is Rounding.HALF_EVEN:
        return x.round()
    magnitude = torch.floor(x.abs() + 0.5)
    return torch.sign(x) * magnitude


def integer_range(bits: int, signed: bool) -> tuple[int, int]:
    """Return the smallest and largest integer that `bits` bits can hold."""
    if signed:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


class _LearnedStepQuantize(Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: Tensor,
        scale: Tensor,
        offset: Tensor,
        qn: int,
        qp: int,
        grad_factor: float,
        rounding: Rounding,
    ) -> Tensor:
        q = (x - offset) / scale
        ctx.save_for_backward(q)
        ctx.qn, ctx.qp, ctx.grad_factor, ctx.rounding = qn, qp, grad_factor, rounding
        clamped = q.clamp(qn, qp)
        rounded = round_values(clamped, rounding)
        return rounded * scale + offset

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        (q,) = ctx.saved_tensors
        below = (q < ctx.qn).to(q.dtype)
        above = (q > ctx.qp).to(q.dtype)
        inside = 1.0 - below - above
        rounding_error = round_values(q, ctx.rounding) - q
        step_term = below * ctx.qn + above * ctx.qp + inside * rounding_error
        grad_scale = (step_term * grad_output * ctx.grad_factor).sum().reshape(1)
        grad_offset = ((below + above) * grad_output * ctx.grad_factor).sum().reshape(1)
        grad_x = inside * grad_output
        return grad_x, grad_scale, grad_offset, None, None, None, None


def lsq_grad_factor(x: Tensor, qp: int) -> float:
    """Return the step-size gradient factor `1 / sqrt(numel(x) * qp)` of both papers."""
    return 1.0 / (x.numel() * qp) ** 0.5


def learned_step_quantize(
    x: Tensor,
    scale: Tensor,
    offset: Tensor,
    qn: int,
    qp: int,
    grad_factor: float,
    rounding: Rounding,
) -> Tensor:
    """Fake-quantize `x`; see the module docstring for the forward and backward rules."""
    result: Tensor = _LearnedStepQuantize.apply(x, scale, offset, qn, qp, grad_factor, rounding)
    return result
