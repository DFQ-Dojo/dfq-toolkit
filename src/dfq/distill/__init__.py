"""Knowledge distillation from a full-precision teacher to a quantized student."""

from dfq.distill.objective import DISTConfig, DistillConfig, KLConfig, distill_losses

__all__ = ["DISTConfig", "DistillConfig", "KLConfig", "distill_losses"]
