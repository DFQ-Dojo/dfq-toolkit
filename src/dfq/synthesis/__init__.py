"""Calibration image synthesis: priors, augmentation, adaptive box sampling, the generation loop."""

from dfq.synthesis.box_sampling import BoxSamplingConfig
from dfq.synthesis.generate import GeneratedBatch, GenerationConfig, StepRecord, generate_batch

__all__ = [
    "BoxSamplingConfig",
    "GeneratedBatch",
    "GenerationConfig",
    "StepRecord",
    "generate_batch",
]
