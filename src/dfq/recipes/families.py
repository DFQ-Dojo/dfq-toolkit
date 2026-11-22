"""Build the detector of a family, in float or from a quantization-aware checkpoint."""

from pathlib import Path

import torch

from dfq.detection import Detector
from dfq.models.maskrcnn.resnet import load_resnet50
from dfq.quant import QuantConfig, quantize
from dfq.recipes.configs import Family, ModelConfig


def float_detector(config: ModelConfig, device: torch.device) -> Detector:
    loaders = {Family.RESNET50: load_resnet50}
    detector = loaders[config.family](config.weights)
    detector.network.to(device)
    return detector


def quantized_detector(
    config: ModelConfig, quant: QuantConfig, checkpoint: Path | None, device: torch.device
) -> Detector:
    """The float detector, quantized, then loaded from `checkpoint` when given."""
    detector = float_detector(config, device)
    quantize(detector.network, quant, detector.quant_scope())
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        detector.network.load_state_dict(state)
    return detector


def is_mask_rcnn(family: Family) -> bool:
    return family in {Family.RESNET50}
