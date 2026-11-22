"""Build the detector of a family, in float or from a quantization-aware checkpoint."""

from pathlib import Path

import torch

from dfq.detection import Detector
from dfq.models.maskrcnn.resnet import load_resnet50
from dfq.models.maskrcnn.transformer_rcnn import load_transformer
from dfq.models.yolo.yolo11 import load_yolo11
from dfq.quant import QuantConfig, quantize
from dfq.recipes.configs import Family, ModelConfig


def is_yolo(family: Family) -> bool:
    return family.value.startswith("yolo")


def float_detector(config: ModelConfig, device: torch.device) -> Detector:
    if config.family is Family.RESNET50:
        detector: Detector = load_resnet50(config.weights)
    elif is_yolo(config.family):
        detector = load_yolo11(config.weights)
    else:
        detector = load_transformer(config.family.value, config.weights)
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
