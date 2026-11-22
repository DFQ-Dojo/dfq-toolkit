"""What every detector provides to the pipelines: records, the Detector protocol, COCO classes."""

from dfq.detection.modes import freeze, generation_mode
from dfq.detection.protocol import Detector
from dfq.detection.records import Detections, Targets, targets_to

__all__ = ["Detections", "Detector", "Targets", "freeze", "generation_mode", "targets_to"]
