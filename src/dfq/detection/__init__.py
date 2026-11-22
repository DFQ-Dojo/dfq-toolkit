"""What every detector provides to the pipelines: records, the Detector protocol, COCO classes."""

from dfq.detection.modes import freeze, generation_mode
from dfq.detection.protocol import Detector
from dfq.detection.records import Detections, Targets

__all__ = ["Detections", "Detector", "Targets", "freeze", "generation_mode"]
