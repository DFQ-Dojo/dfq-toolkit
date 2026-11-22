"""YOLO11 modules (vendored from ultralytics 8.3.53)."""

from .block import C2PSA, DFL, SPPF, C3k2
from .conv import Concat, Conv, DWConv
from .head import Detect

__all__ = ["C2PSA", "DFL", "SPPF", "C3k2", "Concat", "Conv", "DWConv", "Detect"]
