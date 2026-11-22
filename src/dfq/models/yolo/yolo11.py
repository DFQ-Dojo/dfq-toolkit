"""YOLO11 as a `Detector`, on the vendored Ultralytics model and loss.

`losses` packs targets into Ultralytics' batch format (image index, class, normalized centre box)
and returns its loss, which Ultralytics scales by the batch size. `predict` runs NMS with
Ultralytics' validation settings (confidence 0.001, IoU 0.7, 300 boxes). Quantization keeps the
first convolution and the detection head in float, as the old `yolo11/quant/quantize.py` did.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from dfq.detection import Detections, Targets
from dfq.quant import QuantScope
from dfq.vendor.ultralytics.nn.tasks import DetectionModel, yolo11_yaml
from dfq.vendor.ultralytics.utils.ops import non_max_suppression

FEATURE_KINDS: dict[str, tuple[type[nn.Module], ...]] = {
    "cnn": (nn.Conv2d,),
    "bn": (nn.BatchNorm2d,),
    "cnnbn": (nn.Conv2d, nn.BatchNorm2d),
}


def ultralytics_batch(images: Tensor, targets: Sequence[Targets]) -> dict[str, Tensor]:
    height, width = images.shape[-2:]
    scale = torch.tensor([width, height, width, height], dtype=images.dtype, device=images.device)
    indices, classes, boxes = [], [], []
    for index, target in enumerate(targets):
        corners = target.boxes / scale
        centres = torch.cat(
            [(corners[:, :2] + corners[:, 2:]) / 2, corners[:, 2:] - corners[:, :2]], dim=1
        )
        indices.append(
            torch.full((len(target.labels),), index, dtype=images.dtype, device=images.device)
        )
        classes.append(target.labels.to(images.dtype))
        boxes.append(centres)
    return {
        "img": images,
        "batch_idx": torch.cat(indices),
        "cls": torch.cat(classes).view(-1, 1),
        "bboxes": torch.cat(boxes).reshape(-1, 4),
    }


@dataclass
class YoloDetector:
    network: DetectionModel
    gains: dict[str, float] = field(default_factory=lambda: {"box": 7.5, "cls": 0.5, "dfl": 1.5})
    confidence: float = 0.001
    iou: float = 0.7
    max_detections: int = 300

    @property
    def num_classes(self) -> int:
        return int(self.network.yaml["nc"])

    def losses(self, images: Tensor, targets: Sequence[Targets]) -> dict[str, Tensor]:
        self.network.args = SimpleNamespace(**self.gains)  # read by Ultralytics' loss
        batch = ultralytics_batch(images, targets)
        loss, _ = self.network.loss(batch)
        return {"yolo": loss}

    def predict(self, images: Tensor) -> list[Detections]:
        predictions, _ = self.network(images)
        kept = non_max_suppression(
            predictions, self.confidence, self.iou, max_det=self.max_detections
        )
        return [Detections(boxes=k[:, :4], scores=k[:, 4], labels=k[:, 5].long()) for k in kept]

    def quant_scope(self) -> QuantScope:
        head = len(self.network.model) - 1
        return QuantScope(include=("model.*",), exclude=("model.0.*", f"model.{head}.*"))

    def logit_layer(self) -> str:
        return f"model.{len(self.network.model) - 1}"

    def logits(self, output: object) -> list[Tensor]:
        """The raw outputs of the head per level; in evaluation mode, the second element."""
        raw = output[1] if isinstance(output, tuple) else output
        assert isinstance(raw, list)
        return raw

    def feature_layers(self, kind: str) -> list[str]:
        if kind == "none":
            return []
        types = FEATURE_KINDS[kind]
        modules = self.network.named_modules()
        return [name for name, module in modules if isinstance(module, types) and "dfl" not in name]


def yolo11_network(scale: str) -> DetectionModel:
    return DetectionModel(yolo11_yaml(scale))


def load_yolo11(weights: Path) -> YoloDetector:
    """Load a checkpoint converted by `scripts/convert_checkpoint.py`: model YAML and state dict."""
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    network = DetectionModel(checkpoint["yaml"])
    network.load_state_dict(checkpoint["state_dict"])
    return YoloDetector(network)
