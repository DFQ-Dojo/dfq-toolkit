"""YOLOv5 as a `Detector`, on the vendored YOLOv5 model and loss.

The loss gains follow YOLOv5's scaling for 3 levels, `nc` classes and `image_size`: `box` x 3/nl,
`cls` x nc/80 x 3/nl, `obj` x (image_size/640)^2 x 3/nl. `predict` runs YOLOv5's NMS with its
validation settings (confidence 0.001, IoU 0.6, 300 boxes). Quantization keeps the first
convolution and the detection head in float, as the old 32-bit layers of the per-bit YAML files
did.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from dfq.detection import Detections, Targets
from dfq.quant import QuantScope
from dfq.vendor.yolov5.loss import ComputeLoss
from dfq.vendor.yolov5.ops import non_max_suppression
from dfq.vendor.yolov5.yolo import Detect, DetectionModel, yolov5_yaml

FEATURE_KINDS: dict[str, tuple[type[nn.Module], ...]] = {
    "cnn": (nn.Conv2d,),
    "bn": (nn.BatchNorm2d,),
    "cnnbn": (nn.Conv2d, nn.BatchNorm2d),
}

GAINS = {"box": 0.05, "cls": 0.5, "obj": 1.0, "cls_pw": 1.0, "obj_pw": 1.0, "anchor_t": 4.0}


def scaled_gains(
    gains: dict[str, float], levels: int, classes: int, image_size: int
) -> dict[str, float]:
    return {
        **gains,
        "box": gains["box"] * 3 / levels,
        "cls": gains["cls"] * classes / 80 * 3 / levels,
        "obj": gains["obj"] * (image_size / 640) ** 2 * 3 / levels,
    }


def target_rows(images: Tensor, targets: Sequence[Targets]) -> Tensor:
    """(image, class, x, y, w, h) rows with centre boxes normalized to the image."""
    height, width = images.shape[-2:]
    scale = torch.tensor([width, height, width, height], dtype=images.dtype, device=images.device)
    rows = []
    for index, target in enumerate(targets):
        corners = target.boxes / scale
        centres = torch.cat(
            [(corners[:, :2] + corners[:, 2:]) / 2, corners[:, 2:] - corners[:, :2]], dim=1
        )
        image = torch.full((len(target.labels), 1), index, dtype=images.dtype, device=images.device)
        rows.append(torch.cat([image, target.labels.to(images.dtype)[:, None], centres], dim=1))
    return torch.cat(rows).reshape(-1, 6)


@dataclass
class Yolov5Detector:
    network: DetectionModel
    image_size: int = 640
    gains: dict[str, float] = field(default_factory=lambda: dict(GAINS))
    confidence: float = 0.001
    iou: float = 0.6
    max_detections: int = 300

    @property
    def num_classes(self) -> int:
        return int(self.network.yaml["nc"])

    def losses(self, images: Tensor, targets: Sequence[Targets]) -> dict[str, Tensor]:
        head = self.network.model[-1]
        assert isinstance(head, Detect)
        hyp = scaled_gains(self.gains, head.nl, self.num_classes, self.image_size)
        criterion = ComputeLoss(self.network, hyp)
        raw = self.network(images)
        loss, _ = criterion(raw, target_rows(images, targets))
        return {"yolo": loss.squeeze()}

    def predict(self, images: Tensor) -> list[Detections]:
        decoded, _ = self.network(images)
        kept = non_max_suppression(decoded, self.confidence, self.iou, max_det=self.max_detections)
        return [Detections(boxes=k[:, :4], scores=k[:, 4], labels=k[:, 5].long()) for k in kept]

    def quant_scope(self) -> QuantScope:
        head = len(self.network.model) - 1
        return QuantScope(include=("model.*",), exclude=("model.0.*", f"model.{head}.*"))

    def logit_layer(self) -> str:
        return f"model.{len(self.network.model) - 1}"

    def logits(self, output: object) -> list[Tensor]:
        raw = output[1] if isinstance(output, tuple) else output
        assert isinstance(raw, list)
        return raw

    def feature_layers(self, kind: str) -> list[str]:
        if kind == "none":
            return []
        types = FEATURE_KINDS[kind]
        return [name for name, module in self.network.named_modules() if isinstance(module, types)]


def yolov5_network(scale: str) -> DetectionModel:
    return DetectionModel(yolov5_yaml(scale))


def load_yolov5(weights: Path, image_size: int = 640) -> Yolov5Detector:
    """Load a checkpoint converted by `scripts/convert_checkpoint.py`: model YAML and state dict."""
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    network = DetectionModel(checkpoint["yaml"])
    network.load_state_dict(checkpoint["state_dict"])
    return Yolov5Detector(network, image_size=image_size)
