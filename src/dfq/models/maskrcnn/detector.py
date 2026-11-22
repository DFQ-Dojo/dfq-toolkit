"""A torchvision Mask R-CNN as a `Detector`, for any backbone.

torchvision heads count the background as class 0. `network_labels[i]` is the head's label of
dfq class `i`: the COCO category id for torchvision's 91-way COCO weights, or `i + 1` for heads
converted from MMDetection. Targets without masks get box-shaped masks, which the mask head
needs during image generation from boxes alone.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchvision.models.detection import MaskRCNN

from dfq.detection import Detections, Targets
from dfq.quant import QuantScope


def box_masks(boxes: Tensor, height: int, width: int) -> Tensor:
    rows = torch.arange(height, device=boxes.device).view(1, height, 1)
    columns = torch.arange(width, device=boxes.device).view(1, 1, width)
    x1, y1, x2, y2 = (boxes[:, i].view(-1, 1, 1) for i in range(4))
    inside = (
        (columns >= x1.floor()) & (columns < x2.ceil()) & (rows >= y1.floor()) & (rows < y2.ceil())
    )
    return inside.to(torch.uint8)


@dataclass
class MaskRCNNDetector:
    network: MaskRCNN
    network_labels: Tensor
    scope: QuantScope
    feature_kinds: dict[str, tuple[type[nn.Module], ...]]

    @property
    def num_classes(self) -> int:
        return len(self.network_labels)

    def network_targets(
        self, images: Tensor, targets: Sequence[Targets]
    ) -> list[dict[str, Tensor]]:
        height, width = images.shape[-2:]
        labels = self.network_labels.to(images.device)
        converted = []
        for target in targets:
            masks = (
                target.masks if target.masks is not None else box_masks(target.boxes, height, width)
            )
            converted.append(
                {"boxes": target.boxes, "labels": labels[target.labels], "masks": masks}
            )
        return converted

    def losses(self, images: Tensor, targets: Sequence[Targets]) -> dict[str, Tensor]:
        network_targets = self.network_targets(images, targets)
        losses: dict[str, Tensor] = self.network(list(images), network_targets)
        return losses

    def predict(self, images: Tensor) -> list[Detections]:
        outputs = self.network(list(images))
        lookup = torch.full((int(self.network_labels.max()) + 1,), -1, dtype=torch.int64)
        lookup[self.network_labels] = torch.arange(len(self.network_labels))
        lookup = lookup.to(images.device)
        detections = []
        for output in outputs:
            labels = lookup[output["labels"]]
            known = labels >= 0
            masks = output["masks"][known] if "masks" in output else None
            detections.append(
                Detections(output["boxes"][known], output["scores"][known], labels[known], masks)
            )
        return detections

    def quant_scope(self) -> QuantScope:
        return self.scope

    def logit_layer(self) -> str:
        return "rpn.head"

    def logits(self, output: object) -> list[Tensor]:
        """The RPN's objectness and box deltas at every level."""
        assert isinstance(output, tuple)
        objectness, deltas = output
        return [*objectness, *deltas]

    def feature_layers(self, kind: str) -> list[str]:
        if kind == "none":
            return []
        types = self.feature_kinds[kind]
        return [
            name
            for name, module in self.network.named_modules()
            if name and isinstance(module, types)
        ]
