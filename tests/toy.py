"""A tiny detector that implements the Detector protocol, for testing the pipelines on CPU."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from dfq.detection import Detections, Targets
from dfq.quant import QuantScope


class ToyNetwork(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.body = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.head = nn.Conv2d(8, num_classes + 4, 1)

    def forward(self, images: Tensor) -> Tensor:
        stem = self.stem(images)
        body = self.body(stem)
        return self.head(body)


class ToyDetector:
    """Predicts one box per image: the mean box regression and the most likely class."""

    def __init__(self, num_classes: int = 3) -> None:
        self._network = ToyNetwork(num_classes)
        self._num_classes = num_classes

    @property
    def network(self) -> nn.Module:
        return self._network

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def losses(self, images: Tensor, targets: Sequence[Targets]) -> dict[str, Tensor]:
        output = self._network(images)
        class_logits = output[:, : self._num_classes].mean(dim=(2, 3))
        labels = torch.stack([target.labels[0] for target in targets])
        boxes = output[:, self._num_classes :].mean(dim=(2, 3))
        target_boxes = torch.stack([target.boxes[0] for target in targets]) / images.shape[-1]
        classification = nn.functional.cross_entropy(class_logits, labels)
        regression = nn.functional.l1_loss(boxes, target_boxes)
        return {"classification": classification, "regression": regression}

    def predict(self, images: Tensor) -> list[Detections]:
        output = self._network(images)
        probs = output[:, : self._num_classes].mean(dim=(2, 3)).softmax(dim=1)
        boxes = output[:, self._num_classes :].mean(dim=(2, 3)).sigmoid() * images.shape[-1]
        scores, labels = probs.max(dim=1)
        return [
            Detections(boxes[i : i + 1], scores[i : i + 1], labels[i : i + 1])
            for i in range(images.shape[0])
        ]

    def quant_scope(self) -> QuantScope:
        return QuantScope(include=("*",), exclude=("stem.*",))

    def logit_layer(self) -> str:
        return "head"

    def logits(self, output: object) -> list[Tensor]:
        assert isinstance(output, Tensor)
        return [output]

    def feature_layers(self, kind: str) -> list[str]:
        return {"conv": ["stem.0", "body.0"], "none": []}[kind]


def toy_batch(batch_size: int = 2, size: int = 16) -> tuple[Tensor, list[Targets]]:
    images = torch.rand(batch_size, 3, size, size)
    targets = [
        Targets(boxes=torch.tensor([[2.0, 2.0, 10.0, 12.0]]), labels=torch.tensor([i % 3]))
        for i in range(batch_size)
    ]
    return images, targets
