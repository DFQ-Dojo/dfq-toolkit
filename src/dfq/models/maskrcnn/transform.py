"""Mask R-CNN's input transform without resizing.

torchvision's `GeneralizedRCNNTransform` normalizes, resizes and pads. In dfq the data pipeline
resizes and pads, as MMDetection's did, and image generation feeds a fixed size, so this
transform only normalizes and batches.
"""

from torch import Tensor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class NormalizeOnly(GeneralizedRCNNTransform):
    def __init__(self) -> None:
        super().__init__(
            min_size=800,
            max_size=1333,
            image_mean=list(IMAGENET_MEAN),
            image_std=list(IMAGENET_STD),
            size_divisible=32,
        )

    def resize(
        self, image: Tensor, target: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        return image, target
