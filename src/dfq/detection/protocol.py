"""The contract between the pipelines and the detector adapters.

`synthesis/`, `distill/` and `training/` depend on `Detector`; each adapter in `models/`
implements it. Images are always a float batch `[B, 3, H, W]` of RGB values in `[0, 1]`, already
resized and padded; the adapter normalizes them as its network expects.
"""

from collections.abc import Sequence
from typing import Protocol

from torch import Tensor, nn

from dfq.detection.records import Detections, Targets
from dfq.quant import QuantScope


class Detector(Protocol):
    @property
    def network(self) -> nn.Module:
        """The PyTorch module: its parameters, modes and hooks."""
        ...

    @property
    def num_classes(self) -> int: ...

    def losses(self, images: Tensor, targets: Sequence[Targets]) -> dict[str, Tensor]:
        """Return the detector's own training losses, each a scalar.

        The network must be in training mode; `generation_mode` keeps its normalization layers
        frozen, as image generation needs.
        """
        ...

    def predict(self, images: Tensor) -> list[Detections]:
        """Return the detections per image. The network must be in evaluation mode."""
        ...

    def quant_scope(self) -> QuantScope:
        """The layers that quantization-aware training quantizes."""
        ...

    def logit_layer(self) -> str:
        """The name of the module whose output holds the raw head outputs, for distillation."""
        ...

    def logits(self, output: object) -> list[Tensor]:
        """Turn the output of `logit_layer` into tensors of shape `[B, ...]`."""
        ...

    def feature_layers(self, kind: str) -> list[str]:
        """The names of the modules whose outputs feature distillation matches, for `kind`."""
        ...
