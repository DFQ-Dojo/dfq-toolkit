# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""The YOLO11 detection model, from ultralytics 8.3.53 (`ultralytics/nn/tasks.py`).

`DetectionModel` builds its layers from a model dict (`cfg/yolo11.yaml` with a `scale`). Each
layer reads the outputs named by its `f` ("from") index; `save` lists the outputs that later
layers read. `loss(batch)` returns Ultralytics' detection loss.
"""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from ..utils.loss import v8DetectionLoss
from ..utils.ops import make_divisible
from .modules import C2PSA, SPPF, C3k2, Concat, Conv, Detect

CONFIG = Path(__file__).resolve().parent.parent / "cfg" / "yolo11.yaml"

MODULES: dict[str, type[nn.Module]] = {
    "Conv": Conv,
    "C3k2": C3k2,
    "SPPF": SPPF,
    "C2PSA": C2PSA,
    "Concat": Concat,
    "Detect": Detect,
    "nn.Upsample": nn.Upsample,
}
SCALED = {Conv, C3k2, SPPF, C2PSA}
REPEATED = {C3k2, C2PSA}


def yolo11_yaml(scale: str) -> dict[str, Any]:
    """The model dict of YOLO11 at `scale` (n, s, m, l or x)."""
    model_dict: dict[str, Any] = yaml.safe_load(CONFIG.read_text())
    model_dict["scale"] = scale
    return model_dict


def initialize_weights(model: nn.Module) -> None:
    """Set batch-norm `eps` and `momentum` as Ultralytics trains, and make activations in place."""
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            module.eps = 1e-3
            module.momentum = 0.03
        elif type(module) in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            module.inplace = True  # type: ignore[assignment]


def layer_arguments(
    module: type[nn.Module],
    args: list[Any],
    inputs: list[int],
    f: int | list[int],
    d: dict[str, Any],
) -> tuple[list[Any], int, bool]:
    """The constructor arguments of one layer, its output channels, and whether it is repeated."""
    nc = d["nc"]
    depth, width, max_channels = d["scales"][d["scale"]]
    if module in SCALED:
        assert isinstance(f, int)
        out = args[0] if args[0] == nc else make_divisible(min(args[0], max_channels) * width, 8)
        built = [inputs[f], out, *args[1:]]
        if module is C3k2 and d["scale"] in "mlx":
            built[2] = True  # use C3k blocks
        return built, out, module in REPEATED
    if module is Concat:
        assert isinstance(f, list)
        return args, sum(inputs[x] for x in f), False
    if module is Detect:
        assert isinstance(f, list)
        return [*args, [inputs[x] for x in f], False], 0, False
    assert isinstance(f, int)
    return args, inputs[f], False


def parse_model(d: dict[str, Any], ch: int) -> tuple[nn.Sequential, list[int]]:
    """Build the layers of model dict `d` for `ch` input channels; return them and `save`."""
    depth, _, _ = d["scales"][d["scale"]]
    inputs = [ch]
    layers: list[nn.Module] = []
    save: list[int] = []
    for i, (f, n, name, args) in enumerate(d["backbone"] + d["head"]):
        module = MODULES[name]
        constants = {"nc": d["nc"], "None": None}
        values = [constants.get(a, a) if isinstance(a, str) else a for a in args]
        repeats = max(round(n * depth), 1) if n > 1 else n
        built, out, repeated = layer_arguments(module, values, inputs, f, d)
        if repeated:
            built.insert(2, repeats)
            repeats = 1
        if repeats > 1:
            layer: nn.Module = nn.Sequential(*(module(*built) for _ in range(repeats)))
        else:
            layer = module(*built)
        layer.i, layer.f = i, f  # type: ignore[assignment]
        sources = [f] if isinstance(f, int) else f
        save.extend(x % i for x in sources if x != -1)
        layers.append(layer)
        if i == 0:
            inputs = []
        inputs.append(out)
    return nn.Sequential(*layers), sorted(save)


class DetectionModel(nn.Module):
    def __init__(self, cfg: dict[str, Any], ch: int = 3) -> None:
        super().__init__()
        self.yaml = cfg
        model, save = parse_model(deepcopy(cfg), ch)
        self.model: nn.Sequential = model
        self.save = save
        self.names = {i: str(i) for i in range(cfg["nc"])}
        head = self.model[-1]
        assert isinstance(head, Detect)
        probe = torch.zeros(1, ch, 256, 256)
        levels = self.predict(probe)
        assert isinstance(levels, list)
        head.stride = torch.tensor([256 / level.shape[-2] for level in levels])
        self.stride = head.stride
        head.bias_init()
        initialize_weights(self)
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        self.criterion: v8DetectionLoss | None = None

    def forward(self, x: Tensor) -> Any:
        return self.predict(x)

    def predict(self, x: Tensor) -> Any:
        """Run every layer; a layer whose `f` is not -1 reads saved outputs instead of the last."""
        outputs: list[Any] = []
        current: Any = x
        for layer in self.model:
            source: int | list[int] = layer.f  # type: ignore[assignment]
            if isinstance(source, list):
                current = [current if j == -1 else outputs[j] for j in source]
            elif source != -1:
                current = outputs[source]
            current = layer(current)
            outputs.append(current if layer.i in self.save else None)
        return current

    def loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """The loss sum times the batch size, and the detached (box, cls, dfl) terms."""
        if self.criterion is None:
            self.criterion = v8DetectionLoss(self)
        predictions = self.predict(batch["img"])
        return self.criterion(predictions, batch)
