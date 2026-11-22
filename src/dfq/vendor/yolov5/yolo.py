# Ultralytics YOLOv5, AGPL-3.0 license
"""The YOLOv5 detection model, from YOLOv5 v7.0 (`models/yolo.py`).

`Detect` predicts, per anchor and cell, centre offsets, size, objectness and class logits. In
training mode it returns each level `[B, anchors, H, W, 5 + nc]`; in evaluation mode it also
returns decoded boxes (centre, width, height in pixels), objectness and class scores. Checkpoints
pickle `DetectionModel` as `Model`.
"""

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from dfq.vendor.ultralytics.nn.tasks import initialize_weights
from dfq.vendor.ultralytics.utils.ops import make_divisible

from .common import C3, SPPF, Concat, Conv

CONFIG = Path(__file__).resolve().parent / "yolov5.yaml"


def yolov5_yaml(scale: str) -> dict[str, Any]:
    """The model dict of YOLOv5 at `scale` (n, s, m, l or x)."""
    model_dict: dict[str, Any] = yaml.safe_load(CONFIG.read_text())
    depth, width = model_dict.pop("scales")[scale]
    model_dict["depth_multiple"] = depth
    model_dict["width_multiple"] = width
    return model_dict


class Detect(nn.Module):
    def __init__(
        self, nc: int = 80, anchors: list[list[int]] | None = None, ch: tuple[int, ...] = ()
    ) -> None:
        super().__init__()
        assert anchors is not None
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.stride = torch.zeros(self.nl)

    def grid(
        self, level: int, nx: int, ny: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        """Cell offsets (minus 0.5) and anchor sizes in pixels, `[1, anchors, ny, nx, 2]` each."""
        shape = (1, self.na, ny, nx, 2)
        ys = torch.arange(ny, device=device, dtype=dtype)
        xs = torch.arange(nx, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        cells = torch.stack((grid_x, grid_y), 2).expand(shape) - 0.5
        anchors = self.anchors[level].to(device=device, dtype=dtype)  # type: ignore[index]
        stride = self.stride[level].to(device)
        sizes = (anchors * stride).view((1, self.na, 1, 1, 2)).expand(shape)
        return cells, sizes

    def forward(self, x: list[Tensor]) -> list[Tensor] | tuple[Tensor, list[Tensor]]:
        raw, decoded = [], []
        for i in range(self.nl):
            output = self.m[i](x[i])
            batch, _, ny, nx = output.shape
            level = output.view(batch, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            raw.append(level)
            if self.training:
                continue
            cells, sizes = self.grid(i, nx, ny, level.device, level.dtype)
            xy, wh, conf = level.sigmoid().split((2, 2, self.nc + 1), 4)
            centres = (xy * 2 + cells) * self.stride[i].to(level.device)
            dimensions = (wh * 2) ** 2 * sizes
            decoded.append(
                torch.cat((centres, dimensions, conf), 4).view(batch, self.na * nx * ny, self.no)
            )
        if self.training:
            return raw
        return torch.cat(decoded, 1), raw


SCALED = {Conv, C3, SPPF}
MODULES: dict[str, type[nn.Module]] = {
    "Conv": Conv,
    "C3": C3,
    "SPPF": SPPF,
    "Concat": Concat,
    "Detect": Detect,
    "nn.Upsample": nn.Upsample,
}


def parse_model(d: dict[str, Any], ch: int) -> tuple[nn.Sequential, list[int]]:
    """Build the layers of model dict `d` for `ch` input channels; return them and `save`."""
    anchors, nc = d["anchors"], d["nc"]
    depth, width = d["depth_multiple"], d["width_multiple"]
    outputs = (len(anchors[0]) // 2) * (nc + 5)
    inputs = [ch]
    layers: list[nn.Module] = []
    save: list[int] = []
    for i, (f, n, name, args) in enumerate(d["backbone"] + d["head"]):
        module = MODULES[name]
        constants = {"nc": nc, "anchors": anchors, "None": None}
        values = [constants.get(a, a) if isinstance(a, str) else a for a in args]
        repeats = max(round(n * depth), 1) if n > 1 else n
        if module in SCALED:
            out = values[0] if values[0] == outputs else make_divisible(values[0] * width, 8)
            values = [inputs[f], out, *values[1:]]
            if module is C3:
                values.insert(2, repeats)
                repeats = 1
        elif module is Concat:
            out = sum(inputs[x] for x in f)
        elif module is Detect:
            values = [values[0], values[1], [inputs[x] for x in f]]
            out = 0
        else:
            out = inputs[f]
        if repeats > 1:
            layer: nn.Module = nn.Sequential(*(module(*values) for _ in range(repeats)))
        else:
            layer = module(*values)
        layer.i, layer.f = i, f  # type: ignore[assignment]
        sources = [f] if isinstance(f, int) else f
        save.extend(x % i for x in sources if x != -1)
        layers.append(layer)
        if i == 0:
            inputs = []
        inputs.append(out)
    return nn.Sequential(*layers), sorted(save)


def initialize_biases(head: Detect) -> None:
    """Objectness for 8 objects per 640-pixel image; class prior 0.6 / (nc - 0.99999)."""
    for conv, stride in zip(head.m, head.stride, strict=True):
        assert isinstance(conv, nn.Conv2d) and conv.bias is not None
        bias = conv.bias.view(head.na, -1)
        bias.data[:, 4] += math.log(8 / (640 / stride) ** 2)
        bias.data[:, 5 : 5 + head.nc] += math.log(0.6 / (head.nc - 0.99999))
        conv.bias = nn.Parameter(bias.view(-1), requires_grad=True)


class DetectionModel(nn.Module):
    def __init__(self, cfg: dict[str, Any], ch: int = 3) -> None:
        super().__init__()
        self.yaml = cfg
        model, save = parse_model(deepcopy(cfg), ch)
        self.model: nn.Sequential = model
        self.save = save
        head = self.model[-1]
        assert isinstance(head, Detect)
        levels = self.forward(torch.zeros(1, ch, 256, 256))
        head.stride = torch.tensor([256 / level.shape[-2] for level in levels])
        anchors = head.anchors
        assert isinstance(anchors, Tensor)
        mean_areas = anchors.prod(-1).mean(-1).view(-1)
        area_change = mean_areas[-1] - mean_areas[0]
        stride_change = head.stride[-1] - head.stride[0]
        if area_change != 0 and area_change.sign() != stride_change.sign():
            anchors[:] = anchors.flip(0)  # order anchors from the finest level to the coarsest
        anchors /= head.stride.view(-1, 1, 1)
        self.stride = head.stride
        initialize_biases(head)
        initialize_weights(self)

    def forward(self, x: Tensor) -> Any:
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


Model = DetectionModel  # the name YOLOv5 checkpoints pickle
