"""Read Ultralytics and YOLOv5 checkpoints, which pickle whole modules, into vendored classes.

A pickled module names its class by import path, such as `ultralytics.nn.modules.block.C3k2`.
`load_pickled` resolves those paths in `dfq.vendor` instead, through `MODULE_PATHS` (Ultralytics)
or `YOLOV5_PATHS` (YOLOv5). It refuses
every other class except PyTorch's, a few containers and NumPy's array rebuilders, so loading a
checkpoint cannot run arbitrary code. `scripts/convert_checkpoint.py` then saves the model as a
plain state dict and its model YAML.
"""

import importlib
import pickle
import types
from pathlib import Path
from typing import Any

import torch

MODULE_PATHS = {
    "ultralytics.nn.tasks": "dfq.vendor.ultralytics.nn.tasks",
    "ultralytics.nn.modules.conv": "dfq.vendor.ultralytics.nn.modules.conv",
    "ultralytics.nn.modules.block": "dfq.vendor.ultralytics.nn.modules.block",
    "ultralytics.nn.modules.head": "dfq.vendor.ultralytics.nn.modules.head",
}

YOLOV5_PATHS = {
    "models.yolo": "dfq.vendor.yolov5.yolo",
    "models.common": "dfq.vendor.yolov5.common",
}

SAFE = {
    ("collections", "OrderedDict"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "slice"),
    ("_codecs", "encode"),
    ("pathlib", "PosixPath"),
    ("pathlib", "WindowsPath"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class Opaque(types.SimpleNamespace):
    """Stands in for a refused class whose instances only carry settings, such as training args."""


def remapped_class(module: str, name: str, mapping: dict[str, str]) -> Any:
    if module == "__builtin__":  # Python 2's name, which old pickles keep
        module = "builtins"
    if module in mapping:
        return getattr(importlib.import_module(mapping[module]), name)
    if module.startswith("torch") or (module, name) in SAFE:
        return getattr(importlib.import_module(module), name)
    if module.startswith(("ultralytics.utils", "models.", "utils.")) and name[0].isupper():
        return Opaque
    raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def remapping_module(mapping: dict[str, str]) -> types.ModuleType:
    """A stand-in for the `pickle` module whose unpickler resolves classes through `mapping`."""

    class Unpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            return remapped_class(module, name, mapping)

    stand_in = types.ModuleType("remapping_pickle")
    stand_in.Unpickler = Unpickler  # type: ignore[attr-defined]
    stand_in.load = pickle.load  # type: ignore[attr-defined]
    return stand_in


def load_pickled(path: Path, mapping: dict[str, str]) -> dict[str, Any]:
    pickle_module = remapping_module(mapping)
    checkpoint: dict[str, Any] = torch.load(
        path, map_location="cpu", weights_only=False, pickle_module=pickle_module
    )
    return checkpoint


def plain_checkpoint(path: Path, mapping: dict[str, str]) -> dict[str, Any]:
    """The model of a pickled checkpoint as `{"yaml": model dict, "state_dict": float weights}`."""
    checkpoint = load_pickled(path, mapping)
    model = checkpoint.get("ema") or checkpoint["model"]
    state = {name: value.float() for name, value in model.state_dict().items()}
    return {"yaml": dict(model.yaml), "state_dict": state}
