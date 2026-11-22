"""Config files: TOML read into Pydantic records, with command-line overrides.

An override is `dotted.key=value`, where `value` is TOML: `train.epochs=3`, `quant.method="lsq+"`.
"""

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def with_override(data: dict[str, Any], override: str) -> dict[str, Any]:
    key, _, text = override.partition("=")
    value = tomllib.loads(f"value = {text}")["value"]
    *parents, leaf = key.split(".")
    updated = dict(data)
    node = updated
    for parent in parents:
        node[parent] = dict(node.get(parent, {}))
        node = node[parent]
    node[leaf] = value
    return updated


def load_config[T: BaseModel](path: Path, model: type[T], overrides: list[str]) -> T:
    data = tomllib.loads(path.read_text())
    for override in overrides:
        data = with_override(data, override)
    return model.model_validate(data)
