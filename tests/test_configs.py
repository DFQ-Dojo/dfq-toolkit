"""Every config file in configs/ must load into its record."""

import unittest
from pathlib import Path

from pydantic import BaseModel

from dfq.common.config import load_config
from dfq.recipes.configs import (
    EvaluateConfig,
    GenerateConfig,
    LabelsConfig,
    PsaqConfig,
    QatConfig,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def record_type(path: Path) -> type[BaseModel]:
    if path.parent.name == "labels":
        return LabelsConfig
    if path.stem == "psaq":
        return PsaqConfig
    prefixes: dict[str, type[BaseModel]] = {
        "generate": GenerateConfig,
        "qat": QatConfig,
        "evaluate": EvaluateConfig,
    }
    return next(kind for prefix, kind in prefixes.items() if path.stem.startswith(prefix))


class ConfigFilesTest(unittest.TestCase):
    def test_all_configs_load(self) -> None:
        paths = sorted(CONFIGS.rglob("*.toml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.name):
                load_config(path, record_type(path), [])


if __name__ == "__main__":
    unittest.main()
