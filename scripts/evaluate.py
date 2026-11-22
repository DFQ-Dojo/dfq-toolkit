"""Evaluate a float or quantized detector on COCO.

uv run python scripts/evaluate.py configs/resnet50/evaluate.toml [key=value ...]
"""

import argparse
from pathlib import Path

from dfq.common import distributed
from dfq.common.config import load_config
from dfq.recipes.configs import EvaluateConfig
from dfq.recipes.run import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value, with a TOML value")
    parser.add_argument("--output", type=Path, help="write the statistics to this JSON file")
    arguments = parser.parse_args()
    config = load_config(arguments.config, EvaluateConfig, arguments.overrides)
    stats = evaluate(config, arguments.output)
    if distributed.is_main():
        for name, value in stats.items():
            print(f"{name:16} {value:.4f}")


if __name__ == "__main__":
    main()
