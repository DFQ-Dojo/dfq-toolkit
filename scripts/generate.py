"""Stage I: generate calibration images from a label set.

uv run python scripts/generate.py configs/resnet50/generate.toml [key=value ...]
"""

import argparse
from pathlib import Path

from dfq.common.config import load_config
from dfq.recipes.configs import GenerateConfig
from dfq.recipes.run import generate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value, with a TOML value")
    arguments = parser.parse_args()
    config = load_config(arguments.config, GenerateConfig, arguments.overrides)
    count = generate(config)
    print(f"{config.output} holds {count} images")


if __name__ == "__main__":
    main()
