"""Generate the PSAQ-ViT baseline's calibration images.

uv run python scripts/psaq.py configs/baselines/psaq.toml [key=value ...]
"""

import argparse
from pathlib import Path

from dfq.common.config import load_config
from dfq.recipes.configs import PsaqConfig
from dfq.recipes.run import psaq


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value, with a TOML value")
    arguments = parser.parse_args()
    config = load_config(arguments.config, PsaqConfig, arguments.overrides)
    count = psaq(config)
    print(f"{config.output} holds {count} images")


if __name__ == "__main__":
    main()
