"""Write a label set: one random box per image, or the boxes of a COCO annotation file.

uv run python scripts/labels.py configs/labels/one_box.toml [key=value ...]
"""

import argparse
from pathlib import Path

from dfq.common.config import load_config
from dfq.recipes.configs import LabelsConfig
from dfq.recipes.run import make_labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value, with a TOML value")
    arguments = parser.parse_args()
    config = load_config(arguments.config, LabelsConfig, arguments.overrides)
    count = make_labels(config)
    print(f"wrote {count} images' labels to {config.output}")


if __name__ == "__main__":
    main()
