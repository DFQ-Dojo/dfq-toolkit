"""Stage II: quantization-aware training, with distillation from the float model.

uv run python scripts/qat.py configs/resnet50/qat_ours.toml [key=value ...]
uv run torchrun --nproc-per-node 8 scripts/qat.py configs/resnet50/qat_ours.toml
"""

import argparse
from pathlib import Path

from dfq.common import distributed
from dfq.common.config import load_config
from dfq.recipes.configs import QatConfig
from dfq.recipes.run import qat


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value, with a TOML value")
    arguments = parser.parse_args()
    config = load_config(arguments.config, QatConfig, arguments.overrides)
    result = qat(config)
    if distributed.is_main():
        print(f"best box mAP {result['best_bbox_mAP']:.4f}; weights in {config.output}")


if __name__ == "__main__":
    main()
