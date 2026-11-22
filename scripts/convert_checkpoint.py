"""Convert a pickled Ultralytics or YOLOv5 checkpoint (`yolo11s.pt`, `yolov5s.pt`) into a plain
dfq checkpoint.

The output holds `{"yaml": model dict, "state_dict": float weights}` and loads without pickled
classes. The input is read with a restricted unpickler; see `dfq.models.yolo.checkpoints`.

    uv run python scripts/convert_checkpoint.py yolo11s.pt weights/yolo11s.pt
"""

import argparse
from pathlib import Path

import torch

from dfq.models.yolo.checkpoints import MODULE_PATHS, YOLOV5_PATHS, plain_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--yolov5", action="store_true", help="the source is a YOLOv5 checkpoint")
    arguments = parser.parse_args()
    mapping = YOLOV5_PATHS if arguments.yolov5 else MODULE_PATHS
    plain = plain_checkpoint(arguments.source, mapping)
    arguments.target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(plain, arguments.target)
    scale = plain["yaml"].get("scale")
    tensors = len(plain["state_dict"])
    print(f"wrote {arguments.target}: scale {scale}, {tensors} tensors")


if __name__ == "__main__":
    main()
