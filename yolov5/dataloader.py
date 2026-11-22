import torch
from utils.dataloaders import create_dataloader
import os
import argparse
from utils.general import (
    colorstr,
    check_yaml,
    check_file,
    check_img_size,
    check_dataset,
)
import random
from utils.torch_utils import (
    torch_distributed_zero_first,
)
from utils.plots import plot_images
import yaml
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyp",
        type=str,
        default=ROOT / "data/hyps/hyp.scratch-low.yaml",
        help="hyperparameters path",
    )
    parser.add_argument("--data", type=str, default=ROOT / "data/coco.yaml", help="dataset.yaml path")
    parser.add_argument(
        "--single-cls",
        action="store_true",
        help="train multi-class data as single-class",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="total batch size for all GPUs, -1 for autobatch",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        type=int,
        default=640,
        help="train, val image size (pixels)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="max dataloader workers (per RANK in DDP mode)",
    )
    return parser.parse_known_args()[0] if known else parser.parse_args()


opt = parse_opt()
opt.data, opt.hyp = (
    check_file(opt.data),
    check_yaml(opt.hyp),
)

hyp = opt.hyp
# Hyperparameters
if isinstance(hyp, str):
    with open(hyp, errors="ignore") as f:
        hyp = yaml.safe_load(f)  # load hyps dict

gs = 32
imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)
with torch_distributed_zero_first(LOCAL_RANK):
    data_dict = check_dataset(opt.data)  # check if None
val_path, names = data_dict["val"], data_dict["names"]
print("val path is", val_path)
print("names are", names)
val_loader = create_dataloader(
    val_path,
    imgsz,
    # opt.batch_size // WORLD_SIZE * 2,
    4,
    # 1,
    gs,
    opt.single_cls,
    hyp=hyp,
    cache=None,
    rect=True,
    rank=-1,
    workers=opt.workers * 2,
    pad=0.5,
    prefix=colorstr("val: "),
)[0]

# test of targets
for i, (imgs, targets, paths, shapes) in enumerate(val_loader):
    bs = imgs.shape[0]
    w, h = imgs.shape[3], imgs.shape[2]
    x, y = w / 2, h / 2
    # plot_boxes(imgs, targets, paths, f'./plt/tmp2/batch{i}/pic.png', names)
    plot_images(imgs, targets, paths, f"./plt/tmp2/batch{i}/pic.png", names)

    # import IPython
    # IPython.embed()
    targets_constrcuct = torch.tensor([])
    for j in range(bs):
        cls = random.randint(0, 80)
        tmp = torch.tensor([j, cls, x, y, w, h])
        targets_constrcuct = torch.cat([targets_constrcuct, tmp.unsqueeze(0)], dim=0)
    plot_images(imgs, targets_constrcuct, paths, f"./plt/tmp2/batchTest{i}/pic.png", names)
    if i >= 2:
        break
    pass

# targets_list = []
# lens_list = []
# for i, (imgs, targets, paths, shapes) in enumerate(val_loader):
#     plot_boxes(imgs, targets, paths, f'./plt/tmp/batch{i}/pic.png', names)
#     targets_list.append(targets)
#     lens_list.append(targets.shape[0])
#     # import IPython
#     # IPython.embed()
#     # if i >= 3:
#     #     break
# targets = torch.cat(targets_list, dim=0)
# with torch_distributed_zero_first(LOCAL_RANK):
#     torch.save(targets, 'targets2.pt')
#     with open('lens2.json', 'w') as fout:
#         json.dump(lens_list, fout, indent=2)
