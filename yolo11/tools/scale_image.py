"""
This program scales the image to ImageNet/Standard range.

For example, if the data looks like:

[1, 2, 3]

Standard range will use min-max to convert it to

[0, 0.5, 1]

Arguments
---------

--syn_dir: The place of synthesized data

--scale:
    has three valid options
    std: min-max convert to [0, 1]
    image_net: min-max convert to [-m/s, (1-m)/s]
    clamp: clamp to [0, 1]
"""

import logging
from typing import List
import torch

from pydantic_settings import (
    BaseSettings,
    CliApp,
)
from pydantic import Field, ValidationError

from pathlib import Path


class GenOptions(BaseSettings, cli_parse_args=True, cli_prog_name="Generation"):
    syn_dir: str = Field(description="places of synthesized data")

    scale: str = Field(
        description="scale. std: [0, 1] image_net: [-m/s, (1-m) /s] clamp: clamp to [0, 1]"
    )


def get_all_pts(syn_dir: str) -> List[Path]:
    return [x for x in Path(syn_dir).iterdir() if x.name.endswith(".pt")]


def std_scale(img: torch.Tensor):
    min_val = img.amin(dim=(2, 3), keepdim=True)
    max_val = img.amax(dim=(2, 3), keepdim=True)

    scaled_img = (img - min_val) / (max_val - min_val).clamp(min=1e-5)

    return scaled_img


def image_net_scale(img: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    scaled_img = std_scale(img)
    return (scaled_img - mean) / std


def std_clamp(img: torch.Tensor):
    return img.clamp(0, 1)


scale_modes = {
    "std": std_scale,
    "image_net": image_net_scale,
    "clamp": std_clamp,
}


def scale_all_pts(syn_dir: str, scale: str):
    all_pts_path = get_all_pts(syn_dir)
    scaled_pts_path = Path(syn_dir) / scale
    scaled_pts_path.mkdir(exist_ok=True, parents=True)
    for pt_path in all_pts_path:
        sample = torch.load(pt_path, map_location="cpu")
        sample = scale_modes[scale](sample)
        torch.save(sample, scaled_pts_path / pt_path.name)


def get_config() -> GenOptions:
    try:
        config = CliApp.run(GenOptions)
    except ValidationError as e:
        logging.fatal(e)
        exit(-1)
    return config


if __name__ == "__main__":
    opt = get_config()
    scale_all_pts(syn_dir=opt.syn_dir, scale=opt.scale)
