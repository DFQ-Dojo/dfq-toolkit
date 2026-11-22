import torch
from pydantic_settings import (
    BaseSettings,
    CliApp,
)
from pydantic import Field, ValidationError

from pathlib import Path
import logging


class GenOptions(BaseSettings, cli_parse_args=True, cli_prog_name="Generation"):
    work_dir: str = Field(description="places of synthesized data")

    img_size: int = Field(description="generate how many images")


def get_config() -> GenOptions:
    try:
        config = CliApp.run(GenOptions)
    except ValidationError as e:
        logging.fatal(e)
        exit(-1)
    return config


def main(opt: GenOptions):
    work_dir = Path(opt.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, opt.img_size, 16):
        t = torch.rand(16, 3, 640, 640)
        torch.save(t, work_dir / f"batch_{i}.pt")


if __name__ == "__main__":
    opt = get_config()
    main(opt)
