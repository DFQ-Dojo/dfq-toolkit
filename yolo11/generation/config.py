import logging
import os
from pathlib import Path
from pydantic import Field, field_validator, ValidationError, BaseModel
from pydantic_settings import (
    BaseSettings,
    CliApp,
    JsonConfigSettingsSource,
    SettingsConfigDict,
)
import sys
from typing import Literal, Optional, Annotated
from enum import StrEnum

FILE = Path(__file__).resolve()

REPO_ROOT = FILE.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

REPO_ROOT = Path(os.path.relpath(REPO_ROOT, Path.cwd()))

CONFIG_ENVIRON_KEY = "GENERATION_CONFIG"

CONFIG_FILE: Optional[Path] = (
    Path(os.environ[CONFIG_ENVIRON_KEY])
    if os.environ.get(CONFIG_ENVIRON_KEY)
    else None
)


class InputLabelKind(StrEnum):
    REAL = "real"
    SAMPLED = "sampled"


class RealLabelConfig(BaseModel):
    kind: Literal[InputLabelKind.REAL]
    data: Path = Field(
        REPO_ROOT / "coco8.yaml", description="dataset.yaml path"
    )
    batch_size: int = Field(16, description="total batch size for all GPUs.")
    workers: int = Field(
        8, description="max dataloader workers (per RANK in DDP mode)"
    )


class SampledLabelConfig(BaseModel):
    kind: Literal[InputLabelKind.SAMPLED]
    sampling_weight_path: Path = Field(description="Use (Fp) sampling labels")
    batch_size: int = Field(description="batch size")


LabelConfig = Annotated[
    RealLabelConfig | SampledLabelConfig, Field(discriminator="kind")
]


class GHyperparameters(BaseModel):
    lr: float = Field(0.2, description="learning rate for optimization")

    r_feature: float = Field(
        0.01, description="coefficient for feature distribution regularization"
    )

    tv_l1: float = Field(
        0.01, description="coefficient for total variation L1 loss"
    )

    tv_l2: float = Field(
        0.001, description="coefficient for total variation L2 loss"
    )

    main_loss_multiplier: float = Field(
        1.0, description="coefficient for the main loss in optimization"
    )

    first_bn_coef: float = Field(
        0.05,
        description="additional regularization for the first BN in the networks, coefficient, useful if colors do not match",
    )


class BoxSamplerConfig(BaseModel):
    box_sampler: bool = Field(
        False, description="Enable False positive (Fp) sampling"
    )

    box_sampler_warmup: int = Field(
        1000,
        description="warmup iterations before we start adding predictions to targets",
    )

    box_sampler_conf: float = Field(
        0.5,
        description="confidence threshold for a prediction to become targets",
    )

    box_sampler_overlap_iou: float = Field(
        0.2,
        description="a prediction must be below this overlap threshold with targets to become a target",
    )

    box_sampler_minarea: float = Field(
        0.0, description="new targets must be larger than this minarea"
    )

    box_sampler_maxarea: float = Field(
        1.0, description="new targets must be smaller than this maxarea"
    )

    box_sampler_earlyexit: int = Field(
        1000000, description="early exit at this iteration"
    )


class GenerationConfig(
    BaseSettings,
    cli_parse_args=True,
    cli_prog_name="Calibration set generator",
):
    """This class contains all the options for task-specific calibration set synthesis."""

    model_config = SettingsConfigDict(json_file=CONFIG_FILE)

    # "dry run" means the run actually does not happen.
    # You can turn this on if you don't want this program to run.
    # We use this option to print some diagnosis information.
    dry_run: bool = Field(
        False,
        description="only prints out the config, rather than running this program.",
    )

    # GENERAL OPTIONS

    # This argument controls the save directory of your calibration set.
    # Your images will save to `yolo11/<project>/<name>`
    # By default the generated image is saved on `yolo11/runs/Distill` as you can see below.
    project: Path = Field(
        REPO_ROOT / "runs/Distill/exp", description="save to project/name"
    )

    # This argument controls the sub directory of your calibration set.
    name: str = Field("exp", description="save to project/name")

    # use "cpu" if you do not have GPUs
    # use "cuda" if you have one GPU
    # use "cuda:x" to choose the (x+1)-th GPU.
    device: str = Field(
        "cuda", description="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )

    # Teacher is the network where we distill from.
    # For example, if you want to calibrate a YOLO11s model, please set this to "yolo11s.pt"
    teacher_weights: Path = Field(
        REPO_ROOT / "yolo11n.pt", description="initial weights path"
    )

    # Relabel model is for adaptive label sampling
    # It should be consistent with teacher_weights under current implementation.
    relabel_weights: Path = Field(
        REPO_ROOT / "yolo11n.pt",
        description="initial weights path for relabel model. relabel model is also used to validate.",
    )

    # DATASET RELATING CONFIGURATIONS

    # To generate a image, we need a label to guide the network.
    # The label has two sources. You can use real labels from COCO2017 or pseudo label that generated using adaptive label sampling.
    # The former uses RealLabelConfig and the latter uses SampledLabelConfig.
    dataset_configs: Annotated[
        RealLabelConfig | SampledLabelConfig, Field(discriminator="kind")
    ]

    # The size of generated image will be <img_size, img_size>
    img_size: int = Field(640, description="train, val image size (pixels)")

    # The size of calibration set.
    calibration_size: int = Field(10000, description="size of calibration set")

    # GENERATION CONFIGURATIONS

    # During the calibration set generation, the optimizer will run on the image for fixed iterations. This option controls that.
    # Generally bigger iteration leds to finer images.
    iterations: int = Field(
        100, description="number of iterations for DI optim"
    )

    # hyperparameters such as learning rate and the coefficients of each part of the loss function.
    hyp: GHyperparameters

    # hyperparameters for adaptive label sampling.
    box_sampler_config: BoxSamplerConfig

    # SAVING OPTIONS

    # The optimization loop checks the mAP and loss for some fixed period.
    save_every: int = Field(100, description="save metrics frequency.")

    # If this option is turned on, the program will save a jpeg thumbnail image of each generated image.
    save_images: bool = Field(True, description="Whether to save plot images")

    # If this option is turned on, the program will not save generated calibration set image.
    skip_generated: bool = Field(False, description="Only save target")

    @field_validator("save_every")
    @classmethod
    def check_save_every(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                "<save_every> is not reasonable to take a negative number"
            )
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            JsonConfigSettingsSource(settings_cls),
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def get_config() -> GenerationConfig:
    try:
        config = CliApp.run(GenerationConfig)
        if config.dry_run:
            print(config)
            exit(0)
    except ValidationError as e:
        logging.fatal(e)
        exit(-1)
    return config
