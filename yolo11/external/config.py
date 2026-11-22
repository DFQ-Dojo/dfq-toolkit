import logging
import os
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import (
    BaseSettings,
    CliApp,
    JsonConfigSettingsSource,
    SettingsConfigDict,
)
from typing import Optional

from quant.quant_mode import QuantizeMode  # type: ignore
from quant.kd_mode import KDMethods  # type: ignore

CONFIG_ENVIRON_KEY = "TRAIN_CONFIG"

CONFIG_FILE: Optional[Path] = (
    Path(os.environ[CONFIG_ENVIRON_KEY])
    if os.environ.get(CONFIG_ENVIRON_KEY)
    else None
)


class THyperparameter(BaseModel):
    optimizer_name: str = Field("Adam", description="name of optimizer.")

    lr0: float

    lrf: float

    momentum: float = Field(0.937, description="training momentum.")

    weight_decay: float

    box: float

    cls: float

    dfl: float


class TrainConfig(
    BaseSettings,
    cli_parse_args=True,
    cli_prog_name="Training Quantized Object Detection Models",
):
    model_config = SettingsConfigDict(json_file=CONFIG_FILE)

    # device for calibration of quantized models.
    device: str | list = Field("cuda", description="device to run this model.")

    # CONFIGURATIONS FOR DATASETS
    dataset_manifest: Path = Field(
        Path("coco.yaml"), description="dataset.yaml path."
    )

    # batch size when loading data.
    batch_size: int = Field(32, description="the batch size when loading data.")

    # This option controls the size of calibration set. If your calibration data set size is s,
    # the training stage will use fraction * s data to calibrate the network.
    # So if your dataset has 1000 images and you want to use only 200 images,
    # you can set this to 0.2.
    fraction: float = Field(
        1.0, description="use how many data to train. 0 - 1"
    )

    # If you are using pseudo data to calibrate the model, please set this to the location of your
    # generated path.
    #
    # for example, your generated dataset locates on ./runs/Distill/exp/
    # then you will need to set generated_weights_path to ./runs/Distill/exp/weights
    generated_weights_path: Optional[Path] = Field(
        None, description="checkpoint from generated images & targets."
    )

    # This is the upper limit of your calibration set size.
    # *NOTE*: This option only works if you are not using real data to calibrate the model.
    calibration_size: int = Field(10000, description="calibration set size")

    # CONFIGURATIONS FOR MODEL

    # This option controls the model of the QAT process.
    model: str = Field("yolo11s.pt", description="the pretrained model")

    # Please see `QuantizeMode` for more information.
    model_quantize_mode: QuantizeMode

    kd_method: Optional[KDMethods]

    # CONFIGURATIONS FOR TRAINING

    hyps: THyperparameter

    ptq: bool = Field(
        False,
        description="only calculate PTQ results. Please note PTQ ignores most of parameters.",
    )

    end_epochs: int = Field(3, description="max training epochs")

    patience: int = Field(100, description="early stop patience.")

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


def get_config() -> TrainConfig:
    try:
        config = CliApp.run(TrainConfig)
    except ValidationError as e:
        logging.fatal(e)
        exit(-1)
    return config
