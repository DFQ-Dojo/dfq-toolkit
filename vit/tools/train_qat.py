# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
from pathlib import Path
import time
import warnings
import logging

from pydantic import ValidationError, Field
from pydantic_settings import (
    BaseSettings,
    CliApp,
    JsonConfigSettingsSource,
    SettingsConfigDict,
)
from typing import Optional

import mmcv
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
import torch

from quant.quant_mode import QuantizeMode
from mmdet_custom.models.quant_swin import convert_swin_transformer
from mmdet_custom.models.quant_vit import convert_vit_baseline

CONFIG_ENVIRON_KEY = "TRAIN_CONFIG"

CONFIG_FILE: Optional[Path] = (
    Path(os.environ[CONFIG_ENVIRON_KEY])
    if os.environ.get(CONFIG_ENVIRON_KEY)
    else None
)


class QatOptions(
    BaseSettings, cli_parse_args=True, cli_prog_name="QAT training"
):
    model_config = SettingsConfigDict(json_file=CONFIG_FILE)

    local_rank: int = Field(0)

    work_dir: str = Field(description="path to save logs")

    # MODEL

    config: str = Field(description="model configuration path")

    pretrained_path: str = Field(description="the pretrained weight")

    quant_mode: QuantizeMode

    # TRAINING

    no_validate: bool = Field(
        description="whether or not to evaluate the checkpoint during training"
    )

    gpus: int = Field(description="number of gpus to use")

    seed: int = Field(0, description="random seed")

    deterministic: bool = Field(
        True, description="make the training procedure deterministic."
    )

    launcher: str = Field("none", description="on distributed training")

    # DATASETS

    dataset_path: str = Field(description="path of dataset configurations.")

    pseudo_data: Optional[str] = Field(None, description="path of pseudo data")

    # OVERRIDES

    lr: float = Field(1e-5, description="overrided learning rate")

    max_epochs: int = Field(15, description="max epochs")

    # KD configs
    enable_kd: bool = Field(False, description="enable knowledge distillation.")

    kd_modules: str = Field("block", description="places to hook MSE")

    original_loss_weight: float = Field(
        0.1, description="kd original loss weight"
    )

    kd_loss_weight: float = Field(0.1, description="weight of KL divergence.")

    mse_loss_weight: float = Field(0.1, description="mse loss weight")

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


def get_config() -> QatOptions:
    try:
        config = CliApp.run(QatOptions)
    except ValidationError as e:
        logging.fatal(e)
        exit(-1)
    return config


def quantize_model(config_name, model, quantize_mode: QuantizeMode):
    """Quantize the selected model

    Parameters
    ----------
    config_name : str
        The model name
    model : nn.Module
        The model
    quantize_mode : QuantizeMode
        Quantize mode

    Returns
    -------
    nn.Module
        Quantized model.

    Raises
    ------
    NotImplementedError
        If the model does not support quantization.
    """
    if "deit" in config_name:
        model.backbone = convert_vit_baseline(model.backbone, quantize_mode)
        return model
    elif "swin" in config_name:
        model.backbone = convert_swin_transformer(model.backbone, quantize_mode)
        return model
    else:
        raise NotImplementedError("not implemented this conversion.")


def build_pretrained_model(cfg, pretrained_path):
    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    _checkpoint = load_checkpoint(model, pretrained_path, "cpu")
    return model


def build_kd_loss(
    student,
    teacher,
    kd_module,
    original_loss_weight,
    kd_loss_weight,
    mse_loss_weight,
):
    """
    Construct and return a student model with a combined loss function that includes knowledge distillation.

    Parameters
    ----------
    student : nn.Module
        The student model to be trained.
    teacher : nn.Module
        A full-precision teacher model with the same architecture as the student.
    kd_module : str
        The knowledge distillation alignment module to use.
    original_loss_weight : float
        Weight for the original task loss (e.g., cross-entropy).
    kd_loss_weight : float
        Weight for the distillation loss (e.g., KL divergence or feature mimicking).
    mse_loss_weight : float
        Weight for the mean squared error (MSE) loss, typically used for feature alignment.

    Returns
    -------
    nn.Module
        The student model wrapped with a knowledge distillation loss function.
    """
    from tools.kd.kd_loss import KDLoss
    from types import MethodType

    kd_loss = KDLoss(
        student=student,
        teacher=teacher,
        kd_module=kd_module,
        original_loss_weight=original_loss_weight,
        kd_loss_weight=kd_loss_weight,
        mse_loss_weight=mse_loss_weight,
    )
    kd_loss.enable_kd()

    original_loss_function = student.train_step

    def train_step(self, sample, optimizer):
        with torch.no_grad():
            teacher.train_step(sample, optimizer)
        ori_loss = original_loss_function(sample, optimizer)
        loss, loss_items = kd_loss(ori_loss["loss"])
        ori_loss["loss"] = loss
        ori_loss["log_vars"]["detection_loss"] = loss_items["detection_loss"]
        ori_loss["log_vars"]["kldiv_loss"] = loss_items["kldiv_loss"]
        ori_loss["log_vars"]["mse_loss"] = loss_items["mse_loss"]
        ori_loss["log_vars"]["loss"] = loss_items["loss"]
        return ori_loss

    student.train_step = MethodType(train_step, student)

    original_train = student.train

    def train(self, mode=True):
        teacher.train(False)
        original_train(mode)

    student.train = MethodType(train, student)

    original_to = student.to

    def to(self, device):
        teacher.to(device)
        original_to(device)
        return self

    student.to = MethodType(to, student)

    original_cuda = student.cuda

    def cuda(self, device=None):
        teacher.cuda(device)
        original_cuda(device)
        return self

    student.cuda = MethodType(cuda, student)

    return student


def main():
    opt = get_config()

    cfg = Config.fromfile(opt.config)

    # overriding -----------
    cfg.optimizer.lr = opt.lr
    cfg.runner.max_epochs = opt.max_epochs
    cfg.data.samples_per_gpu = 1

    # work_dir is determined in this priority: CLI > segment in file > filename
    if opt.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = opt.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(opt.config))[0]
        )

    cfg.gpu_ids = range(1) if opt.gpus is None else range(opt.gpus)

    # init distributed env first, since logger depends on the dist info.
    if opt.launcher == "none":
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f"We treat {cfg.gpu_ids} as gpu-ids, and reset to "
                f"{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in "
                "non-distribute training time."
            )
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(opt.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    cfg.device = "cuda"  # fix 'ConfigDict' object has no attribute 'device'
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    with open(osp.join(cfg.work_dir, osp.basename(opt.config)), "w") as f:
        f.write(str(cfg._cfg_dict))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info
    meta["config"] = cfg._cfg_dict
    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg._cfg_dict}")

    # set random seeds
    seed = init_random_seed(opt.seed)
    logger.info(
        f"Set random seed to {seed}, deterministic: {opt.deterministic}"
    )
    set_random_seed(seed, deterministic=opt.deterministic)
    cfg.seed = seed
    meta["seed"] = seed
    meta["exp_name"] = osp.basename(opt.config)

    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    _checkpoint = load_checkpoint(model, opt.pretrained_path, "cpu")

    model = quantize_model(opt.config, model, opt.quant_mode)
    model.backbone.start_quantize()

    print(model)

    if opt.enable_kd:
        teacher = build_pretrained_model(cfg, opt.pretrained_path)
        model = build_kd_loss(
            student=model,
            teacher=teacher,
            kd_module=opt.kd_modules,
            original_loss_weight=opt.original_loss_weight,
            kd_loss_weight=opt.kd_loss_weight,
            mse_loss_weight=opt.mse_loss_weight,
        )
        print("built kd targets")

    dataset_config = Config.fromfile(opt.dataset_path)
    if dataset_config.data.train.type == "SynDataset":
        assert opt.pseudo_data, "you should provide the path of generated image"
        cfg.data.workers_per_gpu = 0
        dataset_config.data.train["syn_dir"] = opt.pseudo_data
    datasets = [build_dataset(dataset_config.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.test.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES,
        )
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not opt.no_validate),
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == "__main__":
    main()
