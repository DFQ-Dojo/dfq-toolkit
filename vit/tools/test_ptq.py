# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
from pathlib import Path
import time
import warnings
import logging
import numpy as np

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
from mmdet.apis import init_random_seed, set_random_seed, train_detector, single_gpu_test
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.core import encode_mask_results
import torch

from quant.quant_mode import QuantizeMode
from mmdet_custom.models.quant_swin_ptq import convert_swin_transformer
from mmdet_custom.models.quant_vit import convert_vit_baseline

CONFIG_ENVIRON_KEY = "TRAIN_CONFIG"

CONFIG_FILE: Optional[Path] = (
    Path(os.environ[CONFIG_ENVIRON_KEY]) if os.environ.get(CONFIG_ENVIRON_KEY) else None
)


class QatOptions(BaseSettings, cli_parse_args=True, cli_prog_name="QAT training"):

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

    original_loss_weight: float = Field(0.1, description="kd original loss weight")

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
    logger.info(f"Set random seed to {seed}, " f"deterministic: {opt.deterministic}")
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
    model=model.to(torch.device(f"cuda:{opt.gpus}" if torch.cuda.is_available() else "cpu"))
    model.eval()
    for module in model.modules():
        if hasattr(module, 'quantize'):
            print(f"Module {module} is in eval mode")

    # print("\n检查量化状态:")

    # # 统计量化层
    # quant_layers = []
    # enabled_layers = []
    # for name, module in model.named_modules():
    #     if hasattr(module, 'quantize'):
    #         quant_layers.append(name)
    #         if module.quantize:
    #             enabled_layers.append(name)

    # print(f"总量化层数: {len(quant_layers)}")
    # print(f"已启用量化层数: {len(enabled_layers)}")

    print("PTQ模式：跳过训练数据准备")
    
    # 只需要校准数据和验证数据
    # 1. 校准数据（从训练集取一小部分）
    dataset_config = Config.fromfile(opt.dataset_path)
    if dataset_config.data.train.type == "SynDataset":
        assert opt.pseudo_data, "you should provide the path of generated image"
        cfg.data.workers_per_gpu = 0
        dataset_config.data.train["syn_dir"] = opt.pseudo_data
    dataset = build_dataset(dataset_config.data.train)
    model.CLASSES = dataset.CLASSES

    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,  # 非分布式
        shuffle=False,  # 不打乱
        seed=cfg.seed,
        persistent_workers=False
    )

    processed = 0

    device = torch.device(f"cuda:{opt.gpus}" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= 32:  # 用100个batch校准
                break

            if isinstance(data, dict) and 'img' in data:
                # 提取图片
                img_data = data['img']
                    
                # 处理DataContainer
                from mmcv.parallel import DataContainer
                if isinstance(img_data, DataContainer):
                    img = img_data.data[0]  # 取第一个样本
                elif isinstance(img_data, (list, tuple)) and len(img_data) > 0:
                    img = img_data[0]
                else:
                    img = img_data
                    
                if isinstance(img, torch.Tensor):
                    # 移动到设备
                    img = img.to(device)
                        
                    # 确保是4D [batch, channel, height, width]
                    if img.dim() == 3:
                        img = img.unsqueeze(0)
                        
                    # 提取img_metas
                    img_metas = None
                    if 'img_metas' in data:
                        metas_data = data['img_metas']
                        if isinstance(metas_data, DataContainer):
                            if hasattr(metas_data.data, '__len__') and len(metas_data.data) > 0:
                                img_metas = metas_data.data[0]
                        elif isinstance(metas_data, (list, tuple)) and len(metas_data) > 0:
                            img_metas = metas_data[0]
                    
                    if isinstance(img_metas[0]['scale_factor'], float):
                        scale = img_metas[0]['scale_factor']
                        img_metas[0]['scale_factor'] = np.array([scale, scale, scale, scale])
                    
                    # 前向传播（收集统计）
                    if img_metas is not None:
                        _ = model(return_loss=False, img=[img], img_metas=[img_metas])
                    else:
                        _ = model(return_loss=False, img=[img])
                        
                    processed += 1
                        
                    if (i + 1) % 10 == 0:
                        print(f"  校准进度: {i+1}")
    
    print(f"校准完成，使用了 {processed} 个样本")
    logger.info(f"校准完成，使用了 {processed} 个样本")
    
    for module in model.modules():
        if hasattr(module, 'end_calibration'):
            module.end_calibration()
    # 结束校准
    print("\n结束校准...")

    val_dataset = build_dataset(dataset_config.data.test)

    val_dataloader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,  # 非分布式
        shuffle=False,  # 不打乱
        seed=cfg.seed,
        persistent_workers=False
    )
    print(val_dataset[0])

    processed = 0
    results = []
    metas = []

    from tqdm import tqdm
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataloader, desc="验证进度", total=len(val_dataloader))):

            if isinstance(data, dict) and 'img' in data:
                # if i >= 1:
                #     break  
                # 提取图片
                img_data = data['img']

                if len(img_data) != len(data['img_metas']):
                    raise ValueError("img and img_metas length mismatch")

                for j in range(len(img_data)):
                    img = img_data[j]
                    if isinstance(img, torch.Tensor):
                        img = img.to(device)
                        if img.dim() == 3:
                            img = img.unsqueeze(0) # [1, C, H, W]
                    img_metas = data['img_metas'][j].data[0]

                    result = model(return_loss=False, img=[img], img_metas=[img_metas])

                    results.append(result[0][0])
                    metas.append(img_metas[0])
                processed += 1
    print(metas[0])
    print(results[0])

    results_corrected = []

    for res, meta in zip(results, metas):
        scale_factor = meta['scale_factor']
        corrected = []
        for cls_bboxes in res:
            if cls_bboxes.shape[0] == 0:
                corrected.append(cls_bboxes)
                continue
            bboxes = cls_bboxes.copy()
            # x1, y1, x2, y2 / scale
            bboxes[:, 0] /= scale_factor[0]  # x1
            bboxes[:, 1] /= scale_factor[1]  # y1
            bboxes[:, 2] /= scale_factor[2]  # x2
            bboxes[:, 3] /= scale_factor[3]  # y2
            corrected.append(bboxes)
        results_corrected.append(corrected)











                # # 保证是 Tensor 且在正确设备
                # if isinstance(img, torch.Tensor):
                #     img = img.to(device)
                #     if img.dim() == 3:
                #         img = img.unsqueeze(0) # [1, C, H, W]


                #     # 取 img_metas
                #     img_metas_data = data['img_metas']
                #     if isinstance(img_metas_data, (list, tuple)):
                #         print(f"img_metas_data type: {type(img_metas_data)}")
                #         img_metas = img_metas_data[0]
                #         if isinstance(img_metas, DataContainer):
                #             print(f"img_metas is DataContainer")
                #             img_metas = img_metas.data[0]
                #     # print(img_metas)
                    
                        
                #     # 前向传播（收集统计）
                #     result = model(return_loss=False, img=[img], img_metas=[img_metas])

                        
                #     results.append(result[0][0])
                #     metas.append(img_metas[0])
                #     processed += 1
    # print(results[0])
    # detections = []
    # for cls_id, bboxes in enumerate(result[0][0]):
    #     for box in bboxes:
    #         x1, y1, x2, y2, score = box
    #         # 这里可以使用 x1, y1, x2, y2, score, cls_id
    #         detections.append({
    #             "cls": cls_id,
    #             "bbox": [x1, y1, x2, y2],
    #             "score": float(score)
    #         })
    # print(detections)
    # metrics = val_dataset.evaluate(
    #     results,
    #     metric=['bbox']
    # )

    # print(f"验证完成，使用了 {processed} 个样本")
    
    metrics = val_dataset.evaluate(results_corrected, metric=['bbox'])
    print("\n评估结果:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
            logger.info(f"{key}: {value:.4f}")
        elif isinstance(value, dict):
        # 有些指标会嵌套字典，例如各类别 AP
            print(f"{key}:")
            logger.info(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f" {sub_key}: {sub_value:.4f}")
                logger.info(f" {sub_key}: {sub_value:.4f}")





    # # 假设 batch=1
    # result = results[9] # 获取第一个样本结果
    # #print(result)

    # # 处理 img_metas，确保是 dict

    # img_metas_dict = metas[9] # 取第一个样本

    # # print(img_metas_dict)



    # ori_img = mmcv.imread(img_metas_dict['filename'])
    # target_h, target_w = img_metas_dict['img_shape'][:2]
    # img = mmcv.imresize(ori_img, (target_w, target_h)) # 先宽后高
    # print(img.shape)


    # # 反归一化
    # mean = img_metas_dict['img_norm_cfg']['mean']
    # std = img_metas_dict['img_norm_cfg']['std']


    # # 输出路径
    # out_file = "output2.jpg"


    # # 可视化结果并保存
    # model.show_result(
    #     img=img,          # 原图
    #     result=result,        # 推理结果
    #     score_thr=0.3,        # 阈值
    #     show=False,           # 不显示窗口
    #     out_file=out_file     # 保存路径
    # )


    # print(f"已保存可视化结果到 {out_file}")


    # return 0

    

    
if __name__ == "__main__":
    main()
