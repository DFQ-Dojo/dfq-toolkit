import logging
import os
import os.path as osp
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm

import mmcv
from mmcv.parallel import DataContainer
from pydantic_settings import (
    BaseSettings,
    CliApp,
    JsonConfigSettingsSource,
    SettingsConfigDict,
)
from pydantic import Field, ValidationError

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import (
    build_dataloader,
    build_dataset,
    replace_ImageToTensor,
)
from mmdet.models import build_detector

from mmdet_custom.models.vit import Attention
from tools.annotate_model import annotate_model
from tools.kde import KernelDensityEstimator

CONFIG_ENVIRON_KEY = "GENERATION_CONFIG"

CONFIG_FILE: Optional[Path] = (
    Path(os.environ[CONFIG_ENVIRON_KEY])
    if os.environ.get(CONFIG_ENVIRON_KEY)
    else None
)


class GenOptions(BaseSettings, cli_parse_args=True, cli_prog_name="Generation"):
    model_config = SettingsConfigDict(json_file=CONFIG_FILE)

    config: str = Field(description="models config")

    pretrained_weights: str = Field(description="path for pretrained weights.")

    work_dir: str = Field(description="directory for saving logs")

    dataset_config: str = Field(description="directory of dataset.")

    workers_per_gpu: int = Field(description="workers per gpu.")

    batch_size: int = Field(description="batch size when generating image.")

    calibration_size: int = Field(description="generation image size.")

    available_gpus: int = Field(description="use how many gpus to do the work.")

    lr: float = Field(0.25)

    iterations: int = Field(2000)

    entropy_weight: float = Field(
        1.0, description="loss when generating image."
    )

    detection_loss_weight: float = Field(10, description="loss of detection.")

    tv_loss_weight: float = Field(
        0.001, description="loss of variance regularization."
    )

    devices: int = Field(0, description="available device.")

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


def get_config() -> GenOptions:
    try:
        config = CliApp.run(GenOptions)
    except ValidationError as e:
        logging.fatal(e)
        exit(-1)
    return config


class AttentionMap:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.feature = output

    def remove(self):
        self.hook.remove()


def differential_entropy(pdf, x_pdf):
    # pdf is a vector because we want to perform a numerical integration
    pdf = pdf + 1e-4
    f = -1 * pdf * torch.log(pdf)
    # Integrate using the composite trapezoidal rule
    ans = torch.trapz(f, x_pdf, dim=-1).mean()
    return ans


def get_image_prior_losses(inputs_jit):
    # Compute total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = (
        torch.norm(diff1)
        + torch.norm(diff2)
        + torch.norm(diff3)
        + torch.norm(diff4)
    )
    return loss_var_l2


def clip(image_tensor, use_fp16=False):
    # Adjust the input based on mean and variance
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(
            image_tensor[:, c], -m / s, (1 - m) / s
        )
    return image_tensor


def scale(img: torch.Tensor) -> torch.Tensor:
    """
    Min-max scales the img tensor along spatial dimensions while preserving channel-wise statistics.

    Args:
        img (torch.Tensor): Input tensor of shape (N, C, H, W)

    Returns:
        torch.Tensor: Min-max normalized tensor of the same shape as `img`
    """
    min_val = img.amin(dim=(2, 3), keepdim=True)  # Shape: (N, C, 1, 1)
    max_val = img.amax(dim=(2, 3), keepdim=True)  # Shape: (N, C, 1, 1)

    # Avoid division by zero
    scaled_img = (img - min_val) / (max_val - min_val).clamp(min=1e-5)

    return scaled_img


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def generate_data(
    batch_size: int,
    p_model: MMDataParallel,
    model_name: str,
    sample,
    weight_entropy: float,
    weight_detection: float,
    weight_tv: float,
    lr: float = 0.25,
    iterations: int = 5000,
    device=torch.device("cuda:0"),
    hooks: list = [],
):
    """Generate a batch of calibration pseudo data.

    Parameters
    ----------
    batch_size : int
    p_model : MMDataParallel
        The full precision teacher model
    model_name : str
    sample : dict
        a batch of labels from dataset
    weight_entropy : float
    weight_detection : float
    weight_tv : float
    lr : float, optional, by default 0.25
    iterations : int, optional, by default 5000
    device : optional, by default torch.device("cuda:0")
    hooks : list, optional, by default []
        Hooks are used to collect distribution information.

    Returns
    -------
    torch.Tensor
        the generated pseudo image.
    """

    # Init Gaussian noise
    img = torch.randn((batch_size, 3, 640, 640)).to(device)
    sample["img"] = DataContainer([img])
    img.requires_grad = True

    optimizer = torch.optim.Adam([img], lr=lr, betas=(0.5, 0.9), eps=1e-8)

    lr_scheduler = lr_cosine_policy(lr, 100, iterations)

    for itr in tqdm(range(iterations)):
        # Learning rate scheduling
        lr_scheduler(optimizer, itr, itr)

        # Forward pass
        optimizer.zero_grad()
        p_model.zero_grad()

        loss_od = p_model.val_step(sample)["loss"]

        loss_tv = torch.norm(get_image_prior_losses(img))

        loss_entropy = 0
        for itr_hook in range(len(hooks)):
            # Hook attention
            attention = hooks[itr_hook].feature
            attention_p = attention.mean(dim=1)[:, 1:, :]
            sims = torch.cosine_similarity(
                attention_p.unsqueeze(1), attention_p.unsqueeze(2), dim=3
            )

            # Compute differential entropy
            kde = KernelDensityEstimator(sims.view(batch_size, -1))
            start_p = sims.min().item()
            end_p = sims.max().item()
            x_plot = (
                torch.linspace(start_p, end_p, steps=10)
                .repeat(batch_size, 1)
                .to(device)
            )
            kde_estimate = kde(x_plot)
            dif_entropy_estimated = differential_entropy(kde_estimate, x_plot)
            loss_entropy -= dif_entropy_estimated

        # Combine loss
        total_loss = (
            weight_entropy * loss_entropy
            + weight_detection * loss_od
            + weight_tv * loss_tv
        )
        print(
            f"entropy loss {weight_entropy * loss_entropy}, detection loss {weight_detection * loss_od}, tv loss {weight_tv * loss_tv}, total loss {total_loss}"
        )

        # Do image update
        total_loss.backward()
        optimizer.step()

    sample["img"] = img.detach()
    return sample


def save_generated(outputs, work_dir, idx, device="cuda"):
    """Save generated images to work_dir

    Parameters
    ----------
    outputs : torch.Tensor
    work_dir : Path
    idx : int
    device : str, optional, by default "cuda"
    """
    torch.save(obj=outputs, f=f"{work_dir}/gen_{idx}_{device}.pt")


def setup_dataloader(dataset_config, batch_size: int, workers_per_gpu: int):
    import time

    dataset_config = Config.fromfile(dataset_config)
    if batch_size > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        dataset_config.data.train.pipeline = replace_ImageToTensor(
            dataset_config.data.train.pipeline
        )
    # build the dataloader
    dataset = build_dataset(dataset_config.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=True,
        seed=int(time.time_ns() + 12344321) % (2**31),
    )
    return data_loader, dataset


def setup_model(
    model_cfg, test_cfg, use_fp16: bool, pretrained_path: str, dataset
):

    print(model_cfg, test_cfg)
    model = build_detector(model_cfg, test_cfg=test_cfg)
    if use_fp16:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, pretrained_path, map_location="cpu")

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
    annotate_model(model)
    return model


def main():
    opt = get_config()

    cfg = Config.fromfile(opt.config)

    # allows not to create
    if opt.work_dir is not None:
        mmcv.mkdir_or_exist(osp.abspath(opt.work_dir))

    data_loader, dataset = setup_dataloader(
        dataset_config=opt.dataset_config,
        batch_size=opt.batch_size,
        workers_per_gpu=opt.workers_per_gpu,
    )

    model = setup_model(
        model_cfg=cfg.model,
        test_cfg=cfg.get("test_cfg"),
        use_fp16=cfg.get("fp16", False),
        pretrained_path=opt.pretrained_weights,
        dataset=dataset,
    )

    device = f"cuda:{opt.devices}"
    model.to(device)
    model = MMDataParallel(model, device_ids=[opt.devices])
    print(model)

    # Hook the attention
    hooks = []
    if "swin" in opt.config:
        for m in model.module.backbone.stages:
            for n in range(len(m.blocks)):
                hooks.append(AttentionMap(m.blocks[n].attn.w_msa.matmul2))
    else:
        for m in model.module.backbone.blocks:
            if isinstance(m.attn, Attention):
                hooks.append(AttentionMap(m.attn.matmul2))

    for idx, sample in enumerate(data_loader):
        if idx * opt.batch_size >= opt.calibration_size:
            break
        outputs = generate_data(
            batch_size=opt.batch_size,
            p_model=model,
            model_name=opt.config,
            sample=sample,
            lr=opt.lr,
            weight_entropy=opt.entropy_weight,
            weight_detection=opt.detection_loss_weight,
            weight_tv=opt.tv_loss_weight,
            iterations=opt.iterations,
            device=device,
            hooks=hooks,
        )
        save_generated(outputs, opt.work_dir, idx, device)


if __name__ == "__main__":
    main()
