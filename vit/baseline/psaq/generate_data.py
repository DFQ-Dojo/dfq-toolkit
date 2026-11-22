import logging
import os
from pathlib import Path
import random
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from baseline.psaq.utils import build_model, KernelDensityEstimator

from pydantic_settings import (
    BaseSettings,
    CliApp,
    JsonConfigSettingsSource,
    SettingsConfigDict,
)
from pydantic import Field, ValidationError


CONFIG_ENVIRON_KEY = "GENERATION_CONFIG"

CONFIG_FILE: Optional[Path] = (
    Path(os.environ[CONFIG_ENVIRON_KEY])
    if os.environ.get(CONFIG_ENVIRON_KEY)
    else None
)


class GenOptions(
    BaseSettings, cli_parse_args=True, cli_prog_name="QAT training"
):
    model_config = SettingsConfigDict(json_file=CONFIG_FILE)

    model_name: str = Field(description="generation model name")

    work_dir: str = Field(description="directory for saving logs")

    batch_size: int = Field(description="batch size when generating image.")

    calibration_size: int = Field(description="generation image size.")

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


model_zoo = {
    "deit_tiny": "deit_tiny_patch16_224",
    "deit_small": "deit_small_patch16_224",
    "deit_base": "deit_base_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_small": "swin_small_patch4_window7_224",
}


class AttentionMap:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.feature = output

    def remove(self):
        self.hook.remove()


def generate_data(batch_size, p_model, device, hooks, model_name):
    # Hook the attention

    # Init Gaussian noise
    img = torch.randn((batch_size, 3, 640, 640)).to(device)
    img.requires_grad = True

    # Init optimizer
    lr = 0.25 if "swin" in model_name else 0.20
    optimizer = optim.Adam([img], lr=lr, betas=[0.5, 0.9], eps=1e-8)

    # Set pseudo labels
    pred = torch.LongTensor(
        [random.randint(0, 999) for _ in range(batch_size)]
    ).to(device)
    var_pred = random.uniform(2500, 3000)  # for batch_size 32

    criterion = nn.CrossEntropyLoss()

    # Train for two epochs
    for lr_it in range(2):
        if lr_it == 0:
            iterations_per_layer = 500
            lim = 15
        else:
            iterations_per_layer = 500
            lim = 30

        lr_scheduler = lr_cosine_policy(lr, 100, iterations_per_layer)

        with tqdm(range(iterations_per_layer)) as pbar:
            for itr in pbar:
                pbar.set_description(f"Epochs {lr_it + 1}/{2}")

                # Learning rate scheduling
                lr_scheduler(optimizer, itr, itr)

                # Apply random jitter offsets (from DeepInversion[1])
                # [1] Yin, Hongxu, et al. "Dreaming to distill: Data-free knowledge transfer via deepinversion.", CVPR2020.
                off = random.randint(-lim, lim)
                img_jit = torch.roll(img, shifts=(off, off), dims=(2, 3))
                # Flipping
                flip = random.random() > 0.5
                if flip:
                    img_jit = torch.flip(img_jit, dims=(3,))

                # Forward pass
                optimizer.zero_grad()
                p_model.zero_grad()

                output = p_model(img_jit)

                loss_oh = criterion(output, pred)
                loss_tv = torch.norm(get_image_prior_losses(img_jit) - var_pred)

                loss_entropy = 0
                for itr_hook in range(len(hooks)):
                    # Hook attention
                    attention = hooks[itr_hook].feature
                    attention_p = attention.mean(dim=1)[:, 1:, :]
                    sims = torch.cosine_similarity(
                        attention_p.unsqueeze(1),
                        attention_p.unsqueeze(2),
                        dim=3,
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
                    dif_entropy_estimated = differential_entropy(
                        kde_estimate, x_plot
                    )
                    loss_entropy -= dif_entropy_estimated

                # Combine loss
                total_loss = loss_entropy + 1.0 * loss_oh + 0.05 * loss_tv

                # Do image update
                total_loss.backward()
                optimizer.step()

                # Clip color outliers
                img.data = clip(img.data)

    return img.detach()


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
        # image_tensor[:, c] = torch.clamp(image_tensor[:, c], 0, 1)
    return image_tensor


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


def save_generated(outputs, work_dir, idx, device="cuda"):
    torch.save(obj=outputs, f=f"{work_dir}/gen_{idx}_{device}.pt")


def main():
    opt = get_config()
    device = f"cuda:{opt.devices}"
    Path(opt.work_dir).mkdir(exist_ok=True, parents=True)

    p_model = build_model(model_zoo[opt.model_name], Pretrained=True).to(device)

    # hook the attention
    hooks = []
    if "swin" in opt.model_name:
        for m in p_model.backbone.stages:
            for n in range(len(m.blocks)):
                hooks.append(AttentionMap(m.blocks[n].attn.w_msa.matmul2))
    else:
        for m in p_model.backbone.blocks:
            hooks.append(AttentionMap(m.attn.matmul2))

    for idx in range(0, opt.calibration_size, opt.batch_size):
        outputs = generate_data(
            opt.batch_size, p_model, device, hooks, opt.model_name
        )
        save_generated(outputs, opt.work_dir, idx, device)


if __name__ == "__main__":
    main()
