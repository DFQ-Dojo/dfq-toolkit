import torch
from torch import nn
import numpy as np
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh  # type: ignore
from quant.qatops import LSQconv2d  # type: ignore

import torchvision  # type: ignore


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


def clip(image_tensor, use_fp16=False):
    """
    Adjust the input based on mean and variance of ImageNet.
    """
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


def jitter_targets(targets, xshift=0, yshift=0, img_shape=(320, 320)):
    """
    Apply horizontal & vertical jittering to the bboxes for given img_shape
    note: img_shape is in real world parameters, but bboxes are still between 0-1
    img_shape = (height, width)
    bbox shape = [center x, center y, w, h]
    """
    assert targets.shape[1] == 6
    targets = targets.clone().detach().cpu()
    height, width = img_shape
    xywh = targets[:, 2:]
    whwh = torch.tensor([width, height, width, height], dtype=torch.float32)
    xyxy = xywh2xyxy(xywh) * whwh

    # adjust the tbox
    xyxy[:, 0] += xshift
    xyxy[:, 2] += xshift
    xyxy[:, 1] += yshift
    xyxy[:, 3] += yshift

    # Limit co-ords
    xyxy[:, 0] = torch.clamp(xyxy[:, 0], min=0, max=width)
    xyxy[:, 2] = torch.clamp(xyxy[:, 2], min=0, max=width)
    xyxy[:, 1] = torch.clamp(xyxy[:, 1], min=0, max=height)
    xyxy[:, 3] = torch.clamp(xyxy[:, 3], min=0, max=height)

    # xyxy --> xywh
    xywh = xyxy2xywh(xyxy / whwh)
    targets[:, 2:] = xywh

    # remove boxes that have 0 area
    oof = (targets[:, -1] * targets[:, -2] * width * height) < 1
    targets = targets[~oof]

    return targets.to(targets.device)


def flip_targets(targets, horizontal=True, vertical=False):
    """horizontal and vertical flipping for `targets`."""
    assert targets.shape[1] == 6
    targets_flipped = targets.clone().detach()
    if horizontal:
        targets_flipped[:, 2] = 0.5 - (targets_flipped[:, 2] - 0.5)
    if vertical:
        targets_flipped[:, 3] = 0.5 - (targets_flipped[:, 3] - 0.5)
    return targets_flipped


def random_erase_masks(inputs_shape, return_cuda=True, device=None):
    """
    return a 1/0 mask with random rectangles marked as 0.
    shape should match inputs_shape
    """
    bs = inputs_shape[0]
    height = inputs_shape[2]
    width = inputs_shape[3]
    masks = []
    _rand_erase = torchvision.transforms.RandomErasing(
        p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0
    )
    for idx in range(bs):
        mask = torch.ones(3, height, width, dtype=torch.float32)
        mask = _rand_erase(mask)
        masks.append(mask)
    masks = torch.stack(masks)
    if return_cuda:
        masks = masks.to(device)
    return masks


def add_name(model: nn.Module):
    cnt = 0

    def recurve(child: nn.Module):
        nonlocal cnt
        for n, c in child.named_children():
            if isinstance(c, LSQconv2d):
                for name, m in model.named_modules():
                    if c is m:
                        c.name = name
                        cnt += 1
                        break
            else:
                recurve(c)

    recurve(model)
    print(f"add name in {cnt} layers")
    return model


def predictions_to_coco(output, inputs):
    """
    NMS predictions --> coco targets
    inputs: input image, only required for shape
    output: list w/ size = batchsize
            each element of list is a #predsx6 tensor, each prediction is of dims: xyxy, conf, cls.
            xyxy is in pixel space and CAN have negative vales, so int and clamp
    targets: [bidx, cls, x, y, w, h] where xywh is in 0-1 format
    """
    bsize, channels, height, width = inputs.shape
    assert height == width, "height and width are different"
    targets = []
    for batchIdx, preds in enumerate(output):
        if preds is not None:
            for box in preds:
                xyxy, conf, cls = box[0:4], box[4].item(), int(box[5].item())
                xyxy = torch.clamp(xyxy, min=0.0)
                xywh = xyxy2xywh(xyxy.view(1, -1))[0]
                xywh = xywh / height
                targets.append(
                    torch.tensor(
                        [
                            batchIdx,
                            cls,
                            conf,
                            xywh[0].item(),
                            xywh[1].item(),
                            xywh[2].item(),
                            xywh[3].item(),
                        ]
                    )
                )
    targets = torch.stack(targets, dim=0)
    return targets
