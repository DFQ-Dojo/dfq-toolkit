import math
import sys
import time

import torch

import train_utils.distributed_utils as utils
from .coco_eval import EvalCOCOMetric
from network_files.transform import GeneralizedRCNNTransform
from typing import Tuple, List

image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]
min_size = 800
max_size = 1333
transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)


def CkptTrain_one_epoch(
    model,
    optimizer,
    ckpt_list,
    device,
    epoch,
    print_freq=50,
    warmup=False,
    scaler=None,
    train_kind="generate",
    kd_loss=None,
    distill=False,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)

    lr_scheduler = None
    if (
        epoch == 0 and warmup is True
    ):  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(ckpt_list) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(
            optimizer, warmup_iters, warmup_factor
        )

    # print('Start training!!!')
    mloss = torch.zeros(1).to(device)  # mean losses
    for i, ckpts in enumerate(
        metric_logger.log_every(ckpt_list, print_freq, header)
    ):
        ckpts = torch.load(ckpts, map_location=device)
        images, targets, original_image_sizes = (
            ckpts[train_kind],
            ckpts["target"],
            ckpts["originSize"],
        )

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        if kd_loss is None:
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                if distill:
                    _, loss_dict = model(images, targets, original_image_sizes)
                else:
                    loss_dict = model(images, targets, original_image_sizes)
                # todo:check一下loss_dict里面包含哪些项

                losses = sum(loss for loss in loss_dict.values())
            # reduce losses over all GPUs for logging purpose
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()
        else:
            losses, ori_loss, mse_loss, kl_loss = kd_loss(
                images, targets, original_image_sizes
            )
            losses_reduced = utils.reduce_loss(losses)
            ori_loss_reduced = utils.reduce_loss(ori_loss)
            mse_loss_reduced = utils.reduce_loss(mse_loss)
            loss_value = losses_reduced.item()
            print(
                f"Ori Loss is {ori_loss}, MSE Loss is {mse_loss}, KL Loss is {kl_loss}"
            )
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            # print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        if kd_loss is None:
            metric_logger.update(loss=losses_reduced)
        else:
            metric_logger.update(
                loss=losses_reduced,
                Ori_loss=ori_loss_reduced,
                MSE_loss=mse_loss_reduced,
            )
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    print_freq=50,
    warmup=False,
    scaler=None,
    check_ptq=False,
    data_loader_test=None,
    mask_rcnn=True,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)

    lr_scheduler = None
    if (
        epoch == 0 and warmup is True
    ):  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(
            optimizer, warmup_iters, warmup_factor
        )

    print("Start training!!!")
    mloss = torch.zeros(1).to(device)  # mean losses
    for i, [images, targets] in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2  # 防止输入的是个一维向量
            original_image_sizes.append((val[0], val[1]))

        images, targets = transform(images, targets)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # loss_dict = model(images, targets)
            loss_dict = model(images, targets, original_image_sizes)

            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # check PTQ result
        if check_ptq and epoch == 0 and i == 0:
            if mask_rcnn:
                det_info, seg_info = evaluate(model, data_loader_test, device)
            else:
                det_info, _ = evaluate(model, data_loader_test, device)
            exit(0)
            # print("****************det info******************")
            # print("************mAP: {}, mAP50: {}**************".format(det_info[0], det_info[1]))
            # print("****************seg info******************")
            # print("************mAP: {}, mAP50: {}**************".format(seg_info[0], seg_info[1]))

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device, distill=False, mask_rcnn=True):
    cpu_device = torch.device("cpu")
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    det_metric = EvalCOCOMetric(
        data_loader.dataset.coco,
        iou_type="bbox",
        results_file_name="det_results.json",
    )
    if mask_rcnn:
        seg_metric = EvalCOCOMetric(
            data_loader.dataset.coco,
            iou_type="segm",
            results_file_name="seg_results.json",
        )

    model.eval()
    with torch.no_grad():
        for image, targets in metric_logger.log_every(data_loader, 100, header):
            image = list(img.to(device) for img in image)

            # original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
            # for img in image:
            #     val = img.shape[-2:]
            #     assert len(val) == 2  # 防止输入的是个一维向量
            #     original_image_sizes.append((val[0], val[1]))

            # image, _ = transform(image, None)

            # 当使用CPU时，跳过GPU相关指令
            if device != torch.device("cpu"):
                torch.cuda.synchronize(device)

            model_time = time.time()
            # outputs = model(image, None, original_image_sizes)
            if distill:
                outputs, _ = model(image)
            else:
                outputs = model(image)

            outputs = [
                {k: v.to(cpu_device) for k, v in t.items()} for t in outputs
            ]
            model_time = time.time() - model_time

            det_metric.update(targets, outputs)
            if mask_rcnn:
                seg_metric.update(targets, outputs)
            metric_logger.update(model_time=model_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 同步所有进程中的数据
    det_metric.synchronize_results()
    if mask_rcnn:
        seg_metric.synchronize_results()

    if utils.is_main_process():
        coco_info = det_metric.evaluate()
        if mask_rcnn:
            seg_info = seg_metric.evaluate()
    else:
        coco_info = None
        if mask_rcnn:
            seg_info = None

    if mask_rcnn:
        return coco_info, seg_info
    else:
        return coco_info, None
