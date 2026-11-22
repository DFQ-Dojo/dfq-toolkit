import argparse
import torch
import os
import torch.nn as nn
from utils.torch_utils import (
    select_device,
    torch_distributed_zero_first,
    smart_DDP,
    de_parallel,
)
from utils.loss import ComputeLoss
from utils.utils import lr_cosine_policy
from utils.dataloaders import create_dataloader
import numpy as np
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import plot_images, output_to_target
from utils.general import (
    LOGGER,
    print_args,
    check_yaml,
    check_file,
    increment_path,
    colorstr,
    check_dataset,
    init_seeds,
    intersect_dicts,
    check_img_size,
    check_suffix,
    labels_to_class_weights,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
)
import torch.distributed as dist
import sys
from pathlib import Path
from datetime import timedelta
import yaml
from models.yolo import Model
from utils.downloads import attempt_download
import torch.optim as optim
import json
import torchvision.utils as vutils
import collections
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


def metric(inputs, targets, paths, shapes, model, device, nc, plots, work_dir, names, idx):
    def loss_func(output, target):
        compute_loss = ComputeLoss(model)
        loss, loss_item = compute_loss(output, target)
        return loss, loss_item

    def process_batch(detections, labels, iouv):
        """
        Return correct prediction matrix.

        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(iouv)):
            x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = (
                    torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
                )  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

    with torch.no_grad():
        preds, train_out = model(inputs)
        loss, loss_item = loss_func(train_out, targets.to(device))
        nb, _, height, width = inputs.shape
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to
        lb = []  # for autolabelling
        preds = non_max_suppression(preds, 0.001, 0.6, labels=lb, multi_label=True, agnostic=False, max_det=300)
        iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        stats = []
        seen = 0
        confusion_matrix = ConfusionMatrix(nc=nc)

        # Metrics
        p, r, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            predn = pred.clone()
            scale_boxes(inputs[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(inputs[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls

        # Plot images
        if plots:
            # plot_images(inputs, targets, paths, work_dir / f"val_labels_gpu{LOCAL_RANK}.jpg", names)  # labels
            plot_images(
                inputs,
                output_to_target(preds),
                paths,
                work_dir / f"pred_batch{idx}_gpu{LOCAL_RANK}.jpg",
                names,
            )  # pred

        # Compute metrics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=work_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
        pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
        LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
        if nt.sum() == 0:
            LOGGER.warning("WARNING ⚠️ no labels found, can not compute metrics without labels")
    return loss, loss_item, map50, map


class Batchhook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)

        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (
        (diff1.abs() / 255.0).mean()
        + (diff2.abs() / 255.0).mean()
        + (diff3.abs() / 255.0).mean()
        + (diff4.abs() / 255.0).mean()
    )
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


# if use multigpu, we draw pics seperately using different gpu
def plot_img(names, target, img, work_dir, bbox_name=None, Img_name=None):
    os.makedirs(f"{work_dir}", exist_ok=True)
    if WORLD_SIZE == 1:
        if bbox_name:
            plot_images(img, target, None, f"{work_dir}/{bbox_name}", names)
        if Img_name:
            vutils.save_image(
                img,
                f"{work_dir}/{Img_name}",
                normalize=True,
                scale_each=True,
                nrow=int(10),
            )
    elif WORLD_SIZE > 1:
        # plot_img = [torch.zeros_like(img) for _ in range(WORLD_SIZE)]
        # dist.all_gather(plot_img, img)
        # concat_img = torch.cat(plot_img, 0)

        # if target is not None:
        #     if isinstance(target, np.ndarray):
        #         target = torch.from_numpy(target).to(img.device)
        #     plot_target = [torch.zeros_like(target) for _ in range(WORLD_SIZE)]
        #     dist.all_gather(plot_target, target)
        #     concat_target = torch.cat(plot_target, 0)

        if bbox_name:
            filename, extension = bbox_name.split(".")
            plot_images(
                img,
                target,
                None,
                f"{work_dir}/{filename}_gpu{LOCAL_RANK}.{extension}",
                names,
            )
        if Img_name:
            filename, extension = Img_name.split(".")
            vutils.save_image(
                img,
                f"{work_dir}/{filename}_gpu{LOCAL_RANK}.{extension}",
                normalize=True,
                scale_each=True,
                nrow=int(10),
            )
    else:
        raise NotImplementedError


# setting id:
#   0: multi resolution, 2k iterations with low resolution, 1k at normal, ResNet50v1.5 works the best, ResNet50 is ok
#   1: 2k normal resolultion, for ResNet50v1.5; Resnet50 works as well
#   2: 20k normal resolution the closes to the paper experiments for ResNet50
# def get_distilled_data(teacher_model, verifier, target, img, path, shape, device, work_dir, setting_id, opt, idx, do_save=True):
#     prefix = f'{work_dir}/plt'

#     bs = opt.batch_size

#     # LOGGER.info('generating ditilled data')
#     best_cost = 1e4

#     # todo: setup target labels (use targets labels in val set to try first)
#     # todo: currently random sample batch_size samples from val set
#     # select_target = random.sample(targets, 1) # target: [batch_id, class_id, box axis], need to change batch_id
#     # idx = random.randint(0, len(targets)-1)
#     # LOGGER.info(f'choose idx {idx}')
#     # target = target.to(device)
#     # img = imgs[idx]
#     # path = paths[idx]
#     # shape = shapes[idx]
#     img = img.float() / 255
#     # vutils.save_image(img, f'{prefix}/generated_images/InitialImg_gpu{LOCAL_RANK}.png', normalize=True, scale_each=True, nrow=int(10))
#     plot_images(img, target, path, f"{prefix}/initial_images/bbox_batch{idx}_gpu{LOCAL_RANK}.jpg", teacher_model.names)  # labels

#     save_every = 100


#     hook_handles = []

#     for i, layer in teacher_model.named_modules():
#         if isinstance(layer, nn.BatchNorm2d):
#             hook_handles.append(
#                 Batchhook(layer))

#     # each conv2d should be followed by a batchnorm2d
#     # assert len(hooks) == len(bn_stats)
#     teacher_model.eval()

#     if setting_id == 0:
#         skipfirst = False
#     else:
#         skipfirst = True

#     iteration = 0
#     loss_record = []
#     for lr_it, lower_res in enumerate([2, 1]):
#         if lr_it == 0:
#             iterations_per_layer = 2001
#         else:
#             iterations_per_layer = 1001 if not skipfirst else 2001
#             if setting_id == 2:
#                 iterations_per_layer = 20001
#             if setting_id == 3:
#                 iterations_per_layer = 100001 # many iterations with early stopping
#             best_step = 0

#         if lr_it==0 and skipfirst:
#             continue

#         size = img.size()
#         inputs = torch.randint(high=255, size=size).to(device, non_blocking=True).float() / 255 # yolo data needs to be [0,1]
#         # inputs = img
#         inputs.requires_grad = True

#         if setting_id == 0:
#             optimizer = optim.Adam([inputs], lr=opt.lr, betas=[0.5, 0.9], eps = 1e-8)
#             do_clip = True
#         elif setting_id == 1:
#             optimizer = optim.Adam([inputs], lr=opt.lr, betas=[0.5, 0.9], eps = 1e-8)
#             do_clip = True
#         elif setting_id == 2 or setting_id == 3:
#             optimizer = optim.Adam([inputs], lr=opt.lr, betas=[0.9, 0.999], eps = 1e-8)
#             do_clip = False

#         lr_scheduler = lr_cosine_policy(opt.lr, 100, iterations_per_layer)

#         iterations_per_layer = 101 #todo:used for debug
#         for iteration_loc in tqdm(range(iterations_per_layer)):
#             lr_scheduler(optimizer, iteration_loc, iteration_loc)

#             inputs_jit = inputs

#             # Forward
#             optimizer.zero_grad()
#             teacher_model.zero_grad()

#             preds, train_out = teacher_model(inputs_jit)

#             # todo:model loss, to be finished
#             compute_loss = ComputeLoss(teacher_model)
#             loss, _ = compute_loss(train_out, target.to(device))

#             # prior loss
#             loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

#             # batch norm feature loss
#             rescale = [opt.first_bn_multiplier] + [1. for _ in range(len(hook_handles)-1)]
#             loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(hook_handles)])

#             # l2 loss on images
#             loss_l2 = torch.norm(inputs_jit.view(bs, -1), dim=1).mean()

#             # combining losses
#             loss_aux = opt.tv_l2 * loss_var_l2 + \
#                         opt.tv_l1 * loss_var_l1 + \
#                         opt.r_feature * loss_r_feature + \
#                         opt.l2 * loss_l2

#             loss = opt.main_loss_multiplier * loss + loss_aux
#             # hook_for_display = None
#             if iteration % save_every==0:
#                 if RANK in {-1, 0}:
#                     print("------------iteration {}----------".format(iteration))
#                     print("total loss", loss.item())
#                     print("loss_r_feature", loss_r_feature.item())
#                     print("main loss", compute_loss(train_out, target.to(device))[0].item())
#                 loss_record.append({'iteration': iteration, 'loss':loss.item(), 'r_feature_loss':loss_r_feature.item(), 'main_loss':compute_loss(train_out, target.to(device))[0].item(), 'sep_loss':compute_loss(train_out, target.to(device))[1].tolist()})

#             loss.backward()
#             optimizer.step()

#             # # clip color outlayers
#             # if do_clip:
#             #     inputs.data = clip(inputs.data)
#             inputs.data = torch.clamp(inputs.data, 0, 1)

#             # update loss
#             if best_cost > loss.item() or iteration == 1:
#                 best_inputs = inputs.data.clone()
#                 best_cost = loss.item()
#                 best_step = iteration

#             # early stopping
#             if setting_id == 3 and iteration - best_step >= opt.patience:
#                 break

#             # save image
#             if do_save and iteration % save_every == 0 and (save_every > 0):
#                 vutils.save_image(inputs, f'{prefix}/generated_images/output{int(iteration/save_every)}_gpu{LOCAL_RANK}.png', normalize=True, scale_each=True, nrow=int(10))
#                 LOGGER.info(f'iteration {iteration}, best step {best_step}')
#                 if verifier:
#                     save_dir = f'{prefix}/generated_images/batch{int(iteration/save_every)}'
#                     os.makedirs(save_dir, exist_ok=True)
#                     metric_loss, loss_item, map50, map = metric(inputs.clone(), target.to(device).clone(), path, shape, verifier, device, verifier.nc, True, Path(save_dir), verifier.names, 0)
#                     LOGGER.info(f'loss is {metric_loss}, items are {loss_item}, map50 is {map50}, map is {map}')

#             iteration += 1
#         # save best image
#         # plot_img(teacher_model.names, target, best_inputs, f'{prefix}/generated_images', f"outputBest.png")
#         LOGGER.info(f'best iteration is {best_step}')
#         vutils.save_image(best_inputs, f'{prefix}/generated_images/output_bacth{idx}_gpu{LOCAL_RANK}.png', normalize=True, scale_each=True, nrow=int(10))
#         # use verifier on best input and save
#         if verifier:
#             save_dir = f'{prefix}/predict_images'
#             metric_loss, loss_item, map50, map = metric(best_inputs.clone(), target.to(device).clone(), path, shape, verifier, device, verifier.nc, True, Path(save_dir), verifier.names, idx)
#             LOGGER.info(f'Best evaluation: loss is {metric_loss}, items are {loss_item}, map50 is {map50}, map is {map}')
#             loss_record.append({'Best iteration': best_step, 'loss':best_cost, 'validation':{'main_loss': metric_loss.item(), 'sep_loss': loss_item.tolist(), 'map50': map50, 'map': map}})
#             # ckpt = {
#             #     'input' : best_inputs,
#             #     'target' : target,
#             #     'path' : path,
#             #     'shape' : shape,
#             # }
#             # torch.save(ckpt, f'{save_dir}/best.pth')


#         # to reduce memory consumption by states of the optimizer we deallocate memory
#         optimizer.state = collections.defaultdict(dict)

#     # if RANK in {-1, 0}:
#     with open(f'{work_dir}/loss/loss_batch{idx}_gpu{LOCAL_RANK}.json', 'w') as fout:
#         json.dump(loss_record, fout, indent=2)


def distill_data(
    teacher_model,
    verifier,
    target_list,
    img_list,
    path_list,
    shape_list,
    device,
    work_dir,
    setting_id,
    opt,
    do_save=True,
):
    prefix = f"{work_dir}/plt"

    bs = opt.batch_size

    save_every = 100

    hook_handles = []

    for i, layer in teacher_model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            hook_handles.append(Batchhook(layer))
    teacher_model.eval()

    if setting_id == 0:
        skipfirst = False
    else:
        skipfirst = True

    # LOGGER.info('generating ditilled data')
    for idx, (target, img, path, shape) in enumerate(zip(target_list, img_list, path_list, shape_list)):
        best_cost = 1e4

        # todo: setup target labels (use targets labels in val set to try first)
        # todo: currently random sample batch_size samples from val set
        img = img.float() / 255
        # vutils.save_image(img, f'{prefix}/generated_images/InitialImg_gpu{LOCAL_RANK}.png', normalize=True, scale_each=True, nrow=int(10))
        plot_images(
            img,
            target,
            path,
            f"{prefix}/initial_bboxes/bbox_batch{idx}_gpu{LOCAL_RANK}.jpg",
            teacher_model.names,
        )  # labels
        for i in range(img.shape[0]):
            img_idx = opt.batch_size // WORLD_SIZE * idx + i
            vutils.save_image(
                img[i],
                f"{prefix}/initial_images/input_idx{img_idx}_gpu{LOCAL_RANK}.png",
                normalize=True,
                scale_each=True,
                nrow=int(10),
            )

        iteration = 0
        loss_record = []
        for lr_it, lower_res in enumerate([2, 1]):
            if lr_it == 0:
                iterations_per_layer = 2001
            else:
                iterations_per_layer = 1001 if not skipfirst else 2001
                if setting_id == 2:
                    iterations_per_layer = 20001
                if setting_id == 3:
                    iterations_per_layer = 100001  # many iterations with early stopping
                best_step = 0

            if lr_it == 0 and skipfirst:
                continue

            size = img.size()
            inputs = (
                torch.randint(high=255, size=size).to(device, non_blocking=True).float() / 255
            )  # yolo data needs to be [0,1]
            # inputs = img
            inputs.requires_grad = True

            if setting_id == 0:
                optimizer = optim.Adam([inputs], lr=opt.lr, betas=[0.5, 0.9], eps=1e-8)
            elif setting_id == 1:
                optimizer = optim.Adam([inputs], lr=opt.lr, betas=[0.5, 0.9], eps=1e-8)
            elif setting_id == 2 or setting_id == 3:
                optimizer = optim.Adam([inputs], lr=opt.lr, betas=[0.9, 0.999], eps=1e-8)

            lr_scheduler = lr_cosine_policy(opt.lr, 100, iterations_per_layer)

            # iterations_per_layer = 101 #todo:used for debug
            for iteration_loc in tqdm(range(iterations_per_layer)):
                lr_scheduler(optimizer, iteration_loc, iteration_loc)

                inputs_jit = inputs

                # Forward
                optimizer.zero_grad()
                teacher_model.zero_grad()

                preds, train_out = teacher_model(inputs_jit)

                # todo:model loss, to be finished
                compute_loss = ComputeLoss(teacher_model)
                loss, _ = compute_loss(train_out, target.to(device))

                # prior loss
                loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

                # batch norm feature loss
                rescale = [opt.first_bn_multiplier] + [1.0 for _ in range(len(hook_handles) - 1)]
                loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(hook_handles)])

                # l2 loss on images
                loss_l2 = torch.norm(inputs_jit.view(bs, -1), dim=1).mean()

                # combining losses
                loss_aux = (
                    opt.tv_l2 * loss_var_l2
                    + opt.tv_l1 * loss_var_l1
                    + opt.r_feature * loss_r_feature
                    + opt.l2 * loss_l2
                )

                loss = opt.main_loss_multiplier * loss + loss_aux
                # hook_for_display = None
                if iteration % save_every == 0:
                    if RANK in {-1, 0}:
                        print("------------iteration {}----------".format(iteration))
                        print("total loss", loss.item())
                        print("loss_r_feature", loss_r_feature.item())
                        print(
                            "main loss",
                            compute_loss(train_out, target.to(device))[0].item(),
                        )
                    loss_record.append(
                        {
                            "iteration": iteration,
                            "loss": loss.item(),
                            "r_feature_loss": loss_r_feature.item(),
                            "main_loss": compute_loss(train_out, target.to(device))[0].item(),
                            "sep_loss": compute_loss(train_out, target.to(device))[1].tolist(),
                        }
                    )

                loss.backward()
                optimizer.step()

                inputs.data = torch.clamp(inputs.data, 0, 1)

                # update loss
                if best_cost > loss.item() or iteration == 1:
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()
                    best_step = iteration

                # early stopping
                if setting_id == 3 and iteration - best_step >= opt.patience:
                    break

                # save image
                if do_save and iteration % save_every == 0 and (save_every > 0):
                    vutils.save_image(
                        inputs,
                        f"{prefix}/generated_images/output{int(iteration / save_every)}_gpu{LOCAL_RANK}.png",
                        normalize=True,
                        scale_each=True,
                        nrow=int(10),
                    )
                    LOGGER.info(f"iteration {iteration}, best step {best_step}")
                    if verifier:
                        save_dir = f"{prefix}/generated_images/batch{int(iteration / save_every)}"
                        os.makedirs(save_dir, exist_ok=True)
                        metric_loss, loss_item, map50, map = metric(
                            inputs.clone(),
                            target.to(device).clone(),
                            path,
                            shape,
                            verifier,
                            device,
                            verifier.nc,
                            True,
                            Path(save_dir),
                            verifier.names,
                            0,
                        )
                        LOGGER.info(f"loss is {metric_loss}, items are {loss_item}, map50 is {map50}, map is {map}")

                iteration += 1
            # save best image
            # plot_img(teacher_model.names, target, best_inputs, f'{prefix}/generated_images', f"outputBest.png")
            LOGGER.info(f"best iteration is {best_step}")
            for i in range(best_inputs.shape[0]):
                img_idx = opt.batch_size // WORLD_SIZE * idx + i
                vutils.save_image(
                    best_inputs[i],
                    f"{prefix}/generated_images/output_idx{img_idx}_gpu{LOCAL_RANK}.png",
                    normalize=True,
                    scale_each=True,
                    nrow=int(10),
                )
            # vutils.save_image(best_inputs, f'{prefix}/generated_images/output_batch{idx}_gpu{LOCAL_RANK}.png', normalize=True, scale_each=True, nrow=int(10))
            ckpt = {
                "input": best_inputs,
                "target": target,
                "path": path,
                "shape": shape,
            }
            torch.save(ckpt, f"{work_dir}/weights/best_batch{idx}_gpu{LOCAL_RANK}.pth")
            # use verifier on best input and save
            if verifier:
                save_dir = f"{prefix}/predict_images"
                metric_loss, loss_item, map50, map = metric(
                    best_inputs.clone(),
                    target.to(device).clone(),
                    path,
                    shape,
                    verifier,
                    device,
                    verifier.nc,
                    True,
                    Path(save_dir),
                    verifier.names,
                    idx,
                )
                LOGGER.info(
                    f"Best evaluation: loss is {metric_loss}, items are {loss_item}, map50 is {map50}, map is {map}"
                )
                loss_record.append(
                    {
                        "Best iteration": best_step,
                        "loss": best_cost,
                        "validation": {
                            "main_loss": metric_loss.item(),
                            "sep_loss": loss_item.tolist(),
                            "map50": map50,
                            "map": map,
                        },
                    }
                )

            # to reduce memory consumption by states of the optimizer we deallocate memory
            optimizer.state = collections.defaultdict(dict)

        # if RANK in {-1, 0}:
        with open(f"{work_dir}/loss/loss_batch{idx}_gpu{LOCAL_RANK}.json", "w") as fout:
            json.dump(loss_record, fout, indent=2)


def load_model(ckpt, weights, cfg, hyp, nc, device, names, opt):
    model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
    exclude = ["anchor"] if (cfg or hyp.get("anchors")) else []  # exclude keys
    csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load
    LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report

    cuda = device.type != "cpu"

    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."
        )
        model = torch.nn.DataParallel(model)

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = 0.0
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.names = names

    return model, imgsz, gs


def debug(verifier, save_dir, device):
    ckpt = torch.load(save_dir)
    best_inputs = ckpt["input"]
    target = ckpt["target"]
    path, shape = ckpt["path"], ckpt["shape"]
    if verifier:
        metric_loss, loss_item, map50, map = metric(
            best_inputs.clone(),
            target.clone(),
            path,
            shape,
            verifier,
            device,
            verifier.nc,
            False,
            Path(save_dir),
            verifier.names,
        )
        LOGGER.info(f"Best evaluation: loss is {metric_loss}, items are {loss_item}, map50 is {map50}, map is {map}")


def prepare(
    hyp,
    opt,
    device,
):
    save_dir, batch_size, weights, data, cfg = (
        Path(opt.save_dir),
        opt.batch_size,
        opt.weights,
        opt.data,
        opt.cfg,
    )

    # Directories
    w = save_dir / "weights"  # weights dir
    w.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Config
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(opt.data)  # check if None
    train_path, val_path = data_dict["train"], data_dict["val"]
    nc = int(data_dict["nc"])  # number of classes
    names = data_dict["names"]  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset

    check_suffix(weights, ".pt")  # check weights
    assert weights.endswith(".pt"), "weigths should be pretrained ckpt!"
    with torch_distributed_zero_first(LOCAL_RANK):
        weights = attempt_download(weights)  # download if not found locally
    ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak

    model, imgsz, gs = load_model(ckpt, weights, cfg, hyp, nc, device, names, opt)

    dataloader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls=False,
        hyp=hyp,
        augment=True,
        # augment=False,
        cache=None,
        rect=False,
        rank=LOCAL_RANK,
        workers=opt.workers,
        image_weights=False,
        quad=False,
        prefix=colorstr("train: "),
        shuffle=True,
        seed=opt.seed,
        max_target=opt.max_target,
        # max_target=1,
        min_target=1,
    )
    # create_dataloader(
    #     val_path,
    #     imgsz,
    #     opt.batch_size,
    #     gs,
    #     single_cls=False,
    #     hyp=hyp,
    #     augment=True,
    #     cache=None,
    #     rect=True,
    #     rank=-1,
    #     workers=opt.workers * 2,
    #     pad=0.5,
    #     prefix=colorstr("val: "),
    # )
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    verifier = None
    if opt.verifier:
        verifier, _, _ = load_model(ckpt, weights, cfg, hyp, nc, device, names, opt)
        verifier.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
        verifier.eval()

    img_list, target_list, path_list, shape_list = [], [], [], []
    for i, (imgs, targets, paths, shapes) in enumerate(dataloader):
        # imgs = imgs.to(device, non_blocking=True).float() / 255
        # if i * batch_size >= 2000:
        #     break
        img_list.append(imgs)
        target_list.append(targets)
        path_list.append(paths)
        shape_list.append(shapes)
    return model, verifier, target_list, img_list, path_list, shape_list


def parse_opt():
    parser = argparse.ArgumentParser(description="distill data")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="initial weights path")
    parser.add_argument("--epochs", type=int, default=10, help="total training epochs")
    parser.add_argument(
        "--hyp",
        type=str,
        default=ROOT / "data/hyps/hyp.scratch-low.yaml",
        help="hyperparameters path",
    )
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--project", default=ROOT / "runs/Distill", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="total batch size for all GPUs, -1 for autobatch",
    )
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        type=int,
        default=640,
        help="train, val image size (pixels)",
    )
    parser.add_argument("--lr", type=float, default=0.2, help="learning rate for optimization")
    parser.add_argument("--do_flip", action="store_true", help="apply flip during model inversion")
    parser.add_argument(
        "--first_bn_multiplier",
        type=float,
        default=10.0,
        help="additional multiplier on first bn layer of R_feature",
    )
    parser.add_argument("--l2", type=float, default=0.00001, help="l2 loss on the image")
    parser.add_argument(
        "--r_feature",
        type=float,
        default=0.05,
        help="coefficient for feature distribution regularization",
    )
    parser.add_argument(
        "--tv_l1",
        type=float,
        default=0.0,
        help="coefficient for total variation L1 loss",
    )
    parser.add_argument(
        "--tv_l2",
        type=float,
        default=0.0001,
        help="coefficient for total variation L2 loss",
    )
    parser.add_argument(
        "--main_loss_multiplier",
        type=float,
        default=1.0,
        help="coefficient for the main loss in optimization",
    )
    parser.add_argument(
        "--setting_id",
        default=0,
        type=int,
        help="settings for optimization: 0 - multi resolution, 1 - 2k iterations, 2 - 20k iterations",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="max dataloader workers (per RANK in DDP mode)",
    )
    parser.add_argument("--verifier", action="store_true", help="evaluate batch with another model")
    parser.add_argument(
        "--patience",
        type=int,
        default=10000,
        help="EarlyStopping patience (steps without improvement)",
    )
    parser.add_argument(
        "--max_target",
        type=int,
        default=None,
        help="filter out imgs with too many targets",
    )

    args = parser.parse_args()
    return args


def main():
    opt = parse_opt()
    if RANK in {-1, 0}:
        print_args(vars(opt))
    opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
        check_file(opt.data),
        check_yaml(opt.cfg),
        check_yaml(opt.hyp),
        str(opt.weights),
        str(opt.project),
    )
    assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=False))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),
        )

    model, verifier, targets, imgs, paths, shapes = prepare(opt.hyp, opt, device)
    prefix = f"{opt.save_dir}/plt"
    os.makedirs(f"{prefix}/initial_bboxes", exist_ok=True)
    os.makedirs(f"{prefix}/initial_images", exist_ok=True)
    os.makedirs(f"{prefix}/generated_images", exist_ok=True)
    os.makedirs(f"{prefix}/predict_images", exist_ok=True)
    os.makedirs(f"{opt.save_dir}/loss", exist_ok=True)
    distill_data(
        model,
        verifier,
        targets,
        imgs,
        paths,
        shapes,
        device,
        opt.save_dir,
        opt.setting_id,
        opt,
        do_save=False,
    )
    # for idx in range(len(targets)):
    # # for idx in range(3):
    #     target, img, path, shape = targets[idx], imgs[idx], paths[idx], shapes[idx]
    #     get_distilled_data(model, verifier, target, img, path, shape, device, opt.save_dir, opt.setting_id, opt, idx, do_save=False)
    #     torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
