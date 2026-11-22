import argparse
import torch
import os
import torch.nn as nn
import random
from utils.torch_utils import (
    select_device,
    torch_distributed_zero_first,
    smart_DDP,
    de_parallel,
)
import torchvision
from utils.loss import ComputeLoss
from utils.utils import (
    jitter_targets,
    flip_targets,
    random_erase_masks,
    predictions_to_coco,
)
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
from tensorboardX import SummaryWriter

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


def save_img(batch_tens, loc):
    """
    Saves a batch_tens of images to location loc
    """
    print("Saving batch_tensor of shape {} to location: {}".format(batch_tens.shape, loc))
    vutils.save_image(batch_tens, loc, normalize=True, scale_each=True)


def distill_data(
    teacher_model,
    verifier,
    target_list,
    img_list,
    path_list,
    shape_list,
    device,
    work_dir,
    opt,
    do_save=True,
):
    assert verifier, "you should have a verifier!"
    prefix = f"{work_dir}/plt"
    with open(f"{work_dir}/hyp.json", "w") as fout:
        json.dump(vars(opt), fout, indent=2)
    bs = opt.batch_size

    save_every = 50

    hook_handles = []
    # txt_writer = open(os.path.join(f'{work_dir}/loss', 'losses.log'), 'wt')

    for i, layer in teacher_model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            hook_handles.append(Batchhook(layer))
    LOGGER.info(f"add bn hook in {len(hook_handles)} layers")
    teacher_model.eval()
    writer = SummaryWriter(os.path.join(f"{work_dir}/loss"))

    LOGGER.info("generating ditilled data")
    # import IPython
    # IPython.embed()
    for idx, (target, img, path, shape) in enumerate(zip(target_list, img_list, path_list, shape_list)):
        best_cost = 1e4

        # todo: setup target labels (use targets labels in val set to try first)
        # todo: currently random sample batch_size samples from val set
        img = img.float() / 255
        # save_img(img, f'{prefix}/initial_images/input_batch{idx}_gpu{LOCAL_RANK}.png')
        # plot_images(img, target, path, f"{prefix}/initial_bboxes/bbox_batch{idx}_gpu{LOCAL_RANK}.jpg", teacher_model.names)  # labels
        # for i in range(img.shape[0]):
        #     img_idx = opt.batch_size // WORLD_SIZE * idx + i
        #     save_img(img[i], f'{prefix}/initial_images/input_idx{img_idx}_gpu{LOCAL_RANK}.png')
        # if verifier:
        #     save_dir = ''
        #     metric_loss, loss_item, map50, map = metric(img.to(device).clone(), target.to(device).clone(), path, shape, verifier, device, verifier.nc, False, Path(save_dir), verifier.names, 0)
        #     LOGGER.info(f'Verifier Initial Real Image: loss is {metric_loss}, items are {loss_item}, map50 is {map50}, map is {map}')

        iteration = 1
        jitter = 20
        loss_record = []
        iterations_per_layer = opt.iterations

        size = img.size()
        LOGGER.info(f"size is {size}")
        inputs = (
            torch.randint(high=255, size=size).to(device, non_blocking=True).float() / 255
        )  # yolo data needs to be [0,1]
        # inputs = img.to(device)
        inputs.requires_grad = True
        # save_img(inputs, f'{prefix}/generated_images/generateInitial_batch{idx}_gpu{LOCAL_RANK}.png')
        # plot_images(inputs.detach().clone(), target, path, f"{prefix}/predict_images/generateBbox_batch{idx}_gpu{LOCAL_RANK}.jpg", teacher_model.names)

        # optimizer = optim.Adam([inputs], lr=opt.lr, betas=[0.0, 0.0], weight_decay=0.0)
        optimizer = optim.Adam([inputs], lr=opt.lr, betas=[0.9, 0.999], eps=1e-8)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations_per_layer, eta_min=0.0)

        # iterations_per_layer = 101 #todo:used for debug
        earlystop = False
        for iteration_loc in tqdm(range(1, iterations_per_layer + 1)):
            # lr_scheduler(optimizer, iteration_loc, iteration_loc)
            lr_scheduler.step()
            # cosine_scheduler.step()

            inputs_jit = inputs
            targets_jit = target.clone().detach()
            # targets_jit = target

            # Random Jitter

            if opt.do_jitter:
                off1, off2 = random.randint(-jitter, jitter), random.randint(-jitter, jitter)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))
                if any([off1, off2]):
                    height, width = inputs_jit.shape[2], inputs_jit.shape[3]
                    targets_jit = jitter_targets(targets_jit, off2, off1, img_shape=(height, width))
                    # plot_images(inputs_jit.detach().clone(), targets_jit, path, f"{prefix}/predict_images/generateBboxJitter_batch{idx}_gpu{LOCAL_RANK}.jpg", teacher_model.names)

            # Random horizontal flips
            flip = random.random() > 0.5
            if flip and opt.do_flip:
                inputs_jit = torch.flip(inputs_jit, dims=(3,))
                targets_jit = flip_targets(targets_jit, horizontal=True, vertical=False)
                # plot_images(inputs_jit.detach().clone(), targets_jit, path, f"{prefix}/predict_images/generateBboxFlip_batch{idx}_gpu{LOCAL_RANK}.jpg", teacher_model.names)

            # Random brightness & contrast
            if opt.rand_brightness:
                rand_brightness = torch.randn(inputs_jit.shape[0], 1, 1, 1).to(device) * 0.2
                inputs_jit = inputs_jit + rand_brightness
                # plot_images(inputs_jit.detach().clone(), targets_jit, path, f"{prefix}/predict_images/generateBboxBrightness_batch{idx}_gpu{LOCAL_RANK}.jpg", teacher_model.names)

            if opt.rand_contrast:
                rand_contrast = 1.0 + torch.randn(inputs_jit.shape[0], 1, 1, 1).to(device) * 0.1
                inputs_jit = inputs_jit * rand_contrast
                # plot_images(inputs_jit.detach().clone(), targets_jit, path, f"{prefix}/predict_images/generateBboxContrast_batch{idx}_gpu{LOCAL_RANK}.jpg", teacher_model.names)

            # Random erase mask
            if opt.random_erase:
                masks = random_erase_masks(inputs_jit.shape, return_cuda=True, device=device)
                inputs_jit = inputs_jit * masks
                # plot_images(inputs_jit.detach().clone(), targets_jit, path, f"{prefix}/predict_images/generateBboxErase_batch{idx}_gpu{LOCAL_RANK}.jpg", teacher_model.names)

            preds, train_out = teacher_model(inputs_jit)

            # todo:model loss, to be finished
            compute_loss = ComputeLoss(teacher_model)
            main_loss, _ = compute_loss(train_out, targets_jit.to(device))
            main_loss_copy = main_loss.clone().detach()
            main_loss = opt.main_loss_multiplier * main_loss

            # prior loss
            loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)
            loss_var_l1_copy = loss_var_l1.clone().detach()
            loss_var_l2_copy = loss_var_l2.clone().detach()
            loss_var_l1 = opt.tv_l1 * loss_var_l1
            loss_var_l2 = opt.tv_l2 * loss_var_l2

            # batch norm feature loss
            numLayers = len(hook_handles)
            loss_r_feature = torch.sum(torch.stack([mod.r_feature for mod in hook_handles[0:numLayers]]))
            loss_r_feature_copy = loss_r_feature.clone().detach()
            loss_r_feature = opt.r_feature * loss_r_feature

            # rescale = [opt.first_bn_multiplier] + [1. for _ in range(len(hook_handles)-1)]
            # loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(hook_handles)])

            # R_feature loss layer_1
            loss_r_feature_first = sum([mod.r_feature for mod in hook_handles[:1]])
            loss_r_feature_first_copy = loss_r_feature_first.clone().detach()
            loss_r_feature_first = opt.first_bn_coef * loss_r_feature_first

            # combining losses
            loss_aux = loss_var_l2 + loss_var_l1 + loss_r_feature_first + loss_r_feature

            loss = main_loss + loss_aux

            # Weighted Loss
            writer.add_scalar("weighted/total_loss", loss.item(), iteration)
            writer.add_scalar("weighted/task_loss", main_loss.item(), iteration)
            writer.add_scalar("weighted/prior_loss_var_l1", loss_var_l1.item(), iteration)
            writer.add_scalar("weighted/prior_loss_var_l2", loss_var_l2.item(), iteration)
            writer.add_scalar("weighted/loss_r_feature", loss_r_feature.item(), iteration)
            writer.add_scalar("weighted/loss_r_feature_first", loss_r_feature_first.item(), iteration)
            # writer.add_scalar("weighted/loss_img_stats", loss_img_stats.item(), iteration)
            # Unweighted loss
            writer.add_scalar("unweighted/task_loss", main_loss_copy.item(), iteration)
            writer.add_scalar("unweighted/prior_loss_var_l1", loss_var_l1_copy.item(), iteration)
            writer.add_scalar("unweighted/prior_loss_var_l2", loss_var_l2_copy.item(), iteration)
            writer.add_scalar("unweighted/loss_r_feature", loss_r_feature_copy.item(), iteration)
            writer.add_scalar(
                "unweighted/loss_r_feature_first",
                loss_r_feature_first_copy.item(),
                iteration,
            )
            writer.add_scalar("learning_rate", float(optimizer.param_groups[0]["lr"]), iteration)

            # hook_for_display = None
            if iteration % save_every == 0:
                # update loss of verifier
                if verifier:
                    save_dir = f"{prefix}/generated_images"
                    verifier_loss, verifier_loss_item, verifier_map50, verifier_map = metric(
                        inputs.clone(),
                        target.to(device).clone(),
                        path,
                        shape,
                        verifier,
                        device,
                        verifier.nc,
                        False,
                        Path(save_dir),
                        verifier.names,
                        0,
                    )
                    # FP sampling
                    if (
                        opt.box_sampler
                        and (iteration >= opt.box_sampler_warmup)
                        and (iteration <= opt.box_sampler_earlyexit)
                    ):
                        im_copy = inputs.clone().detach().cpu()
                        # get teacher's output
                        with torch.no_grad():
                            preds = verifier(inputs.clone())[0]
                            output = non_max_suppression(preds, 0.001, 0.6, classes=None, agnostic=False)
                        new_targets = predictions_to_coco(output, im_copy)
                        # print('have a look')
                        # import IPython
                        # IPython.embed()
                        new_targets = new_targets[new_targets[:, 2] > opt.box_sampler_conf]
                        new_targets = torch.index_select(
                            new_targets, dim=1, index=torch.tensor([0, 1, 3, 4, 5, 6])
                        )  # # remove confidence value

                        add_targets = torch.zeros((len(new_targets),), dtype=torch.long)
                        minus_targets = torch.zeros((len(target),), dtype=torch.long)
                        bs = im_copy.shape[0]
                        for batch_idx in range(bs):
                            _target = target[
                                target[:, 0] == batch_idx
                            ]  # todo:check if the float precision is a problem
                            _new_targets = new_targets[new_targets[:, 0] == batch_idx]
                            # add new target if iou is less than a given point
                            if _new_targets.shape[0] > 0:
                                ious = torchvision.ops.box_iou(
                                    xywh2xyxy(_new_targets[:, 2:]),
                                    xywh2xyxy(_target[:, 2:]),
                                )
                                max_ious, _ = torch.max(ious, dim=1)
                                _add_targets = (max_ious < opt.box_sampler_overlap_iou).long()
                                add_targets[new_targets[:, 0] == batch_idx] += _add_targets
                                # filter out samples in initial targets
                                if _target.shape[0] > 1:
                                    initial_ious, _ = torch.max(ious, dim=0)
                                    _minus_targets = (initial_ious < opt.box_sampler_overlap_iou).long()
                                    if _minus_targets.sum() == _target.shape[0]:
                                        max_idx = torch.argmax(initial_ious)
                                        _minus_targets[max_idx] = 0
                                        print(f"we select idx {max_idx}")
                                    minus_targets[target[:, 0] == batch_idx] += _minus_targets

                        # todo:continue finishing others
                        new_targets = new_targets[add_targets.bool()]
                        assert len(new_targets) == add_targets.sum().item()
                        areas = new_targets[:, -1] * new_targets[:, -2]
                        area_limits = (areas < opt.box_sampler_maxarea) * (areas > opt.box_sampler_minarea)
                        new_targets = new_targets[area_limits.bool()]

                        print(
                            "Fp sampling: Minus {} old targets to batch for iteration: {} ".format(
                                minus_targets.sum().item(), iteration
                            )
                        )
                        target = target[minus_targets == 0]
                        print(
                            "Fp sampling: Adding {} new targets to batch for iteration: {} ".format(
                                len(new_targets), iteration
                            )
                        )
                        target = torch.cat((target, new_targets), dim=0)

                    # early stopping
                    if best_cost > verifier_loss.item() or iteration < 100:
                        best_inputs = inputs.data.clone()
                        best_cost = verifier_loss.item()
                        best_step = iteration

                    loss_record.append(
                        {
                            "iteration": iteration,
                            "weighted": {
                                "loss": loss.item(),
                                "main_loss": main_loss.item(),
                                "l1_loss": loss_var_l1.item(),
                                "l2_loss": loss_var_l2.item(),
                                "r_feature_loss": loss_r_feature.item(),
                                "r_feature_first_loss": loss_r_feature_first.item(),
                            },
                            "unweighted": {
                                "main_loss": main_loss_copy.item(),
                                "l1_loss": loss_var_l1_copy.item(),
                                "l2_loss": loss_var_l2_copy.item(),
                                "r_feature_loss": loss_r_feature_copy.item(),
                                "r_feature_first_loss": loss_r_feature_first_copy.item(),
                            },
                            "verifier": {
                                "loss": verifier_loss.item(),
                                "map50": verifier_map50,
                                "map": verifier_map,
                            },
                        }
                    )

                    if iteration - best_step > opt.patience:
                        earlystop = True

                    if RANK in {-1, 0}:
                        print("------------iteration {}----------".format(iteration))
                        print("************Train****************")
                        print("total loss", loss.item())
                        print("loss_r_feature", loss_r_feature.item())
                        print("loss_r_feature_first", loss_r_feature_first.item())
                        print("main loss", main_loss.item())
                        print("l2 loss", loss_var_l2.item())
                        print("************Val****************")
                        print("val loss", verifier_loss.item())
                        print("map: ", verifier_map)
                        print("map50: ", verifier_map50)
                        print("************Bbox param****************")
                        print(f"bbox: {target[0]}")

            # Forward
            optimizer.zero_grad()
            teacher_model.zero_grad()

            loss.backward()
            optimizer.step()

            if opt.do_clip:
                with torch.no_grad():
                    inputs.clamp_(min=0.0, max=1.0)

            # save image
            assert not do_save
            if do_save and iteration % save_every == 0 and (save_every > 0):
                # vutils.save_image(inputs, f'{prefix}/generated_images/output{int(iteration/save_every)}_gpu{LOCAL_RANK}.png', normalize=True, scale_each=True, nrow=int(10))

                # LOGGER.info(f'iteration {iteration}, best step {best_step}')
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
                    # LOGGER.info(f'loss is {metric_loss}, items are {loss_item}, map50 is {map50}, map is {map}')
                    target_dict = {}
                    for i in range(inputs.shape[0]):
                        img_idx = opt.batch_size // WORLD_SIZE * idx + i
                        save_img(
                            inputs[i],
                            f"{prefix}/generated_images/batch{int(iteration / save_every)}/generated_idx{img_idx}_gpu{LOCAL_RANK}.png",
                        )
                        target_batch = target[target[:, 0] == i]
                        target_list = [teacher_model.names[int(targ[1].item())] for targ in target_batch]
                        target_dict[i + 1] = target_list
                    with open(
                        f"{prefix}/generated_images/batch{int(iteration / save_every)}/class.json",
                        "w",
                    ) as fout:
                        json.dump(target_dict, fout, indent=2)

            iteration += 1

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)

        # if earlystop:
        #     break

        # save best image
        # plot_img(teacher_model.names, target, best_inputs, f'{prefix}/generated_images', f"outputBest.png")
        LOGGER.info(f"best iteration is {best_step}")
        # for i in range(best_inputs.shape[0]):
        #     img_idx = opt.batch_size // WORLD_SIZE * idx + i
        #     save_img(best_inputs[i], f'{prefix}/generated_images/output_idx{img_idx}_gpu{LOCAL_RANK}.png')
        # save_img(best_inputs, f'{prefix}/generated_images/outputBest_batch{idx}_gpu{LOCAL_RANK}.png')
        # save_img(inputs, f'{prefix}/generated_images/outputLast_batch{idx}_gpu{LOCAL_RANK}.png')
        if opt.save_target:
            ckpt = {
                "target": target,
            }
        else:
            ckpt = {
                # 'initial' : img,
                "generate": best_inputs,
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
    save_dir, batch_size, weights, verifier_weights, cfg = (
        Path(opt.save_dir),
        opt.batch_size,
        opt.weights,
        opt.verifier_weights,
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
    # is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset

    check_suffix(weights, ".pt")  # check weights
    assert weights.endswith(".pt"), "weigths should be pretrained ckpt!"
    with torch_distributed_zero_first(LOCAL_RANK):
        verifier_weights = attempt_download(verifier_weights)
        weights = attempt_download(weights)  # download if not found locally
    ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
    verifier_ckpt = torch.load(verifier_weights, map_location="cpu")

    model, imgsz, gs = load_model(ckpt, weights, cfg, hyp, nc, device, names, opt)

    dataloader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls=False,
        hyp=hyp,
        # augment=True,
        augment=False,
        cache=None,
        rect=False,
        rank=LOCAL_RANK,
        workers=opt.workers,
        image_weights=False,
        quad=False,
        prefix=colorstr("train: "),
        shuffle=True,
        seed=opt.seed,
        # calibration_size=opt.calibration_size,
        # max_target=opt.max_target,
        # max_target=1,
        # min_target=1
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
        verifier, _, _ = load_model(verifier_ckpt, verifier_weights, cfg, hyp, nc, device, names, opt)
        verifier.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
        verifier.eval()

    sampling_path = opt.sampling_path
    img_path = opt.img_path
    if sampling_path:
        sample_list = []
        for rank in range(WORLD_SIZE):
            rank_list = [
                os.path.join(sampling_path, file) for file in os.listdir(sampling_path) if f"gpu{rank}" in file
            ]
            sample_list.append(rank_list)
    if img_path:
        img_ckpt_list = [os.path.join(img_path, file) for file in os.listdir(img_path)]
    img_list, target_list, path_list, shape_list = [], [], [], []
    for i, (imgs, targets, paths, shapes) in enumerate(dataloader):
        # imgs = imgs.to(device, non_blocking=True).float() / 255
        if i * batch_size >= opt.calibration_size:
            break
        if not img_path:
            img_list.append(imgs)
        else:
            idx = i * WORLD_SIZE + 1 - RANK
            fp_imgs = torch.load(img_ckpt_list[idx], map_location="cpu")["generate"]
            img_list.append(fp_imgs)
        if not sampling_path:
            target_list.append(targets)
        else:
            # bias = len(sample_list) // WORLD_SIZE
            # fp_targets = torch.load(sample_list[i + RANK*bias], map_location='cpu')['target']
            # idx = i * WORLD_SIZE + RANK
            fp_targets = torch.load(sample_list[RANK][i], map_location="cpu")["target"]
            target_list.append(fp_targets)
            # todo:need to have a check
            # if i >= 1 and RANK==0:
            #     # print(sample_list[:10])
            #     print('RANK', RANK, 'path', sample_list[RANK][i], flush=True)
            #     # print('RANK', RANK, 'target', targets[:20], flush=True)
            #     # print('RANK', RANK, 'fp targets', fp_targets[:20], flush=True)
            # if i >= 5:
            #     exit(0)
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
        "--first_bn_coef",
        type=float,
        default=0.0,
        help="additional regularization for the first BN in the networks, coefficient, useful if colors do not match",
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
    parser.add_argument("--iterations", default=2000, type=int, help="number of iterations for DI optim")
    parser.add_argument("--calibration_size", type=int, default=None, help="size of calibration set")
    parser.add_argument("--do-save", action="store_true", help="Whether to use do save")

    # verifier model(student)
    parser.add_argument("--verifier", action="store_true", help="evaluate batch with another model")
    parser.add_argument(
        "--verifier_weights",
        type=str,
        default=ROOT / "yolov5s.pt",
        help="verifier weights path",
    )

    # data augmentation
    parser.add_argument("--do_jitter", action="store_true", help="apply jitter during model inversion")
    parser.add_argument("--do_flip", action="store_true", help="apply flip during model inversion")
    parser.add_argument("--do_clip", action="store_true", help="apply clip during model inversion")
    parser.add_argument(
        "--rand_brightness",
        action="store_true",
        help="DA: randomly adjust brightness during optizn",
    )
    parser.add_argument(
        "--rand_contrast",
        action="store_true",
        help="DA: randomly adjust contrast during optizn",
    )
    parser.add_argument(
        "--random_erase",
        action="store_true",
        help="DA: randomly set rectangular regions to 0 during optizn",
    )

    # FP sampling
    parser.add_argument("--sampling-path", type=str, default=None, help="Use (Fp) sampling labels")
    parser.add_argument("--img-path", type=str, default=None, help="Use (Fp) sampling imgs")
    parser.add_argument("--save-target", action="store_true", help="Only save target")
    parser.add_argument("--box-sampler", action="store_true", help="Enable False positive (Fp) sampling")
    parser.add_argument(
        "--box-sampler-warmup",
        type=int,
        default=1000,
        help="warmup iterations before we start adding predictions to targets",
    )
    parser.add_argument(
        "--box-sampler-conf",
        type=float,
        default=0.5,
        help="confidence threshold for a prediction to become targets",
    )
    parser.add_argument(
        "--box-sampler-overlap-iou",
        type=float,
        default=0.2,
        help="a prediction must be below this overlap threshold with targets to become a target",
    )  # Increasing box overlap leads to more overlapped objects appearing
    parser.add_argument(
        "--box-sampler-minarea",
        type=float,
        default=0.0,
        help="new targets must be larger than this minarea",
    )
    parser.add_argument(
        "--box-sampler-maxarea",
        type=float,
        default=1.0,
        help="new targets must be smaller than this maxarea",
    )
    parser.add_argument(
        "--box-sampler-earlyexit",
        type=int,
        default=1000000,
        help="early exit at this iteration",
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
    print(f"do save is {opt.do_save}")
    distill_data(
        model,
        verifier,
        targets,
        imgs,
        paths,
        shapes,
        device,
        opt.save_dir,
        opt,
        do_save=opt.do_save,
    )


if __name__ == "__main__":
    main()
