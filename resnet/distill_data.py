import time
import os

import torch

import transforms
from my_dataset_coco import CocoDetection, custom_collate_fn
from backbone import resnet50_fpn_backbone
from network_files import MaskRCNN
from train_utils import (
    GroupedBatchSampler,
    create_aspect_ratio_groups,
    init_distributed_mode,
    mkdir,
)
from torch.utils.data import Subset
from train_utils.distributed_utils import (
    init_seeds,
    MetricLogger,
    is_main_process,
)
from train_utils.coco_eval import EvalCOCOMetric
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import collections
import json
import torchvision.utils as vutils
from typing import Tuple, List
from network_files.transform import GeneralizedRCNNTransform
from copy import deepcopy
from draw_box_utils import draw_objs
import numpy as np
from torchvision import transforms as visionTransform


@torch.no_grad()
# input should be tensor
def predict(
    model,
    inputs,
    device,
    original_image_sizes,
    save_path,
    image_idx,
    category_index,
):
    model.eval()
    bs = inputs.shape[0]
    with torch.no_grad():
        for i in range(bs):
            img = inputs[i].clone().unsqueeze(0)
            predictions = model(img.to(device))[0]

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            predict_mask = predictions["masks"].to("cpu").numpy()
            predict_mask = np.squeeze(predict_mask, axis=1)

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
                return

            print(f"boxes num is {len(predict_boxes)}")
            to_pil_image = visionTransform.ToPILImage()
            image = to_pil_image(img.squeeze(0))

            plot_img = draw_objs(
                image,
                boxes=predict_boxes,
                classes=predict_classes,
                scores=predict_scores,
                masks=predict_mask,
                category_index=category_index,
                line_thickness=3,
                font="arial.ttf",
                font_size=20,
            )

            plot_img.save(
                f"{save_path}/image{image_idx * bs + i}_gpu{args.rank}.jpg"
            )


@torch.no_grad()
def load_metric(coco):
    det_metric = EvalCOCOMetric(
        coco, iou_type="bbox", results_file_name="det_results.json"
    )
    seg_metric = EvalCOCOMetric(
        coco, iou_type="segm", results_file_name="seg_results.json"
    )

    return det_metric, seg_metric


@torch.no_grad()
def metric(
    model, inputs, targets, device, original_image_sizes, det_metric, seg_metric
):
    print("arrive1")
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")

    # 当使用CPU时，跳过GPU相关指令
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    model_time = time.time()
    output = model(inputs, None, original_image_sizes)

    output = [{k: v.to(cpu_device) for k, v in t.items()} for t in output]
    model_time = time.time() - model_time

    det_metric.update(targets, output)
    seg_metric.update(targets, output)
    metric_logger.update(model_time=model_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 同步所有进程中的数据
    det_metric.synchronize_results()
    seg_metric.synchronize_results()

    if is_main_process():
        coco_info = det_metric.evaluate()
        seg_info = seg_metric.evaluate()
    else:
        coco_info = None
        seg_info = None

    return coco_info, seg_info


class Batchhook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.running_mean_record = module.running_mean.clone()
        self.running_var_record = module.running_var.clone()

    def hook_fn(self, module, input, output):
        module.running_mean.data.copy_(self.running_mean_record.data)
        module.running_var.data.copy_(self.running_var_record.data)

        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = (
            input[0]
            .permute(1, 0, 2, 3)
            .contiguous()
            .view([nch, -1])
            .var(1, unbiased=False)
        )

        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2
        )

        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
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
    loss_var_l1 = (
        (diff1.abs() / 255.0).mean()
        + (diff2.abs() / 255.0).mean()
        + (diff3.abs() / 255.0).mean()
        + (diff4.abs() / 255.0).mean()
    )
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


# teacher model: mode='exact' ; student model: mode='lsq_sym'
def create_model(
    num_classes,
    load_pretrain_weights=True,
    mode="exact",
    a_bits=8,
    w_bits=8,
    distill=False,
):
    # 如果GPU显存很小，batch_size不能设置很大，建议将norm_layer设置成FrozenBatchNorm2d(默认是nn.BatchNorm2d)
    # FrozenBatchNorm2d的功能与BatchNorm2d类似，但参数无法更新
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    # backbone = resnet50_fpn_backbone(norm_layer=FrozenBatchNorm2d,
    #                                  trainable_layers=3)
    # resnet50 imagenet weights url: https://download.pytorch.org/models/resnet50-0676ba61.pth
    backbone = resnet50_fpn_backbone(
        pretrain_path="resnet50.pth",
        trainable_layers=3,
        mode=mode,
        a_bits=a_bits,
        w_bits=w_bits,
    )
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        mode="exact",
        a_bits=a_bits,
        w_bits=w_bits,
        distill=distill,
    )

    if load_pretrain_weights:
        # coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        weights_dict = torch.load("./mask_rcnn_weights.pth", map_location="cpu")

        print("MaskRCNN Network")
        print(model.load_state_dict(weights_dict, strict=False))

    return model


def prepare(args):
    device = torch.device(args.device)
    start_shift_size, calibration_size, mode, num_bits, a_bits, w_bits = (
        args.start_shift_size,
        args.calibration_size,
        args.mode,
        args.num_bits,
        args.a_bits,
        args.w_bits,
    )
    # Data loading code
    print("Loading data")

    data_transform = {
        "train": transforms.Compose(
            [transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5)]
        ),
    }

    COCO_root = args.data_path

    # load train data set
    # coco2017 -> annotations -> instances_train2017.json
    train_dataset = CocoDetection(COCO_root, "train", data_transform["train"])
    coco = train_dataset.coco

    # get a subset
    if calibration_size != 0:
        print(f"before calibration: dataset len is {len(train_dataset)}")
        train_dataset = Subset(
            train_dataset,
            range(start_shift_size, start_shift_size + calibration_size),
        )
        print(f"after calibration: dataset len is {len(train_dataset)}")

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

    if args.aspect_ratio_group_factor >= 0:
        # 统计所有图像比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(
            train_dataset, k=args.aspect_ratio_group_factor
        )
        train_batch_sampler = GroupedBatchSampler(
            train_sampler, group_ids, args.batch_size
        )
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True
        )

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.workers,
        collate_fn=custom_collate_fn,
    )

    model = create_model(
        num_classes=args.num_classes + 1,
        load_pretrain_weights=args.pretrain,
        mode="exact",
        a_bits=a_bits,
        w_bits=w_bits,
        distill=True,
    )
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )

    verifier = None
    if args.verifier:
        verifier = create_model(
            num_classes=args.num_classes + 1,
            load_pretrain_weights=args.pretrain,
            mode="exact",
            a_bits=a_bits,
            w_bits=w_bits,
        )
        verifier.to(device)
        if args.distributed:
            verifier = torch.nn.parallel.DistributedDataParallel(
                verifier, device_ids=[args.gpu]
            )

    img_list, target_list = [], []
    for i, [images, targets] in enumerate(data_loader):
        img_list.append(images)
        target_list.append(targets)

    return model, verifier, img_list, target_list, coco


def print_info(args, text):
    if args.rank in [-1, 0]:
        print(text)


def save_img(batch_tens, loc):
    """
    Saves a batch_tens of images to location loc
    """
    vutils.save_image(batch_tens, loc, normalize=True, scale_each=True)


def distill_data(
    teacher_model,
    verifier,
    img_list,
    target_list,
    args,
    det_metric=None,
    seg_metric=None,
):
    work_dir = args.output_dir
    prefix = f"{work_dir}/plt"
    with open(f"{work_dir}/hyp.json", "w") as fout:
        json.dump(vars(args), fout, indent=2)
    device = torch.device(args.device)
    hook_handles = []

    # add hook in batchnorm layers
    for i, layer in teacher_model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            hook_handles.append(Batchhook(layer))
    print_info(args, f"add bn hook in {len(hook_handles)} layers")
    print_info(args, "generating ditilled data")
    teacher_model.eval()

    save_every = 1000
    show_every = 50

    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    min_size = 800
    max_size = 1333
    transform = GeneralizedRCNNTransform(
        min_size, max_size, image_mean, image_std
    )

    label_json_path = "./coco91_indices.json"
    assert os.path.exists(label_json_path), (
        "json file {} dose not exist.".format(label_json_path)
    )
    with open(label_json_path, "r") as json_file:
        category_index = json.load(json_file)

    for idx, (img, target) in enumerate(zip(img_list, target_list)):
        iteration = 1
        loss_record = []
        iterations_per_layer = args.iterations

        images = list(image.to(device) for image in img)
        target = [{k: v.to(device) for k, v in t.items()} for t in target]
        inputs = list(torch.randn(image.size(), device=device) for image in img)

        # todo: transform function
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for image in images:
            val = image.shape[-2:]
            assert len(val) == 2  # 防止输入的是个一维向量
            original_image_sizes.append((val[0], val[1]))

        target_clone = deepcopy(target)
        target_save = deepcopy(target)
        image_clone = deepcopy(images)
        inputs, target = transform(inputs, target)
        images, _ = transform(images, target_clone)

        save_img(
            images.tensors,
            f"{prefix}/initial_images/batch{idx}_gpu{args.rank}.png",
        )
        inputs.tensors.requires_grad_()

        optimizer = optim.Adam(
            [inputs.tensors], lr=args.lr, betas=[0.9, 0.999], eps=1e-8
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iterations_per_layer, eta_min=0.0
        )
        clamp_max, clamp_min = (
            images.tensors.max().item(),
            images.tensors.min().item(),
        )

        for iteration_loc in tqdm(range(1, iterations_per_layer + 1)):
            lr_scheduler.step()
            optimizer.zero_grad()
            teacher_model.zero_grad()

            _, loss_dict = teacher_model(inputs, target, original_image_sizes)
            main_loss = sum(loss for loss in loss_dict.values())
            main_loss_copy = main_loss.clone().detach()
            main_loss = args.main_loss_multiplier * main_loss

            # prior loss
            loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs.tensors)
            loss_var_l1_copy = loss_var_l1.clone().detach()
            loss_var_l2_copy = loss_var_l2.clone().detach()
            loss_var_l1 = args.tv_l1 * loss_var_l1
            loss_var_l2 = args.tv_l2 * loss_var_l2

            # batch norm feature loss
            numLayers = len(hook_handles)
            loss_r_feature = torch.sum(
                torch.stack(
                    [mod.r_feature for mod in hook_handles[0:numLayers]]
                )
            )
            loss_r_feature_copy = loss_r_feature.clone().detach()
            loss_r_feature = args.r_feature * loss_r_feature

            # R_feature loss layer_1
            loss_r_feature_first = sum(
                [mod.r_feature for mod in hook_handles[:1]]
            )
            loss_r_feature_first_copy = loss_r_feature_first.clone().detach()
            loss_r_feature_first = args.first_bn_coef * loss_r_feature_first

            # combining losses
            loss_aux = (
                loss_var_l2
                + loss_var_l1
                + loss_r_feature_first
                + loss_r_feature
            )

            loss = main_loss + loss_aux

            if iteration % show_every == 0:
                debug_dict = {
                    key: value.item() for key, value in loss_dict.items()
                }
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
                        "debug_main": debug_dict,
                    }
                )

                print("------------iteration {}----------".format(iteration))
                print("************Train****************")
                print("total loss", loss.item())
                print("loss_r_feature", loss_r_feature.item())
                print("loss_r_feature_first", loss_r_feature_first.item())
                print("main loss", main_loss.item())
                print("l2 loss", loss_var_l2.item())
                print("************Main****************")
                for key in debug_dict.keys():
                    print(key, debug_dict[key])

            if iteration % save_every == 0:
                os.makedirs(
                    f"{prefix}/generated_images/iter{iteration}", exist_ok=True
                )
                save_img(
                    inputs.tensors,
                    f"{prefix}/generated_images/iter{iteration}/batch{idx}_gpu{args.rank}.png",
                )
            if iteration == iterations_per_layer:
                os.makedirs(
                    f"{work_dir}/weights/iter{iteration}", exist_ok=True
                )
                ckpt = {
                    "initial": images,
                    "generate": inputs,
                    "target": target,
                    "originSize": original_image_sizes,
                }
                torch.save(
                    ckpt,
                    f"{work_dir}/weights/iter{iteration}/batch{idx}_gpu{args.rank}.pth",
                )

            loss.backward()

            optimizer.step()

            # todo: check
            assert len(inputs.tensors.shape) == 4
            if args.do_clip:
                with torch.no_grad():
                    for c in range(3):
                        m, s = image_mean[c], image_std[c]
                        inputs.tensors[:, c] = torch.clamp(
                            inputs.tensors[:, c], -m / s, (1 - m) / s
                        )

            iteration += 1

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)

        with open(
            f"{work_dir}/loss/batch{idx}_gpu{args.rank}.json", "w"
        ) as fout:
            json.dump(loss_record, fout, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data-path", help="dataset")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument(
        "--num-classes", default=90, type=int, help="num_classes"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=4,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--lr",
        default=0.2,
        type=float,
        help="initial learning rate, 0.02 is the default value for training "
        "on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument(
        "--print-freq", default=50, type=int, help="print frequency"
    )
    parser.add_argument(
        "--output-dir", default="./multi_train", help="path where to save"
    )
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)

    parser.add_argument(
        "--world-size",
        default=4,
        type=int,
        help="number of distributed processes",
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--pretrain",
        type=bool,
        default=True,
        help="load COCO pretrain weights.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="exact",
        help="exact or quantize for conv layer",
    )
    parser.add_argument(
        "--calibration_size",
        type=int,
        default=0,
        help="Size of Calibration set",
    )
    parser.add_argument(
        "--start_shift_size",
        type=int,
        default=0,
        help="Size of Start Shift, enabling mutiple machines to work on data generation",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Global training seed"
    )
    parser.add_argument(
        "--num_bits",
        type=int,
        default=8,
        help="Quantization num bits for both activation and weight params",
    )
    parser.add_argument(
        "--a_bits",
        type=int,
        default=8,
        help="Quantization num bits for activation(If use, drop num_bits)",
    )
    parser.add_argument(
        "--w_bits",
        type=int,
        default=8,
        help="Quantization num bits for weight(If use, drop num_bits)",
    )

    # verifier model(student)
    parser.add_argument(
        "--verifier",
        action="store_true",
        help="predict batch with another model",
    )
    parser.add_argument(
        "--iterations",
        default=2000,
        type=int,
        help="number of iterations for DI optim",
    )

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
        "--do_clip",
        action="store_true",
        help="apply clip during model inversion",
    )

    args = parser.parse_args()

    if args.output_dir:
        mkdir(args.output_dir)

    init_seeds(args.seed)
    init_distributed_mode(args)
    prefix = f"{args.output_dir}/plt"
    os.makedirs(f"{prefix}/initial_images", exist_ok=True)
    os.makedirs(f"{prefix}/predict_images", exist_ok=True)
    os.makedirs(f"{args.output_dir}/loss", exist_ok=True)
    os.makedirs(f"{args.output_dir}/weights", exist_ok=True)
    model, verifier, imgs, targets, coco = prepare(args)
    print("load data")
    distill_data(model, verifier, imgs, targets, args, None, None)
