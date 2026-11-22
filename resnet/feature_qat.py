import time
import os
import datetime

import torch

import transforms
from my_dataset_coco import CocoDetection, custom_collate_fn
from backbone import resnet50_fpn_backbone
from network_files import MaskRCNN
import train_utils.train_eval_utils as utils
from train_utils import (
    save_on_master,
    mkdir,
)
from train_utils.distributed_utils import init_seeds
from train_utils.kd_loss import KDLoss


def create_model(
    num_classes, load_pretrain_weights=True, mode="exact", num_bits=8
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
        num_bits=num_bits,
    )
    model = MaskRCNN(
        backbone, num_classes=num_classes, mode="exact", num_bits=num_bits
    )

    if load_pretrain_weights:
        # coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        weights_dict = torch.load("./mask_rcnn_weights.pth", map_location="cpu")
        # for k in list(weights_dict.keys()):
        #     if ("box_predictor" in k) or ("mask_fcn_logits" in k):
        #         del weights_dict[k]

        print("MaskRCNN Network")
        print(model.load_state_dict(weights_dict, strict=False))

    return model


def main(args):
    calibration_size, mode, check_ptq, num_bits, patience = (
        args.calibration_size,
        args.mode,
        args.check_ptq,
        args.num_bits,
        args.patience,
    )
    print(args)

    device = torch.device(args.device)

    # 用来保存coco_info的文件
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    det_results_file = f"{args.output_dir}/det_results{now}.txt"
    seg_results_file = f"{args.output_dir}/seg_results{now}.txt"

    # Data loading code
    print("Loading data")

    data_transform = {"val": transforms.Compose([transforms.ToTensor()])}

    COCO_root = args.data_path

    # train_dataset = CocoDetection(COCO_root, "train", data_transform["train"])
    # Trainloader
    ckpt_path = args.ckpt_path
    ckpts_list = [
        os.path.join(ckpt_path, item) for item in os.listdir(ckpt_path)
    ]

    # get a subset
    if calibration_size != 0:
        ckpt = torch.load(ckpts_list[0], map_location="cpu")
        # bs = ckpt[args.train_kind].tensors.shape[0]
        bs = ckpt["generate"].tensors.shape[0]
        print(f"before calibration: dataset len is {len(ckpts_list) * bs}")
        sublen = int(calibration_size / bs) + 1
        ckpts_list = ckpts_list[:sublen]
        print(f"after calibration: dataset len is {len(ckpts_list) * bs}")
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    # train_dataset = VOCInstances(data_root, year="2012", txt_name="train.txt")

    # load validation data set
    # coco2017 -> annotations -> instances_val2017.json
    val_dataset = CocoDetection(COCO_root, "val", data_transform["val"])
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    # val_dataset = VOCInstances(data_root, year="2012", txt_name="val.txt")

    print("Creating data loaders")
    batch_size = args.batch_size
    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, 8]
    )  # number of workers
    print("Using %g dataloader workers" % nw)

    data_loader_test = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=custom_collate_fn,
    )

    print("Creating model")
    # create model num_classes equal background + classes
    model = create_model(
        num_classes=args.num_classes + 1,
        load_pretrain_weights=args.pretrain,
        mode=mode,
        num_bits=num_bits,
    )
    model.to(device)
    if args.kd:
        teacher_model = create_model(
            num_classes=args.num_classes + 1,
            load_pretrain_weights=args.pretrain,
            mode="exact",
            num_bits=8,
        )
        teacher_model.to(device)

    # teacher_model.eval()
    # det_info, seg_info = utils.evaluate(teacher_model, data_loader_test, device=device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_steps, gamma=args.lr_gamma
    )

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(
            args.resume, map_location="cpu"
        )  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        utils.evaluate(model, data_loader_test, device=device)
        return

    kd_loss = None
    if args.kd:
        for param in teacher_model.parameters():
            param.requires_grad = False

        kd_loss = KDLoss(
            model,
            teacher_model,
            args.kd,
            args.module,
            args.ori_loss_weight,
            args.kd_loss_weight,
            args.mse_loss_weight,
            device,
        )

    train_loss = []
    learning_rate = []
    val_map = []

    print("Start training")
    start_time = time.time()
    # write into txt
    with open(det_results_file, "a") as f1, open(seg_results_file, "a") as f2:
        header = "{:<10} | {:<10} | {:<10}| {:<10}| {:<10}".format(
            "Epoch", "mAP", "mAP50", "Loss", "Lr"
        )
        f1.write(header + "\n")
        f2.write(header + "\n")
        f1.write("-" * len(header) + "\n")  # 添加分隔线
        f2.write("-" * len(header) + "\n")

    best_mAP = 0.0  # early stop
    best_epoch = -1
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        if args.kd:
            teacher_model.train()
        mean_loss, lr = utils.CkptTrain_one_epoch(
            model,
            optimizer,
            ckpts_list,
            device,
            epoch,
            args.print_freq,
            warmup=True,
            scaler=scaler,
            train_kind=args.train_kind,
            kd_loss=kd_loss,
        )

        # update learning rate
        lr_scheduler.step()

        # evaluate after every epoch
        if args.kd:
            teacher_model.eval()
        det_info, seg_info = utils.evaluate(
            model, data_loader_test, device=device
        )

        # 只在主进程上进行写操作
        early_stop = False
        # early stop
        mAP_metric = 0.9 * det_info[0] + 0.1 * det_info[1]
        # early_stop = False

        if mAP_metric > best_mAP:
            best_mAP = mAP_metric
            best_epoch = epoch
            print(f"Best epoch:{best_epoch}, Weighted mAP:{best_mAP}")
        elif epoch - best_epoch >= patience:
            print("Early Stop!!!")
            early_stop = True

        train_loss.append(mean_loss.item())
        learning_rate.append(lr)
        val_map.append(det_info[1])  # pascal mAP

        # write detection into txt
        with open(det_results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            # 只需要写入mAP / mAP50
            result_info = [
                f"{i:.4f}" for i in det_info[:2] + [mean_loss.item()]
            ] + [f"{lr:.6f}"]
            txt = "{:<10} | {:<10} | {:<10} | {:<10} | {:<10}".format(
                epoch, *result_info
            )
            f.write(txt + "\n")

        # write seg into txt
        with open(seg_results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [
                f"{i:.4f}" for i in seg_info[:2] + [mean_loss.item()]
            ] + [f"{lr:.6f}"]
            txt = "{:<10} | {:<10} | {:<10} | {:<10} | {:<10}".format(
                epoch, *result_info
            )
            f.write(txt + "\n")

        if early_stop:
            print("Early stopping triggered.")
            break

        if args.output_dir:
            # 只在主进程上执行保存权重操作
            save_files = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            save_on_master(
                save_files, os.path.join(args.output_dir, "model_last.pth")
            )
            if best_epoch == epoch:
                save_on_master(
                    save_files, os.path.join(args.output_dir, "model_best.pth")
                )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr

        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map

        plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 训练文件的根目录(coco2017)
    parser.add_argument("--data-path", default="/data/coco2017", help="dataset")
    # 训练设备类型
    parser.add_argument("--device", default="cuda", help="device")
    # 检测目标类别数(不包含背景)
    parser.add_argument(
        "--num-classes", default=90, type=int, help="num_classes"
    )
    # 每块GPU上的batch_size
    parser.add_argument(
        "-b",
        "--batch-size",
        default=4,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )
    # 指定接着从哪个epoch数开始训练
    parser.add_argument(
        "--start_epoch", default=0, type=int, help="start epoch"
    )
    # 训练的总epoch数
    parser.add_argument(
        "--epochs",
        default=26,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    # 数据加载以及预处理的线程数
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    # 学习率，这个需要根据gpu的数量以及batch_size进行设置0.02 / bs * num_GPU
    parser.add_argument(
        "--lr",
        default=0.005,
        type=float,
        help="initial learning rate, 0.02 is the default value for training "
        "on 8 gpus and 2 images_per_gpu",
    )
    # SGD的momentum参数
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    # SGD的weight_decay参数
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    # 针对torch.optim.lr_scheduler.StepLR的参数
    parser.add_argument(
        "--lr-step-size",
        default=8,
        type=int,
        help="decrease lr every step-size epochs",
    )
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs",
    )
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )
    # 训练过程打印信息的频率
    parser.add_argument(
        "--print-freq", default=50, type=int, help="print frequency"
    )
    # 文件保存地址
    parser.add_argument(
        "--output-dir", default="./multi_train", help="path where to save"
    )
    # 基于上次的训练结果接着训练
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--test-only", action="store_true", help="test only")

    # 开启的进程数(注意不是线程)
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
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--pretrain",
        type=bool,
        default=True,
        help="load COCO pretrain weights.",
    )
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument(
        "--amp",
        default=False,
        help="Use torch.cuda.amp for mixed precision training",
    )

    # QAT Training额外添加参数
    parser.add_argument(
        "--check-ptq",
        action="store_true",
        help="check model's performance after initialization",
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
        "--seed", type=int, default=0, help="Global training seed"
    )
    parser.add_argument(
        "--num_bits",
        type=int,
        default=8,
        help="Quantization num bits for both activation and weight params",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Wait epochs before the program ends",
    )

    # QAT + feature Distillation额外添加参数
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="path to data ckpt"
    )
    parser.add_argument(
        "--train_kind",
        type=str,
        default=None,
        help="Use pseudo or real data for training",
    )
    parser.add_argument(
        "--kd",
        type=str,
        default="",
        help="Knowledge distillation method. Default: disable",
    )
    parser.add_argument(
        "--kd-loss-weight", type=float, default=1.0, help="weight of kd loss"
    )
    parser.add_argument(
        "--mse-loss-weight", type=float, default=1.0, help="weight of mse loss"
    )
    parser.add_argument(
        "--ori-loss-weight",
        type=float,
        default=1.0,
        help="weight of original loss",
    )
    parser.add_argument(
        "--module",
        type=str,
        default="",
        help='name of the teacher module used in kd. Default (""): use the output of model.',
    )

    args = parser.parse_args()

    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    if args.output_dir:
        mkdir(args.output_dir)

    init_seeds(args.seed)
    main(args)
