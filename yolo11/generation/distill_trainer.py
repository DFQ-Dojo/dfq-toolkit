# This file implements DistillTrainer.

import os
import collections
from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel
import torch
from torch.utils.tensorboard import SummaryWriter

from ultralytics.nn.tasks import DetectionModel  # type: ignore
from ultralytics.models.yolo.detect.predict import DetectionPredictor  # type: ignore
from ultralytics.utils import LOGGER, LOCAL_RANK, RANK  # type: ignore

from generation.config import (
    BoxSamplerConfig,
    GenerationConfig,
    InputLabelKind,
    LabelConfig,
)
from external.data import load_gmanifest
from external.data import (
    LabelBatch,
    LabelSet,
    get_labelset_from_yolo,
    save_gmanifest,
    to_yolo_format,
    get_sampled_labelset,
)

from generation.fp_sampler import FpSampler
from generation.metrics import MetricCalculator, metric_from_label_set
from generation.v8_loss import v8DetectionLoss, v8loss_from_predict_model  # type: ignore

import torchvision.utils as vutils  # type: ignore

WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


def save_img(batch_tens, loc):
    """
    Saves a batch_tens of images to location loc
    """
    print(
        "Saving batch_tensor of shape {} to location: {}".format(
            batch_tens.shape, loc
        )
    )
    vutils.save_image(batch_tens, loc, normalize=True, scale_each=True)


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


class BatchHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, _):
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


def use_bn_hook(model: torch.nn.Module):

    hook_handlers = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            hook_handlers.append(BatchHook(layer))
    LOGGER.info(f"added bn hook on {len(hook_handlers)} layers.")

    return hook_handlers


class CSGLossResult(BaseModel, arbitrary_types_allowed=True):
    # the five tensor below are excluded from the computation graph.
    loss_main: torch.Tensor
    loss_var_l1: torch.Tensor
    loss_var_l2: torch.Tensor
    loss_r_feature: torch.Tensor
    loss_first_bn: torch.Tensor

    weighted_main_loss: torch.Tensor
    weighted_tv_l1: torch.Tensor
    weighted_tv_l2: torch.Tensor
    weighted_r_feature: torch.Tensor
    weighted_loss_first_bn: torch.Tensor

    total_loss: torch.Tensor

    def report(self, writer: SummaryWriter, iteration: int, lr: float):
        # Weighted Loss
        writer.add_scalar(
            "weighted/total_loss", self.total_loss.item(), iteration
        )
        writer.add_scalar(
            "weighted/task_loss", self.weighted_main_loss.item(), iteration
        )
        writer.add_scalar(
            "weighted/prior_loss_var_l1", self.weighted_tv_l1.item(), iteration
        )
        writer.add_scalar(
            "weighted/prior_loss_var_l2", self.weighted_tv_l2.item(), iteration
        )
        writer.add_scalar(
            "weighted/loss_r_feature", self.weighted_r_feature.item(), iteration
        )
        writer.add_scalar(
            "weighted/loss_r_feature_first",
            self.weighted_loss_first_bn.item(),
            iteration,
        )

        # Unweighted loss
        writer.add_scalar(
            "unweighted/task_loss", self.loss_main.item(), iteration
        )
        writer.add_scalar(
            "unweighted/prior_loss_var_l1",
            self.loss_var_l1.item(),
            iteration,
        )
        writer.add_scalar(
            "unweighted/prior_loss_var_l2",
            self.loss_var_l2.item(),
            iteration,
        )
        writer.add_scalar(
            "unweighted/loss_r_feature",
            self.loss_r_feature.item(),
            iteration,
        )
        writer.add_scalar(
            "unweighted/loss_r_feature_first",
            self.loss_first_bn.item(),
            iteration,
        )
        writer.add_scalar(
            "learning_rate",
            lr,
            iteration,
        )

    def dump(self) -> dict:
        return {k: v.item() for k, v in self.model_dump().items()}


class CSGLoss(BaseModel, arbitrary_types_allowed=True):
    """Calibration Set Generation loss"""

    main_loss_multiplier: float
    tv_l1_multiplier: float
    tv_l2_multiplier: float
    r_feature_multiplier: float
    first_bn_multiplier: float

    def calculate_loss(
        self,
        image: torch.Tensor,
        loss_main: torch.Tensor,
        hook_handlers: List[BatchHook],
    ) -> CSGLossResult:

        loss_main, weighted_loss_main = (
            loss_main.clone().detach(),
            self.main_loss_multiplier * loss_main,
        )

        loss_var_l1, loss_var_l2 = get_image_prior_losses(image)

        loss_var_l1, weighted_tv_l1 = (
            loss_var_l1.clone().detach(),
            self.tv_l1_multiplier * loss_var_l1,
        )
        loss_var_l2, weighted_tv_l2 = (
            loss_var_l2.clone().detach(),
            self.tv_l2_multiplier * loss_var_l2,
        )

        # batch norm feature loss
        numLayers = len(hook_handlers)
        loss_r_feature = torch.sum(
            torch.stack([mod.r_feature for mod in hook_handlers[0:numLayers]])
        )
        loss_r_feature, weighted_loss_r_feature = (
            loss_r_feature.clone().detach(),
            self.r_feature_multiplier * loss_r_feature,
        )

        # R_feature loss layer_1
        loss_r_feature_first = sum([mod.r_feature for mod in hook_handlers[:1]])
        loss_r_feature_first, weighted_loss_r_feature_first = (
            loss_r_feature_first.clone().detach(),
            self.first_bn_multiplier * loss_r_feature_first,
        )

        loss = (
            weighted_loss_main
            + weighted_tv_l1
            + weighted_tv_l2
            + weighted_loss_r_feature
            + weighted_loss_r_feature_first
        )

        return CSGLossResult(
            loss_main=loss_main,
            loss_var_l1=loss_var_l1,
            loss_var_l2=loss_var_l2,
            loss_r_feature=loss_r_feature,
            loss_first_bn=loss_r_feature_first,
            weighted_main_loss=weighted_loss_main,
            weighted_tv_l1=weighted_tv_l1,
            weighted_tv_l2=weighted_tv_l2,
            weighted_r_feature=weighted_loss_r_feature,
            weighted_loss_first_bn=weighted_loss_r_feature_first,
            total_loss=loss,
        )


class DistillResult(BaseModel, arbitrary_types_allowed=True):
    loss_record: list
    img: torch.Tensor
    labels: LabelBatch


def save_distill_results(
    distill_result: DistillResult,
    exp_path: Path,
    gen_batch_idx: int,
    skip_images: bool,
    device: torch.device,
):
    loss_path = exp_path / "loss"
    loss_path.mkdir(parents=True, exist_ok=True)
    loss_path = loss_path / f"loss_batch{gen_batch_idx}_gpu{device}.json"
    with loss_path.open("w") as f:
        import json

        json.dump(distill_result.loss_record, f, indent=2)

    weight_path = exp_path / "weights"
    weight_path.mkdir(parents=True, exist_ok=True)

    labels_path = weight_path / "labels"
    labels_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        distill_result.labels,
        labels_path / f"best_batch{gen_batch_idx}_gpu{device}.pt",
    )

    if skip_images:
        return

    images_path = weight_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        distill_result.img,
        # TODO: maybe better do not couple device generated and later processing...
        images_path / f"best_batch{gen_batch_idx}_gpu{device}.pt",
    )


class DistillTrainer(BaseModel, arbitrary_types_allowed=True):
    teacher_model: DetectionPredictor
    dataset: LabelSet
    v8_loss: v8DetectionLoss
    csg_loss: CSGLoss
    fp_sampler: FpSampler
    metrics: MetricCalculator
    writer: SummaryWriter

    lr: float
    device: torch.device
    hook_handlers: List[BatchHook]
    iterations_per_layer: int

    img_length: int
    save_every: int
    save_images: bool
    skip_generated: bool
    work_dir: Path

    # side effect:
    # for every `save_every` interval,
    # save pictures to workdir / plt / generated_images / batch x
    # save loss record at workdir / loss
    # save weights at workdir / weights
    def train_one_batch(
        self, gen_batch_idx: int, sample_batch: LabelBatch
    ) -> DistillResult:
        from tqdm import tqdm  # type: ignore

        best_cost = 1e4

        loss_record = []

        batch_img_size = torch.Size(
            [sample_batch.size(), 3, self.img_length, self.img_length]
        )
        inputs = (
            torch.randint(high=255, size=batch_img_size)
            .to(self.device, non_blocking=True)
            .float()
            / 255
        )
        inputs.requires_grad_()

        sample_batch.batch_indices = sample_batch.batch_indices.to(self.device)
        sample_batch.bboxes = sample_batch.bboxes.to(self.device)
        sample_batch.classes = sample_batch.classes.to(self.device)

        optimizer = torch.optim.Adam(
            [inputs], lr=self.lr, betas=(0.9, 0.999), eps=1e-8
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.iterations_per_layer, eta_min=0.0
        )

        for iteration in tqdm(range(1, self.iterations_per_layer + 1)):
            _, preds_item = self.teacher_model.inference(inputs)

            main_loss, _ = self.v8_loss(
                preds_item, to_yolo_format(sample_batch)
            )

            loss_result = self.csg_loss.calculate_loss(
                inputs, main_loss, self.hook_handlers
            )

            """
            first time loss:
            s
            loss_main=tensor(417.3170, device='cuda:0')
            loss_var_l1=tensor(1.3334, device='cuda:0')
            loss_var_l2=tensor(10228.8418, device='cuda:0')
            loss_r_feature=tensor(375.9451, device='cuda:0')
            loss_first_bn=tensor(0.3008, device='cuda:0')
            ---
            m
            loss_main=tensor(211.7672, device='cuda:0')
            loss_var_l1=tensor(1.3335, device='cuda:0')
            loss_var_l2=tensor(7233.3804, device='cuda:0')
            loss_r_feature=tensor(524.2798, device='cuda:0')
            loss_first_bn=tensor(0.3018, device='cuda:0')
            ---
            l
            loss_main=tensor(114.7332, device='cuda:0')
            loss_var_l1=tensor(1.3333, device='cuda:0')
            loss_var_l2=tensor(5113.8062, device='cuda:0')
            loss_r_feature=tensor(558.5369, device='cuda:0')
            loss_first_bn=tensor(0.3058, device='cuda:0')
            """

            loss_result.report(
                self.writer, iteration, float(optimizer.param_groups[0]["lr"])
            )

            loss = loss_result.total_loss

            prefix = self.work_dir / "plt"
            if iteration % self.save_every == 0:
                save_dir = f"{prefix}/generated_images"
                (
                    verifier_loss,
                    _,
                    verifier_map50,
                    verifier_map,
                ) = self.metrics.metric(
                    inputs.detach(),
                    sample_batch,
                    False,
                    Path(save_dir),
                    0,
                )

                sample_batch = self.fp_sampler.sample(
                    iteration, inputs, sample_batch
                )

                # early stopping
                if best_cost > verifier_loss.item() or iteration < 100:
                    best_inputs = inputs.data.clone()
                    best_cost = verifier_loss.item()
                    best_step = iteration

                loss_record.append(
                    {
                        "iteration": iteration,
                        "csg_loss": loss_result.dump(),
                        "verifier": {
                            "loss": verifier_loss.item(),
                            "map50": verifier_map50,
                            "map": verifier_map,
                        },
                    }
                )

                if RANK in {-1, 0}:
                    print(
                        "------------iteration {}----------".format(iteration)
                    )
                    print("************Train****************")
                    print(loss_result)
                    print("************Val****************")
                    print("val loss", verifier_loss.item())
                    print("map: ", verifier_map)
                    print("map50: ", verifier_map50)
                    print("************Bbox param****************")

            # Forward
            optimizer.zero_grad()
            self.teacher_model.model.zero_grad()

            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            # save image
            if self.save_images and iteration % self.save_every == 0:
                save_dir = f"{prefix}/generated_images/batch{int(iteration / self.save_every)}"
                os.makedirs(save_dir, exist_ok=True)
                metric_loss, loss_item, map50, map = self.metrics.metric(
                    inputs.detach(),
                    sample_batch,
                    False,
                    Path(save_dir),
                    0,
                )

                for i in range(inputs.shape[0]):
                    img_idx = (
                        sample_batch.size() // WORLD_SIZE * gen_batch_idx + i
                    )
                    save_img(
                        inputs[i].detach(),
                        f"{prefix}/generated_images/batch{int(iteration / self.save_every)}/generated_idx{img_idx}_gpu{LOCAL_RANK}.png",
                    )
                with open(
                    f"{prefix}/generated_images/batch{int(iteration / self.save_every)}/label.json",
                    "w",
                ) as fout:
                    fout.write(sample_batch.__str__())

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)

        LOGGER.info(f"best iteration is {best_step}")
        metric_loss, loss_item, map50, map = self.metrics.metric(
            best_inputs.detach(),
            sample_batch,
            True,
            prefix,
            gen_batch_idx,
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

        return DistillResult(
            loss_record=loss_record, img=best_inputs, labels=sample_batch
        )

    def save_data_manifest(self):
        manifest_dir = self.work_dir / "weights"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        save_gmanifest(self.dataset, manifest_dir)

    def train_all(self):
        for idx, sample_batch in enumerate(self.dataset.generator):
            # TODO(optim): this is easily parallelized by batch.
            distill_results = self.train_one_batch(
                gen_batch_idx=idx,
                sample_batch=sample_batch,
            )
            save_distill_results(
                distill_results,
                self.work_dir,
                idx,
                self.skip_generated,
                self.device,
            )
        self.save_data_manifest()


def build_yolo_predictor(
    model_pt: Path = Path("yolo11n.pt"),
    device: torch.device = torch.device("cpu"),
) -> DetectionPredictor:
    args = dict(model=model_pt, device=device)
    predictor = DetectionPredictor(overrides=args)
    predictor.setup_model(model_pt)
    return predictor


# build yolo teacher needs to de-Fuse batch normal layer
# so it cannot use the default setup_model. must hand-write it.
def build_yolo_teacher(
    model_pt: Path = Path("yolo11n.pt"),
    device: torch.device = torch.device("cuda"),
) -> Tuple[DetectionModel, List[BatchHook]]:
    from ultralytics.engine.predictor import AutoBackend  # type: ignore

    args = dict(model=model_pt)
    predictor = DetectionPredictor(overrides=args)
    predictor.model = AutoBackend(
        weights=model_pt,
        fuse=False,  # for bn to work
        device=device,
    )
    predictor.device = predictor.model.device
    predictor.args.half = predictor.model.fp16

    hook_handlers = use_bn_hook(predictor.model)
    predictor.model.eval()

    return predictor, hook_handlers


def build_distill_labelset(
    label_config: LabelConfig, calibration_size: int, device: torch.device
) -> LabelSet:
    if label_config.kind == InputLabelKind.REAL:
        return get_labelset_from_yolo(
            label_config.data,
            label_config.batch_size,
            label_config.workers,
            calibration_size,
        )
    elif label_config.kind == InputLabelKind.SAMPLED:
        manifest = load_gmanifest(label_config.sampling_weight_path)
        return get_sampled_labelset(
            sampling_label_path=label_config.sampling_weight_path / "labels",
            batch_size=label_config.batch_size,
            calibration_size=calibration_size,
            nc=manifest.nc,
            names=manifest.names,
            device=device,
        )


def build_fp_sampler(
    config: BoxSamplerConfig, relabel_model: DetectionPredictor
) -> FpSampler:
    return FpSampler(
        enable_box_sampler=config.box_sampler,
        box_sampler_warmup=config.box_sampler_warmup,
        box_sampler_early_exit=config.box_sampler_earlyexit,
        box_sampler_overlap_iou=config.box_sampler_overlap_iou,
        box_sampler_min_area=config.box_sampler_minarea,
        box_sampler_max_area=config.box_sampler_maxarea,
        box_sampler_conf=config.box_sampler_conf,
        relabel_model=relabel_model,
    )


def build_distill_trainer(
    save_dir: Path, config: GenerationConfig
) -> DistillTrainer:

    device = torch.device(config.device)

    print(f"device is {device}")

    teacher_model, hook_handlers = build_yolo_teacher(
        config.teacher_weights, device
    )

    relabel_model = build_yolo_predictor(config.relabel_weights, device)

    writer = SummaryWriter(save_dir / "loss")

    dataset = build_distill_labelset(
        label_config=config.dataset_configs,
        calibration_size=config.calibration_size,
        device=device,
    )

    metrics = metric_from_label_set(
        label_set=dataset, predictor=relabel_model, device=device
    )

    fp_sampler = build_fp_sampler(config.box_sampler_config, relabel_model)

    v8_loss = v8loss_from_predict_model(teacher_model)
    csg_loss = CSGLoss(
        main_loss_multiplier=config.hyp.main_loss_multiplier,
        tv_l1_multiplier=config.hyp.tv_l1,
        tv_l2_multiplier=config.hyp.tv_l2,
        r_feature_multiplier=config.hyp.r_feature,
        first_bn_multiplier=config.hyp.first_bn_coef,
    )

    return DistillTrainer(
        teacher_model=teacher_model,
        dataset=dataset,
        v8_loss=v8_loss,
        csg_loss=csg_loss,
        fp_sampler=fp_sampler,
        metrics=metrics,
        writer=writer,
        lr=config.hyp.lr,
        device=device,
        hook_handlers=hook_handlers,
        iterations_per_layer=config.iterations,
        img_length=config.img_size,
        save_every=config.save_every,
        save_images=config.save_images,
        skip_generated=config.skip_generated,
        work_dir=save_dir,
    )
