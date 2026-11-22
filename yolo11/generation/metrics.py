from typing import Tuple
import torch

from pydantic import BaseModel
from pathlib import Path

from ultralytics.utils import LOGGER, LOCAL_RANK  # type: ignore
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy, scale_boxes  # type: ignore
from ultralytics.utils.metrics import box_iou, ap_per_class  # type: ignore
from ultralytics.utils.plotting import plot_images, output_to_target  # type: ignore
from ultralytics.models.yolo.detect.predict import DetectionPredictor  # type: ignore

import numpy as np

from external.data import LabelSet
from external.data import LabelBatch, merge_to_target_tensor, to_yolo_format

from generation.v8_loss import v8DetectionLoss, v8loss_from_predict_model  # type: ignore


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
        x = torch.where(
            (iou >= iouv[i]) & correct_class
        )  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[
                    np.unique(matches[:, 1], return_index=True)[1]
                ]
                matches = matches[
                    np.unique(matches[:, 0], return_index=True)[1]
                ]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def calculate_batch(
    preds: torch.Tensor,
    plot_image: bool,
    inputs: torch.Tensor,
    sample_labels: LabelBatch,
    work_dir: Path,
    names: dict[int, str],
    device: torch.device,
    idx: int,
) -> Tuple[int, list]:
    with torch.no_grad():
        preds = non_max_suppression(
            preds, 0.001, 0.6, multi_label=True, agnostic=False, max_det=300
        )
        if plot_image:
            batch_indices, class_ids, xywh, confs = output_to_target(preds)
            plot_images(
                images=inputs,
                batch_idx=batch_indices,
                cls=class_ids,
                bboxes=xywh,
                confs=confs,
                paths=sample_labels.sample_paths,
                fname=work_dir
                / f"pred_batch{idx}_gpu{LOCAL_RANK}_{torch.rand(1).item()}.jpg",
                names=names,
            )  # pred

        _, _, height, width = inputs.shape

        targets = merge_to_target_tensor(sample_labels).to(device)
        targets[:, 2:] *= torch.tensor(
            (width, height, width, height), device=device
        )  # to
        iouv = torch.linspace(
            0.5, 0.95, 10, device=device
        )  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        stats = []
        seen = 0

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = (
                labels.shape[0],
                pred.shape[0],
            )  # number of labels, predictions
            shape = sample_labels.original_shapes[si]
            correct = torch.zeros(
                npr, niou, dtype=torch.bool, device=device
            )  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append(
                        (
                            correct,
                            *torch.zeros((2, 0), device=device),
                            labels[:, 0],
                        )
                    )
                continue

            # Predictions
            predn = pred.clone()
            scale_boxes(
                inputs[si].shape[1:],
                predn[:, :4],
                shape,
            )  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(
                    inputs[si].shape[1:],
                    tbox,
                    shape,
                )  # native-space labels
                labelsn = torch.cat(
                    (labels[:, 0:1], tbox), 1
                )  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            stats.append(
                (correct, pred[:, 4], pred[:, 5], labels[:, 0])
            )  # (correct, conf, pcls, tcls

    return seen, stats


def calculate_map(
    seen: int,
    stats: list,
    work_dir: Path,
    names: dict[int, str],
    nc: int,
) -> Tuple[float, float]:
    mp = 0
    mr = 0
    map50 = 0
    map = 0
    stats_np = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats_np) and stats_np[0].any():
        _, _, p, r, _, ap, _, _, _, _, _, _ = ap_per_class(
            *stats_np, plot=False, save_dir=work_dir, names=names
        )
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(
        stats_np[3].astype(int), minlength=nc
    )  # number of targets per class
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(
            "WARNING ⚠️ no labels found, can not compute metrics without labels"
        )
    return map, map50


class MetricCalculator(BaseModel, arbitrary_types_allowed=True):
    nc: int
    names: dict[int, str]
    device: torch.device
    predictor: DetectionPredictor
    loss_function: v8DetectionLoss

    def metric(
        self,
        inputs: torch.Tensor,
        sample_labels: LabelBatch,
        plot_image: bool,
        work_dir: Path,
        idx: int,
    ):
        with torch.no_grad():
            preds, preds_item = self.predictor.inference(inputs)

            loss, loss_item = self.loss_function(
                preds_item, to_yolo_format(sample_labels)
            )
            work_dir.mkdir(exist_ok=True, parents=True)
            seen, stats = calculate_batch(
                preds,
                plot_image,
                inputs,
                sample_labels,
                work_dir,
                self.names,
                self.device,
                idx,
            )

            map50, map = calculate_map(
                seen, stats, work_dir, self.names, self.nc
            )
        return loss, loss_item, map50, map


def metric_from_label_set(
    label_set: LabelSet, predictor: DetectionPredictor, device: torch.device
) -> MetricCalculator:
    return MetricCalculator(
        nc=label_set.nc,
        names=label_set.names,
        predictor=predictor,
        device=device,
        loss_function=v8loss_from_predict_model(predictor),
    )
