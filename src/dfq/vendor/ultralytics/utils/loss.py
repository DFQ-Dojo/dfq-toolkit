# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""The YOLO detection loss, from ultralytics 8.3.53 (`ultralytics/utils/loss.py`).

The loss has three terms, each multiplied by its gain in `model.args`: CIoU of the decoded boxes
(`box`), binary cross-entropy of the class scores (`cls`), and distribution focal loss of the box
sides (`dfl`). Their sum is multiplied by the batch size, as Ultralytics does.
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .metrics import complete_iou
from .ops import xywh2xyxy
from .tal import TaskAlignedAssigner, bbox2dist, dist2bbox, make_anchors


def distribution_focal_loss(pred_dist: Tensor, target: Tensor, reg_max: int) -> Tensor:
    """DFL (https://arxiv.org/abs/2006.04388): cross-entropy to the two bins around each target."""
    clipped = target.clamp_(0, reg_max - 1 - 0.01)
    left = clipped.long()
    right = left + 1
    left_weight = right - clipped
    right_weight = 1 - left_weight
    left_loss = F.cross_entropy(pred_dist, left.view(-1), reduction="none").view(left.shape)
    right_loss = F.cross_entropy(pred_dist, right.view(-1), reduction="none").view(left.shape)
    return (left_loss * left_weight + right_loss * right_weight).mean(-1, keepdim=True)


def box_losses(
    pred_dist: Tensor,
    pred_bboxes: Tensor,
    anchor_points: Tensor,
    target_bboxes: Tensor,
    target_scores: Tensor,
    target_scores_sum: Tensor,
    fg_mask: Tensor,
    reg_max: int,
) -> tuple[Tensor, Tensor]:
    """The CIoU loss and the DFL of the foreground anchors, weighted by their target scores."""
    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    iou = complete_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
    iou_loss = ((1.0 - iou) * weight).sum() / target_scores_sum
    target_ltrb = bbox2dist(anchor_points, target_bboxes, reg_max - 1)
    dfl = distribution_focal_loss(
        pred_dist[fg_mask].view(-1, reg_max), target_ltrb[fg_mask], reg_max
    )
    dfl_loss = (dfl * weight).sum() / target_scores_sum
    return iou_loss, dfl_loss


class v8DetectionLoss:  # noqa: N801 - Ultralytics' name
    def __init__(self, model: Any, tal_topk: int = 10) -> None:
        head = model.model[-1]
        self.device = next(model.parameters()).device
        self.hyp = model.args
        self.stride = head.stride
        self.nc = head.nc
        self.reg_max = head.reg_max
        self.no = head.nc + head.reg_max * 4
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.proj = torch.arange(head.reg_max, dtype=torch.float, device=self.device)

    def padded_targets(self, targets: Tensor, batch_size: int, scale: Tensor) -> Tensor:
        """`[B, max boxes, 5]` (class, corner box) from (image, class, centre box) rows."""
        rows, columns = targets.shape
        if rows == 0:
            return torch.zeros(batch_size, 0, columns - 1, device=self.device)
        images = targets[:, 0]
        _, counts = images.unique(return_counts=True)
        padded = torch.zeros(batch_size, int(counts.max()), columns - 1, device=self.device)
        for image in range(batch_size):
            matches = images == image
            count = int(matches.sum())
            if count:
                padded[image, :count] = targets[matches, 1:]
        padded[..., 1:5] = xywh2xyxy(padded[..., 1:5].mul_(scale))
        return padded

    def decode(self, anchor_points: Tensor, pred_dist: Tensor) -> Tensor:
        batch, anchors, channels = pred_dist.shape
        bins = pred_dist.view(batch, anchors, 4, channels // 4).softmax(3)
        distances = bins.matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(distances, anchor_points, xywh=False)

    def __call__(self, preds: Any, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """The gain-weighted loss sum times the batch size, and the detached (box, cls, dfl)."""
        feats = preds[1] if isinstance(preds, tuple) else preds
        joined = torch.cat([level.view(feats[0].shape[0], self.no, -1) for level in feats], 2)
        pred_distri, pred_scores = joined.split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        image_size = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        )
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        rows = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.padded_targets(rows.to(self.device), batch_size, image_size[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        loss = torch.zeros(3, device=self.device)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = box_losses(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                self.reg_max,
            )
        gains = torch.tensor([self.hyp.box, self.hyp.cls, self.hyp.dfl], device=self.device)
        weighted = loss * gains
        return weighted.sum() * batch_size, weighted.detach()
