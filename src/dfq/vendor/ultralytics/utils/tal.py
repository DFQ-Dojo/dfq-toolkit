# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Anchors and task-aligned assignment, from ultralytics 8.3.53 (`ultralytics/utils/tal.py`).

The assigner (TOOD, https://arxiv.org/abs/2108.07755) scores each anchor against each ground-truth
box by `score ** alpha * iou ** beta`, keeps the `topk` anchors inside each box, gives an anchor
claimed by several boxes to the one it overlaps most, and returns per-anchor targets.
"""

import torch
from torch import Tensor, nn

from .metrics import complete_iou


def make_anchors(
    feats: list[Tensor], strides: Tensor, offset: float = 0.5
) -> tuple[Tensor, Tensor]:
    """The centre of every grid cell of every level, in cells, and each cell's stride."""
    points, stride_values = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for feature, stride in zip(feats, strides, strict=True):
        height, width = feature.shape[2:]
        xs = torch.arange(end=width, device=device, dtype=dtype) + offset
        ys = torch.arange(end=height, device=device, dtype=dtype) + offset
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        points.append(torch.stack((grid_x, grid_y), -1).view(-1, 2))
        stride_values.append(
            torch.full((height * width, 1), float(stride), dtype=dtype, device=device)
        )
    return torch.cat(points), torch.cat(stride_values)


def dist2bbox(distance: Tensor, anchor_points: Tensor, xywh: bool = True, dim: int = -1) -> Tensor:
    """Boxes from (left, top, right, bottom) distances to anchor points."""
    left_top, right_bottom = distance.chunk(2, dim)
    x1y1 = anchor_points - left_top
    x2y2 = anchor_points + right_bottom
    if xywh:
        return torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points: Tensor, bbox: Tensor, reg_max: int) -> Tensor:
    """(left, top, right, bottom) distances from anchor points to corner boxes, clipped to bins."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)


def candidates_in_boxes(centres: Tensor, boxes: Tensor, eps: float = 1e-9) -> Tensor:
    """`[B, boxes, anchors]`: whether each anchor centre lies strictly inside each box."""
    batch, count, _ = boxes.shape
    left_top, right_bottom = boxes.view(-1, 1, 4).chunk(2, 2)
    deltas = torch.cat((centres[None] - left_top, right_bottom - centres[None]), dim=2)
    return deltas.view(batch, count, centres.shape[0], -1).amin(3).gt_(eps)


def highest_overlaps(
    positive: Tensor, overlaps: Tensor, count: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Give an anchor that several boxes claim to the box it overlaps most."""
    foreground = positive.sum(-2)
    if foreground.max() > 1:
        claimed = (foreground.unsqueeze(1) > 1).expand(-1, count, -1)
        best = overlaps.argmax(1)
        is_best = torch.zeros(positive.shape, dtype=positive.dtype, device=positive.device)
        is_best.scatter_(1, best.unsqueeze(1), 1)
        positive = torch.where(claimed, is_best, positive).float()
        foreground = positive.sum(-2)
    return positive.argmax(-2), foreground, positive


class TaskAlignedAssigner(nn.Module):
    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def box_metrics(
        self, scores: Tensor, boxes: Tensor, labels: Tensor, gt_boxes: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """The alignment metric and the CIoU of each (box, anchor) pair inside `mask`."""
        batch, count = gt_boxes.shape[:2]
        anchors = boxes.shape[-2]
        mask = mask.bool()
        overlaps = torch.zeros([batch, count, anchors], dtype=boxes.dtype, device=boxes.device)
        box_scores = torch.zeros([batch, count, anchors], dtype=scores.dtype, device=scores.device)
        images = torch.arange(end=batch).view(-1, 1).expand(-1, count)
        classes = labels.squeeze(-1).long().cpu()  # CPU indices, as upstream builds them
        box_scores[mask] = scores[images, :, classes][mask]
        predicted = boxes.unsqueeze(1).expand(-1, count, -1, -1)[mask]
        truth = gt_boxes.unsqueeze(2).expand(-1, -1, anchors, -1)[mask]
        overlaps[mask] = complete_iou(truth, predicted).squeeze(-1).clamp_(0)
        metric = box_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return metric, overlaps

    def top_candidates(self, metrics: Tensor, valid: Tensor) -> Tensor:
        """`[B, boxes, anchors]`: the `topk` anchors of each valid box by metric."""
        _, indices = torch.topk(metrics, self.topk, dim=-1, largest=True)
        indices.masked_fill_(~valid, 0)
        counts = torch.zeros(metrics.shape, dtype=torch.int8, device=indices.device)
        ones = torch.ones_like(indices[:, :, :1], dtype=torch.int8, device=indices.device)
        for k in range(self.topk):
            counts.scatter_add_(-1, indices[:, :, k : k + 1], ones)
        counts.masked_fill_(counts > 1, 0)
        return counts.to(metrics.dtype)

    def targets(
        self, labels: Tensor, gt_boxes: Tensor, matched: Tensor, foreground: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, count = gt_boxes.shape[:2]
        offsets = torch.arange(end=batch, dtype=torch.int64, device=labels.device)[..., None]
        flat_index = matched + offsets * count
        target_labels = labels.long().flatten()[flat_index]
        target_boxes = gt_boxes.view(-1, gt_boxes.shape[-1])[flat_index]
        target_labels.clamp_(0)
        shape = (target_labels.shape[0], target_labels.shape[1], self.num_classes)
        one_hot = torch.zeros(shape, dtype=torch.int64, device=target_labels.device)
        one_hot.scatter_(2, target_labels.unsqueeze(-1), 1)
        is_foreground = foreground[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(is_foreground > 0, one_hot, 0)
        return target_labels, target_boxes, target_scores

    @torch.no_grad()
    def forward(
        self,
        pd_scores: Tensor,
        pd_bboxes: Tensor,
        anc_points: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return target labels, boxes and scores per anchor, the foreground mask, matched boxes."""
        count = gt_bboxes.shape[1]
        if count == 0:
            background = torch.full_like(pd_scores[..., 0], self.num_classes)
            zeros = torch.zeros_like(pd_scores[..., 0])
            return (
                background,
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                zeros,
                zeros,
            )
        inside = candidates_in_boxes(anc_points, gt_bboxes)
        metric, overlaps = self.box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, inside * mask_gt
        )
        top = self.top_candidates(metric, mask_gt.expand(-1, -1, self.topk).bool())
        positive = top * inside * mask_gt
        matched, foreground, positive = highest_overlaps(positive, overlaps, count)
        labels, boxes, scores = self.targets(gt_labels, gt_bboxes, matched, foreground)
        metric = metric * positive
        best_metric = metric.amax(dim=-1, keepdim=True)
        best_overlap = (overlaps * positive).amax(dim=-1, keepdim=True)
        normalized = (metric * best_overlap / (best_metric + self.eps)).amax(-2).unsqueeze(-1)
        return labels, boxes, scores * normalized, foreground.bool(), matched
