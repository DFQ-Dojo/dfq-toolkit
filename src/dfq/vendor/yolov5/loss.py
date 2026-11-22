# Ultralytics YOLOv5, AGPL-3.0 license
"""The YOLOv5 loss, from YOLOv5 v7.0 (`utils/loss.py`, `utils/metrics.py`).

Each target is matched to the anchors whose size ratio is below `anchor_t`, at its own cell and
the two neighbouring cells nearest its centre. The loss sums CIoU of the matched boxes (`box`),
objectness against the IoU, weighted per level (`obj`), and class BCE (`cls`), each times its
gain in `hyp`, and multiplies the sum by the batch size. Focal loss, label smoothing and
automatic balancing, which dfq does not use, are removed.
"""

import math
from typing import Any

import torch
from torch import Tensor, nn


def centre_complete_iou(box1: Tensor, box2: Tensor, eps: float = 1e-7) -> Tensor:
    """CIoU of centre boxes `[..., 4]` (x, y, w, h), as YOLOv5's `bbox_iou(..., CIoU=True)`."""
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1 / 2, x1 + w1 / 2, y1 - h1 / 2, y1 + h1 / 2
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2 / 2, x2 + w2 / 2, y2 - h2 / 2, y2 + h2 / 2
    inter_w = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0)
    inter_h = (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    inter = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    enclosing_w = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    enclosing_h = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    diagonal = enclosing_w**2 + enclosing_h**2 + eps
    centre_distance = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
    ) / 4
    aspect = 4 / math.pi**2 * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = aspect / (aspect - iou + (1 + eps))
    return iou - (centre_distance / diagonal + aspect * alpha)


class ComputeLoss:
    def __init__(self, model: Any, hyp: dict[str, float]) -> None:
        head = model.model[-1]
        self.device = next(model.parameters()).device
        self.hyp = hyp
        self.bce_cls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([hyp["cls_pw"]], device=self.device)
        )
        self.bce_obj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([hyp["obj_pw"]], device=self.device)
        )
        self.balance = {3: [4.0, 1.0, 0.4]}.get(head.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.na, self.nc, self.nl = head.na, head.nc, head.nl
        self.anchors = head.anchors

    def match(
        self, p: list[Tensor], targets: Tensor
    ) -> tuple[list[Tensor], list[Tensor], list[tuple[Tensor, ...]], list[Tensor]]:
        """Per level: classes, boxes relative to cells, (image, anchor, row, column), anchors."""
        na, nt = self.na, targets.shape[0]
        classes: list[Tensor] = []
        boxes: list[Tensor] = []
        indices: list[tuple[Tensor, ...]] = []
        anchor_sizes: list[Tensor] = []
        gain = torch.ones(7, device=self.device)
        anchor_index = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), anchor_index[..., None]), 2)
        g = 0.5
        neighbours = (
            torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device).float() * g
        )
        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]
            t = targets * gain
            offsets: Tensor | int = 0
            if nt:
                ratio = t[..., 4:6] / anchors[:, None]
                fits = torch.max(ratio, 1 / ratio).max(2)[0] < self.hyp["anchor_t"]
                t = t[fits]
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                left, top = ((gxy % 1 < g) & (gxy > 1)).T
                right, bottom = ((gxi % 1 < g) & (gxi > 1)).T
                chosen = torch.stack((torch.ones_like(left), left, top, right, bottom))
                t = t.repeat((5, 1, 1))[chosen]
                offsets = (torch.zeros_like(gxy)[None] + neighbours[:, None])[chosen]
            else:
                t = targets[0]
            image_class, gxy, gwh, anchor = t.chunk(4, 1)
            anchor_ids = anchor.long().view(-1)
            image, cls = image_class.long().T
            cells = (gxy - offsets).long()
            gi, gj = cells.T
            indices.append(
                (image, anchor_ids, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1))
            )
            boxes.append(torch.cat((gxy - cells, gwh), 1))
            anchor_sizes.append(anchors[anchor_ids])
            classes.append(cls)
        return classes, boxes, indices, anchor_sizes

    def __call__(self, p: list[Tensor], targets: Tensor) -> tuple[Tensor, Tensor]:
        """Loss for raw outputs `p` and target rows (image, class, x, y, w, h normalized)."""
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        classes, boxes, indices, anchor_sizes = self.match(p, targets)
        for i, level in enumerate(p):
            image, anchor, gj, gi = indices[i]
            target_obj = torch.zeros(level.shape[:4], dtype=level.dtype, device=self.device)
            count = image.shape[0]
            if count:
                pxy, pwh, _, pcls = level[image, anchor, gj, gi].split((2, 2, 1, self.nc), 1)
                predicted = torch.cat(
                    (pxy.sigmoid() * 2 - 0.5, (pwh.sigmoid() * 2) ** 2 * anchor_sizes[i]), 1
                )
                iou = centre_complete_iou(predicted, boxes[i]).squeeze()
                lbox += (1.0 - iou).mean()
                target_obj[image, anchor, gj, gi] = iou.detach().clamp(0).type(target_obj.dtype)
                if self.nc > 1:
                    target_cls = torch.full_like(pcls, 0.0, device=self.device)
                    target_cls[range(count), classes[i]] = 1.0
                    lcls += self.bce_cls(pcls, target_cls)
            lobj += self.bce_obj(level[..., 4], target_obj) * self.balance[i]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        batch = target_obj.shape[0]
        return (lbox + lobj + lcls) * batch, torch.cat((lbox, lobj, lcls)).detach()
