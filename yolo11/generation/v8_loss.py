import torch
import torch.nn as nn

from pydantic import BaseModel
from ultralytics.utils.ops import xywh2xyxy  # type: ignore
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors  # type: ignore
from ultralytics.utils.loss import BboxLoss  # type: ignore
from ultralytics.models.yolo.detect.predict import DetectionPredictor  # type: ignore


class v8DetectionLoss(BaseModel, arbitrary_types_allowed=True):
    """This class is equivalent with Ultralytics' v8DetectionLoss except for its constructor."""

    bce: nn.BCEWithLogitsLoss
    device: torch.device
    stride: torch.Tensor
    nc: int
    no: int
    detect_reg_max: int
    use_dfl: bool
    assigner: TaskAlignedAssigner
    bbox_loss: BboxLoss
    proj: torch.Tensor
    hyp_box_gain: float
    hyp_cls_gain: float
    hyp_dfl_gain: float

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(
                batch_size, counts.max(), ne - 1, device=self.device
            )
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.detect_reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (
                batch["batch_idx"].view(-1, 1),
                batch["cls"].view(-1, 1),
                batch["bboxes"],
            ),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device),
            batch_size,
            scale_tensor=imgsz[[1, 0, 1, 0]],
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(
            anchor_points, pred_distri
        )  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum()
            / target_scores_sum
        )  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp_box_gain  # box gain
        loss[1] *= self.hyp_cls_gain  # cls gain
        loss[2] *= self.hyp_dfl_gain  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


def new_v8_detection_loss(
    device: torch.device,
    stride: torch.Tensor,
    nc: int,
    detect_reg_max: int,
    hyp_box_gain: float,
    hyp_cls_gain: float,
    hyp_dfl_gain: float,
    tal_topk=10,
) -> v8DetectionLoss:
    """Construct a v8 loss based on hyperparameters

    Parameters
    ----------
    device : torch.device
    stride : torch.Tensor
    nc : int
    detect_reg_max : int
    hyp_box_gain : float
    hyp_cls_gain : float
    hyp_dfl_gain : float
    tal_topk : int, optional

    Returns
    -------
    v8DetectionLoss
    """
    bce = nn.BCEWithLogitsLoss(reduction="none")
    no = nc + detect_reg_max * 4
    use_dfl = detect_reg_max > 1

    assigner = TaskAlignedAssigner(
        topk=tal_topk, num_classes=nc, alpha=0.5, beta=6.0
    )
    bbox_loss = BboxLoss(detect_reg_max).to(device)
    proj = torch.arange(detect_reg_max, dtype=torch.float, device=device)

    return v8DetectionLoss(
        bce=bce,
        device=device,
        stride=stride,
        nc=nc,
        no=no,
        detect_reg_max=detect_reg_max,
        use_dfl=use_dfl,
        assigner=assigner,
        bbox_loss=bbox_loss,
        proj=proj,
        hyp_box_gain=hyp_box_gain,
        hyp_cls_gain=hyp_cls_gain,
        hyp_dfl_gain=hyp_dfl_gain,
    )


def v8loss_from_predict_model(predictor: DetectionPredictor) -> v8DetectionLoss:
    """Construct a v8 loss based on an Ultralytics YOLO model.

    Parameters
    ----------
    predictor : DetectionPredictor

    Returns
    -------
    v8DetectionLoss
    """
    device = next(predictor.model.model.model.parameters()).device
    stride = predictor.model.model.stride
    nc = 80  # TODO: remove this hack
    detect_reg_max = predictor.model.model.model[-1].reg_max
    hyp_box_gain = predictor.args.box
    hyp_cls_gain = predictor.args.cls
    hyp_dfl_gain = predictor.args.dfl
    return new_v8_detection_loss(
        device=device,
        stride=stride,
        nc=nc,
        detect_reg_max=detect_reg_max,
        hyp_box_gain=hyp_box_gain,
        hyp_cls_gain=hyp_cls_gain,
        hyp_dfl_gain=hyp_dfl_gain,
    )
