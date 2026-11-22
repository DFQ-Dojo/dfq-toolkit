from typing import List, Dict

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

from . import det_utils
from . import boxes as box_ops


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Arguments:

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    # sampled_pos_inds_subset = torch.nonzero(torch.gt(labels, 0)).squeeze(1)
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]

    labels_pos = labels[sampled_pos_inds_subset]

    # shape=[num_proposal, num_classes]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = (
        det_utils.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            size_average=False,
        )
        / labels.numel()
    )

    return classification_loss, box_loss


def maskrcnn_inference(x, labels):
    # type: (Tensor, List[Tensor]) -> List[Tensor]
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    Args:
        x (Tensor): the mask logits
        labels (list[BoxList]): bounding boxes that are used as
            reference, one for ech image

    Returns:
        results (list[BoxList]): one BoxList for each image, containing
            the extra field mask
    """
    mask_prob = x.sigmoid()

    # select masks corresponding to the predicted classes
    num_masks = x.shape[0]
    boxes_per_image = [label.shape[0] for label in labels]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_prob = mask_prob[index, labels][:, None]
    mask_prob = mask_prob.split(boxes_per_image, dim=0)

    return mask_prob


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (M, M), 1.0)[:, 0]


def maskrcnn_loss(
    mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs
):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """

    Args:
        mask_logits:
        proposals:
        gt_masks:
        gt_labels:
        mask_matched_idxs:

    Returns:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    labels = [
        gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)
    ]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    # 计算预测mask与真实gt_mask之间的BCELoss
    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[
            torch.arange(labels.shape[0], device=labels.device), labels
        ],
        mask_targets,
    )
    return mask_loss


class RoIHeads(torch.nn.Module):
    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        box_roi_pool,  # Multi-scale RoIAlign pooling
        box_head,  # TwoMLPHead
        box_predictor,  # FastRCNNPredictor
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,  # default: 0.5, 0.5
        batch_size_per_image,
        positive_fraction,  # default: 512, 0.25
        bbox_reg_weights,  # None
        # Faster R-CNN inference
        score_thresh,  # default: 0.05
        nms_thresh,  # default: 0.5
        detection_per_img,  # default: 100
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
    ):
        super(RoIHeads, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # default: 0.5
            bg_iou_thresh,  # default: 0.5
            allow_low_quality_matches=False,
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction,  # default: 512
        )  # default: 0.25

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool  # Multi-scale RoIAlign pooling
        self.box_head = box_head  # TwoMLPHead
        self.box_predictor = box_predictor  # FastRCNNPredictor

        self.score_thresh = score_thresh  # default: 0.05
        self.nms_thresh = nms_thresh  # default: 0.5
        self.detection_per_img = detection_per_img  # default: 100

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        Args:
            proposals:
            gt_boxes:
            gt_labels:

        Returns:

        """
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(
            proposals, gt_boxes, gt_labels
        ):
            if gt_boxes_in_image.numel() == 0:
                # background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],),
                    dtype=torch.int64,
                    device=device,
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],),
                    dtype=torch.int64,
                    device=device,
                )
            else:
                # set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(
                    gt_boxes_in_image, proposals_in_image
                )

                matched_idxs_in_image = self.proposal_matcher(
                    match_quality_matrix
                )

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(
                    min=0
                )
                labels_in_image = gt_labels_in_image[
                    clamped_matched_idxs_in_image
                ]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # label background (below the low threshold)
                bg_inds = (
                    matched_idxs_in_image
                    == self.proposal_matcher.BELOW_LOW_THRESHOLD
                )  # -1
                labels_in_image[bg_inds] = 0

                # label ignore proposals (between low and high threshold)
                ignore_inds = (
                    matched_idxs_in_image
                    == self.proposal_matcher.BETWEEN_THRESHOLDS
                )  # -2
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        # BalancedPositiveNegativeSampler
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])

    def select_training_samples(
        self,
        proposals,  # type: List[Tensor]
        targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        if targets is None:
            raise ValueError("target should not be None.")

        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposal
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(
            proposals, gt_boxes, gt_labels
        )
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)

        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros(
                    (1, 4), dtype=dtype, device=device
                )
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]

        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [
            boxes_in_image.shape[0] for boxes_in_image in proposals
        ]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []

        for boxes, scores, image_shape in zip(
            pred_boxes_list, pred_scores_list, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove prediction with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            # gt: Computes input > other element-wise.
            # inds = torch.nonzero(torch.gt(scores, self.score_thresh)).squeeze(1)
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1.0)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximun suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
        distill=False,
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, (
                    "target boxes must of float type"
                )
                assert t["labels"].dtype == torch.int64, (
                    "target labels must of int64 type"
                )

        if (self.training or distill) and targets is not None:
            proposals, matched_idxs, labels, regression_targets = (
                self.select_training_samples(proposals, targets)
            )
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        # box_features_shape: [num_proposals, channel, height, width]
        box_features = self.box_roi_pool(features, proposals, image_shapes)

        # box_features_shape: [num_proposals, representation_size]
        box_features = self.box_head(box_features)

        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}

        if distill:
            if labels is not None and regression_targets is not None:
                loss_classifier, loss_box_reg = fastrcnn_loss(
                    class_logits, box_regression, labels, regression_targets
                )
                losses = {
                    "loss_classifier": loss_classifier,
                    "loss_box_reg": loss_box_reg,
                }
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
        else:
            if self.training:
                assert labels is not None and regression_targets is not None
                loss_classifier, loss_box_reg = fastrcnn_loss(
                    class_logits, box_regression, labels, regression_targets
                )
                losses = {
                    "loss_classifier": loss_classifier,
                    "loss_box_reg": loss_box_reg,
                }
            else:
                boxes, scores, labels = self.postprocess_detections(
                    class_logits, box_regression, proposals, image_shapes
                )
                num_images = len(boxes)
                for i in range(num_images):
                    result.append(
                        {
                            "boxes": boxes[i],
                            "labels": labels[i],
                            "scores": scores[i],
                        }
                    )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if (self.training or distill) and matched_idxs is not None:
                if matched_idxs is None:
                    raise ValueError(
                        "if in training, matched_idxs should not be None"
                    )

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[
                        0
                    ]  # 寻找对应gt类别大于0，即正样本
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            mask_features = self.mask_roi_pool(
                features, mask_proposals, image_shapes
            )
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)

            loss_mask = {}
            if distill:
                if not (
                    targets is None
                    or pos_matched_idxs is None
                    or mask_logits is None
                ):
                    gt_masks = [t["masks"] for t in targets]
                    gt_labels = [t["labels"] for t in targets]
                    rcnn_loss_mask = maskrcnn_loss(
                        mask_logits,
                        mask_proposals,
                        gt_masks,
                        gt_labels,
                        pos_matched_idxs,
                    )
                    loss_mask = {"loss_mask": rcnn_loss_mask}
                labels = [r["labels"] for r in result]
                mask_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(mask_probs, result):
                    r["masks"] = mask_prob
            else:
                if self.training:
                    if (
                        targets is None
                        or pos_matched_idxs is None
                        or mask_logits is None
                    ):
                        raise ValueError(
                            "targets, pos_matched_idxs, mask_logits cannot be None when training"
                        )

                    gt_masks = [t["masks"] for t in targets]
                    gt_labels = [t["labels"] for t in targets]
                    rcnn_loss_mask = maskrcnn_loss(
                        mask_logits,
                        mask_proposals,
                        gt_masks,
                        gt_labels,
                        pos_matched_idxs,
                    )
                    loss_mask = {"loss_mask": rcnn_loss_mask}
                else:
                    labels = [r["labels"] for r in result]
                    mask_probs = maskrcnn_inference(mask_logits, labels)
                    for mask_prob, r in zip(mask_probs, result):
                        r["masks"] = mask_prob

            losses.update(loss_mask)

        return result, losses
