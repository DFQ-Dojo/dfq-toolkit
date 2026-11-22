"""What a Mask R-CNN needs to compute what MMDetection 2's computed, and to load its weights.

- `AlignedRoIPool` pools regions as MMDetection's `SingleRoIExtractor` with `RoIAlign`
  (`aligned=True`, `sampling_ratio=0`); torchvision's pooler uses `aligned=False`. Both assign a box
  of side `s` to level `floor(log2(s / 56))`.
- `convert_state_dict` renames an MMDetection checkpoint's keys to torchvision's and moves the
  background class from the last index to index 0 (`fc_cls`), adding zero rows where MMDetection
  has no background output (`fc_reg`, `conv_logits`).
"""

import torch
from torch import Tensor
from torchvision.ops import MultiScaleRoIAlign, roi_align

STRIDES = (4, 8, 16, 32)


class AlignedRoIPool(MultiScaleRoIAlign):
    """A `MultiScaleRoIAlign`, as torchvision's Mask R-CNN requires, with MMDetection's pooling."""

    def __init__(self, output_size: int) -> None:
        names = [str(i) for i in range(len(STRIDES))]
        super().__init__(featmap_names=names, output_size=output_size, sampling_ratio=0)
        self.side = output_size

    def forward(
        self, features: dict[str, Tensor], boxes: list[Tensor], image_shapes: list[tuple[int, int]]
    ) -> Tensor:
        levels = [features[str(i)] for i in range(len(STRIDES))]
        indices = [
            torch.full((len(b), 1), i, dtype=b.dtype, device=b.device) for i, b in enumerate(boxes)
        ]
        rois = torch.cat([torch.cat(indices), torch.cat(boxes)], dim=1)
        sides = ((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2])).sqrt()
        assigned = torch.floor(torch.log2(sides / 56 + 1e-6)).clamp(0, len(STRIDES) - 1).long()
        channels = levels[0].shape[1]
        size = self.side
        pooled = rois.new_zeros((len(rois), channels, size, size))
        for level, (feature, stride) in enumerate(zip(levels, STRIDES, strict=True)):
            chosen = torch.nonzero(assigned == level).squeeze(1)
            region = roi_align(
                feature,
                rois[chosen],
                (size, size),
                spatial_scale=1 / stride,
                sampling_ratio=0,
                aligned=True,
            )
            pooled[chosen] = region.to(pooled.dtype)
        return pooled


RENAMES = (
    ("backbone.", "backbone.body."),
    ("neck.lateral_convs.", "backbone.fpn.inner_blocks."),
    ("neck.fpn_convs.", "backbone.fpn.layer_blocks."),
    ("rpn_head.rpn_conv.", "rpn.head.conv.0.0."),
    ("rpn_head.rpn_cls.", "rpn.head.cls_logits."),
    ("rpn_head.rpn_reg.", "rpn.head.bbox_pred."),
    ("roi_head.bbox_head.shared_fcs.0.", "roi_heads.box_head.fc6."),
    ("roi_head.bbox_head.shared_fcs.1.", "roi_heads.box_head.fc7."),
    ("roi_head.bbox_head.fc_cls.", "roi_heads.box_predictor.cls_score."),
    ("roi_head.bbox_head.fc_reg.", "roi_heads.box_predictor.bbox_pred."),
    ("roi_head.mask_head.convs.", "roi_heads.mask_head."),
    ("roi_head.mask_head.upsample.", "roi_heads.mask_predictor.conv5_mask."),
    ("roi_head.mask_head.conv_logits.", "roi_heads.mask_predictor.mask_fcn_logits."),
)


def renamed(key: str) -> str:
    for old, new in RENAMES:
        if key.startswith(old):
            key = new + key[len(old) :]
            break
    # MMCV's ConvModule names its convolution `conv`; torchvision's Conv2dNormActivation `0`.
    for prefix in ("backbone.fpn.", "roi_heads.mask_head."):
        if key.startswith(prefix):
            key = key.replace(".conv.", ".0.")
    return key


def background_first(key: str, value: Tensor) -> Tensor:
    if key.startswith("roi_heads.box_predictor.cls_score."):
        return torch.cat([value[-1:], value[:-1]])
    if key.startswith("roi_heads.box_predictor.bbox_pred."):
        return torch.cat([value.new_zeros((4, *value.shape[1:])), value])
    if key.startswith("roi_heads.mask_predictor.mask_fcn_logits."):
        return torch.cat([value.new_zeros((1, *value.shape[1:])), value])
    return value


def convert_state_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
    converted = {}
    for key, value in state.items():
        name = renamed(key)
        converted[name] = background_first(name, value)
    return converted
