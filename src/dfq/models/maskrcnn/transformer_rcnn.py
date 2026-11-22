"""Swin and DeiT Mask R-CNN from MMDetection-format COCO checkpoints, on torchvision.

The network follows MMDetection's Mask R-CNN configs that the checkpoints were trained with:
anchors 32 to 512 at ratios 0.5, 1, 2; at most 1000 proposals per image after NMS; aligned
RoIAlign. Quantization covers the backbone's convolutions and linear layers, as the old
`quant_swin.py` and `quant_vit.py` did; the FPN and heads stay in float.

- `swin_t`, `swin_s`: Swin with depths (2, 2, 6, 2) and (2, 2, 18, 2), from MMDetection's
  `mask_rcnn_swin-*-p4-w7_fpn_fp16_ms-crop-3x_coco` checkpoints.
- `deit_t`, `deit_s`: ViT with 192 or 384 channels, from ViT-Adapter's
  `mask_rcnn_deit_*_fpn_3x_coco` checkpoints.
"""

from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision.models.detection import MaskRCNN
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from dfq.models.maskrcnn.deit import Block, ViTBaseline
from dfq.models.maskrcnn.detector import MaskRCNNDetector
from dfq.models.maskrcnn.mmdet_parts import AlignedRoIPool, convert_state_dict
from dfq.models.maskrcnn.swin import AttentionValues, SwinBlock, SwinTransformer
from dfq.models.maskrcnn.transform import NormalizeOnly
from dfq.quant import QuantScope

SWIN_SCOPE = QuantScope(include=("backbone.body.*",))
DEIT_SCOPE = QuantScope(include=("backbone.body.patch_embed.*", "backbone.body.blocks.*"))

FEATURE_KINDS: dict[str, tuple[type[nn.Module], ...]] = {
    "block": (SwinBlock, Block),
    "layer_norm": (nn.LayerNorm,),
    "all": (SwinBlock, Block, nn.LayerNorm),
    "attention": (AttentionValues,),
}


SWIN_DEPTHS = {"swin_t": (2, 2, 6, 2), "swin_s": (2, 2, 18, 2)}
DEIT_WIDTHS = {"deit_t": (192, 3), "deit_s": (384, 6)}


def backbone_body(variant: str) -> SwinTransformer | ViTBaseline:
    if variant in SWIN_DEPTHS:
        return SwinTransformer(SWIN_DEPTHS[variant])
    dim, num_heads = DEIT_WIDTHS[variant]
    return ViTBaseline(dim=dim, num_heads=num_heads)


class BackboneWithFPN(nn.Module):
    def __init__(self, body: SwinTransformer | ViTBaseline) -> None:
        super().__init__()
        self.body = body
        self.fpn = FeaturePyramidNetwork(body.num_features, 256, extra_blocks=LastLevelMaxPool())
        self.out_channels = 256

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        features = self.body(images)
        pyramid: dict[str, Tensor] = self.fpn(features)
        return pyramid


def transformer_mask_rcnn(variant: str) -> MaskRCNN:
    backbone = BackboneWithFPN(backbone_body(variant))
    network = MaskRCNN(
        backbone,
        num_classes=81,
        rpn_post_nms_top_n_train=1000,
        box_roi_pool=AlignedRoIPool(7),
        mask_roi_pool=AlignedRoIPool(14),
    )
    network.transform = NormalizeOnly()
    return network


def transformer_detector(network: MaskRCNN, variant: str) -> MaskRCNNDetector:
    labels = torch.arange(1, 81)
    scope = SWIN_SCOPE if variant.startswith("swin") else DEIT_SCOPE
    return MaskRCNNDetector(network, labels, scope, FEATURE_KINDS)


def load_transformer(variant: str, weights: Path) -> MaskRCNNDetector:
    """Load an MMDetection-format checkpoint (`state_dict` with MMDetection names)."""
    network = transformer_mask_rcnn(variant)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = convert_state_dict(checkpoint["state_dict"])
    missing, unexpected = network.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"weights do not fit: missing {missing}, unexpected {unexpected}")
    return transformer_detector(network, variant)
