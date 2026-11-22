"""ResNet-50 FPN Mask R-CNN from torchvision's COCO weights.

The backbone uses ordinary `BatchNorm2d`, as the paper's ResNet code did, so that the batch-norm
prior can read its statistics; `conv1` and `layer1` stay frozen (three trainable stages). The
weights file is torchvision's `maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth`.

Quantization covers the convolutions of `layer1` to `layer4`, except the last bottleneck of
`layer4`, as before. The stem, FPN and heads stay in float.
"""

from pathlib import Path

import torch
from torch import nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import Bottleneck

from dfq.detection.coco import coco_classes
from dfq.models.maskrcnn.detector import MaskRCNNDetector
from dfq.models.maskrcnn.transform import NormalizeOnly
from dfq.quant import QuantScope

SCOPE = QuantScope(include=("backbone.body.layer*",), exclude=("backbone.body.layer4.2.*",))

FEATURE_KINDS: dict[str, tuple[type[nn.Module], ...]] = {
    "cnn": (nn.Conv2d,),
    "bn": (nn.BatchNorm2d,),
    "cnnbn": (nn.Conv2d, nn.BatchNorm2d),
    "block": (Bottleneck,),
}


def resnet50_mask_rcnn() -> MaskRCNN:
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50", weights=None, norm_layer=nn.BatchNorm2d, trainable_layers=3
    )
    network = MaskRCNN(backbone, num_classes=91)
    network.transform = NormalizeOnly()
    return network


def resnet50_detector(network: MaskRCNN) -> MaskRCNNDetector:
    labels = torch.tensor([item.coco_id for item in coco_classes()])
    return MaskRCNNDetector(network, labels, SCOPE, FEATURE_KINDS)


def load_resnet50(weights: Path) -> MaskRCNNDetector:
    network = resnet50_mask_rcnn()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    missing, unexpected = network.load_state_dict(state, strict=False)
    untracked = [name for name in missing if not name.endswith("num_batches_tracked")]
    if untracked or unexpected:
        raise ValueError(f"weights do not fit: missing {untracked}, unexpected {unexpected}")
    return resnet50_detector(network)
