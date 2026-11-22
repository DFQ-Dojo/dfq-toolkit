import torch.nn as nn
import timm

from tools.annotate_model import annotate_model


def attention_forward(self, x):
    B, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(
        0
    )  # make torchscript happy (cannot use tensor as tuple)

    # attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    del q, k

    # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    del attn, v
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def window_attention_forward(self, x, mask=None):
    B_, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(
        0
    )  # make torchscript happy (cannot use tensor as tuple)

    q = q * self.scale
    # attn = (q @ k.transpose(-2, -1))
    attn = self.matmul1(q, k.transpose(-2, -1))

    relative_position_bias = self.relative_position_bias_table[
        self.relative_position_index.view(-1)
    ].view(
        self.window_size[0] * self.window_size[1],
        self.window_size[0] * self.window_size[1],
        -1,
    )  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(
        2, 0, 1
    ).contiguous()  # nH, Wh*Ww, Wh*Ww
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
            1
        ).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B


class ViTClassifier(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = head

    def forward(self, x):
        """
        backbone output is of shape (B, C, H, W)
        """
        feature_map = self.backbone(x)[0]
        B, C, H, W = feature_map.shape
        feature_map = feature_map.reshape(B, C, H * W)
        pooled_feature = self.pool(feature_map).flatten(1)
        return self.head(pooled_feature)


def build_model(name, Pretrained=True):
    """
    Get a vision transformer model.
    This will replace matrix multiplication operations with matmul modules in the model.

    Currently support almost all models in timm.models.transformers, including:
    - vit_tiny/small/base/large_patch16/patch32_224/384,
    - deit_tiny/small/base(_distilled)_patch16_224,
    - deit_base(_distilled)_patch16_384,
    - swin_tiny/small/base/large_patch4_window7_224,
    - swin_base/large_patch4_window12_384

    These models are finetuned on imagenet-1k and should use ViTImageNetLoaderGenerator
    for calibration and testing.
    """
    classifier_net = timm.create_model(name, pretrained=Pretrained)

    from mmcv import Config

    cfg = Config.fromfile(
        "./configs/mask_rcnn/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py"
    )

    cfg.model.backbone.out_indices = (3,)
    cfg.model.backbone.convert_weights = True

    cfg.model.backbone.init_cfg = Config(
        dict(
            checkpoint="./pretrained/swin_tiny_patch4_window7_224.pth",
        )
    )

    from mmdet.models.builder import build_detector

    backbone_model = build_detector(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    backbone_model.backbone.init_weights()

    model = ViTClassifier(
        backbone=backbone_model.backbone, head=classifier_net.head
    )

    return annotate_model(model)
