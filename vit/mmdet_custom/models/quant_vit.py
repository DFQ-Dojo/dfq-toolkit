from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers.mlp import Mlp

from mmdet_custom.models.vit import Attention, Block, PatchEmbed
from mmdet_custom.models.vit_baseline import ViTBaseline
from quant.lsqplus import QuantConv2dPlus, QuantLinearPlus
from quant.qatops import LSQLinear, LSQconv2d
from quant.quant_mode import QuantizeMode
from quant.quantize import convert_conv, convert_linear


class LSQPatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(
        self, img_size, patch_size, grid_size, num_patches, flatten, proj, norm
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.num_patches = num_patches
        self.flatten = flatten
        self.proj = proj
        self.norm = norm

    def start_quantize(self):
        if isinstance(self.proj, (LSQconv2d, QuantConv2dPlus)):
            self.proj.start_quantize()

    def forward(self, input: torch.Tensor):
        input = self.proj(input)
        _, _, H, W = input.shape
        if self.flatten:
            input = input.flatten(2).transpose(1, 2)  # BCHW -> BNC
        input = self.norm(input)
        return input, H, W


def convert_lsq_patch_embed(
    patch_embed: PatchEmbed, mode: QuantizeMode
) -> LSQPatchEmbed:
    return LSQPatchEmbed(
        img_size=patch_embed.img_size,
        patch_size=patch_embed.patch_size,
        grid_size=patch_embed.grid_size,
        num_patches=patch_embed.num_patches,
        flatten=patch_embed.flatten,
        proj=convert_conv(patch_embed.proj, mode),
        norm=deepcopy(patch_embed.norm),
    )


class LSQAttention(nn.Module):
    def __init__(self, num_heads, scale, qkv, attn_drop, proj, proj_drop):
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.qkv = qkv
        self.attn_drop = attn_drop
        self.proj = proj
        self.proj_drop = proj_drop

    def start_quantize(self):
        if isinstance(self.proj, (LSQLinear, QuantLinearPlus)):
            self.proj.start_quantize()
        if isinstance(self.qkv, (LSQLinear, QuantLinearPlus)):
            self.qkv.start_quantize()

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(
            0
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def convert_lsq_attention(attn: Attention, mode: QuantizeMode) -> LSQAttention:
    return LSQAttention(
        num_heads=attn.num_heads,
        scale=attn.scale,
        qkv=convert_linear(attn.qkv, mode),
        proj=convert_linear(attn.proj, mode),
        attn_drop=deepcopy(attn.attn_drop),
        proj_drop=deepcopy(attn.proj_drop),
    )


class LSQMlp(nn.Module):
    def __init__(self, fc1, act, fc2, drop):
        super().__init__()
        self.fc1 = fc1
        self.act = act
        self.fc2 = fc2
        self.drop = drop

    def start_quantize(self):
        if isinstance(self.fc1, (LSQLinear, QuantLinearPlus)):
            self.fc1.start_quantize()
        if isinstance(self.fc2, (LSQLinear, QuantLinearPlus)):
            self.fc2.start_quantize()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def convert_mlp(mlp: Mlp, mode: QuantizeMode) -> LSQMlp:
    return LSQMlp(
        fc1=convert_linear(mlp.fc1, mode),
        act=deepcopy(mlp.act),
        fc2=convert_linear(mlp.fc2, mode),
        drop=deepcopy(mlp.drop),
    )


class LSQBlock(nn.Module):
    def __init__(
        self,
        norm1,
        attn,
        norm2,
        mlp,
        gamma1=None,
        gamma2=None,
        residual=None,
        with_cp=False,
        drop_path=0.0,
        use_residual=False,
        layer_scale=False,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.use_residual = use_residual
        self.norm1 = norm1
        self.attn = attn
        self.drop_path = drop_path
        self.norm2 = norm2
        self.mlp = mlp
        self.layer_scale = layer_scale
        if self.layer_scale:
            self.gamma1 = gamma1
            self.gamma2 = gamma2
        if self.use_residual:
            self.residual = residual

    def start_quantize(self):
        if isinstance(self.attn, LSQAttention):
            self.attn.start_quantize()
        if isinstance(self.mlp, LSQMlp):
            self.mlp.start_quantize()

    def forward(self, x, H, W):

        if self.layer_scale:
            x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual:
            B, N, C = x.shape
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
            x = self.residual(x)
            x = x.permute(0, 2, 3, 1).reshape(B, N, C)

        return x


def convert_lsq_block(block: Block, mode: QuantizeMode) -> LSQBlock:
    return LSQBlock(
        norm1=deepcopy(block.norm1),
        attn=convert_lsq_attention(block.attn, mode),
        norm2=deepcopy(block.norm2),
        mlp=convert_mlp(block.mlp, mode),
        layer_scale=block.layer_scale,
        gamma1=block.gamma1 if block.layer_scale else None,
        gamma2=block.gamma2 if block.layer_scale else None,
        use_residual=block.use_residual,
        residual=block.residual if block.use_residual else None,
        with_cp=block.with_cp,
        drop_path=block.drop_path,
    )


def convert_lsq_blocks(
    blocks: nn.Sequential, mode: QuantizeMode
) -> nn.Sequential:
    return nn.Sequential(
        *[
            convert_lsq_block(block, mode)
            if isinstance(block, Block)
            else block
            for block in blocks
        ]
    )


class LSQViTBaseline(nn.Module):
    def __init__(
        self,
        pos_embed,
        patch_embed,
        pos_drop,
        out_indices,
        blocks,
        pretrain_size,
        norm1,
        norm2,
        norm3,
        norm4,
        up1,
        up2,
        up3,
        up4,
    ):
        super().__init__()
        self.pos_embed = pos_embed
        self.pretrain_size = pretrain_size
        self.patch_embed = patch_embed
        self.pos_drop = pos_drop
        self.out_indices = out_indices
        self.blocks = blocks
        self.norm1 = norm1
        self.norm2 = norm2
        self.norm3 = norm3
        self.norm4 = norm4
        self.up1 = up1
        self.up2 = up2
        self.up3 = up3
        self.up4 = up4

    def start_quantize(self):
        if isinstance(self.patch_embed, LSQPatchEmbed):
            self.patch_embed.start_quantize()
        for block in self.blocks:
            if isinstance(block, LSQBlock):
                block.start_quantize()

    def _get_pos_embed(self, pos_embed: torch.Tensor, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1
        ).permute(0, 3, 1, 2)
        pos_embed = (
            F.interpolate(
                pos_embed, size=(H, W), mode="bicubic", align_corners=False
            )
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
        )
        return pos_embed

    def forward_features(self, x):
        outs = []
        x, H, W = self.patch_embed(x)
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)
        for index, blk in enumerate(self.blocks):
            x = blk(x, H, W)
            if index in self.out_indices:
                outs.append(x)
        return outs, H, W

    def forward(self, x):
        outs, H, W = self.forward_features(x)
        if len(outs) == 1:  # for ViTDet
            f1 = f2 = f3 = f4 = outs[0]
        else:  # for ViT
            f1, f2, f3, f4 = outs
        bs, n, dim = f1.shape

        # Final Norm
        f1 = self.norm1(f1).transpose(1, 2).reshape(bs, dim, H, W)
        f2 = self.norm2(f2).transpose(1, 2).reshape(bs, dim, H, W)
        f3 = self.norm3(f3).transpose(1, 2).reshape(bs, dim, H, W)
        f4 = self.norm4(f4).transpose(1, 2).reshape(bs, dim, H, W)

        f1 = self.up1(f1).contiguous()
        f2 = self.up2(f2).contiguous()
        f3 = self.up3(f3).contiguous()
        f4 = self.up4(f4).contiguous()

        return [f1, f2, f3, f4]


def convert_vit_baseline(
    vit_baseline: ViTBaseline, mode: QuantizeMode
) -> LSQViTBaseline:
    return LSQViTBaseline(
        pos_embed=vit_baseline.pos_embed,
        patch_embed=convert_lsq_patch_embed(vit_baseline.patch_embed, mode),
        pos_drop=vit_baseline.pos_drop,
        out_indices=vit_baseline.out_indices,
        blocks=convert_lsq_blocks(vit_baseline.blocks, mode),
        pretrain_size=vit_baseline.pretrain_size,
        norm1=vit_baseline.norm1,
        norm2=vit_baseline.norm2,
        norm3=vit_baseline.norm3,
        norm4=vit_baseline.norm4,
        up1=vit_baseline.up1,
        up2=vit_baseline.up2,
        up3=vit_baseline.up3,
        up4=vit_baseline.up4,
    )
