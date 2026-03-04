from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.backbones.swin import (
    WindowMSA,
    ShiftWindowMSA,
    SwinBlock,
    SwinBlockSequence,
    SwinTransformer,
)

from quant.lsqplus import QuantLinearPlus, QuantConv2dPlus
from quant.qatops import LSQLinear, LSQconv2d
from quant.ptq import QLinear, Qconv2d
from quant.quantize_ptq import convert_conv, convert_linear
from mmcv.cnn.bricks.transformer import FFN
from mmdet.models.utils.transformer import PatchMerging, PatchEmbed

from quant.quant_mode import QuantizeMode


class LSQWindowMSA(nn.Module):
    def __init__(
        self,
        embed_dims,
        window_size,
        num_heads,
        scale,
        relative_position_bias_table,
        relative_position_index,
        qkv: nn.Module,
        attn_drop,
        proj: nn.Module,
        proj_drop,
        softmax,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = scale
        self.relative_position_bias_table = relative_position_bias_table
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = qkv
        self.attn_drop = attn_drop
        self.proj = proj
        self.proj_drop = proj_drop
        self.softmax = softmax

    def start_quantize(self):
        if isinstance(self.qkv, (QLinear, LSQLinear, QuantLinearPlus)):
            self.qkv.start_quantize()
        if isinstance(self.proj, (QLinear, LSQLinear, QuantLinearPlus)):
            self.proj.start_quantize()

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(
                B, N, 3, self.num_heads, C // self.num_heads
            )  # B N 3 #heads #dim per heads
            .permute(2, 0, 3, 1, 4)  # 3 B #heads N #dim per heads
        )
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # B #heads N #dim per heads
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

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
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


def convert_windowmsa(window_msa: WindowMSA, mode: QuantizeMode) -> LSQWindowMSA:
    return LSQWindowMSA(
        embed_dims=window_msa.embed_dims,
        window_size=window_msa.window_size,
        num_heads=window_msa.num_heads,
        scale=window_msa.scale,
        relative_position_bias_table=window_msa.relative_position_bias_table,
        relative_position_index=window_msa.relative_position_index,
        qkv=convert_linear(window_msa.qkv, mode),
        attn_drop=window_msa.attn_drop,
        proj=convert_linear(window_msa.proj, mode),
        proj_drop=window_msa.proj_drop,
        softmax=window_msa.softmax,
    )


class LSQShiftWindowMSA(nn.Module):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self, window_size, shift_size, w_msa, drop):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.w_msa = w_msa
        self.drop = drop

    def start_quantize(self):
        if isinstance(self.w_msa, LSQWindowMSA):
            self.w_msa.start_quantize()

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, "input feature has wrong size"
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(
            B, H // window_size, W // window_size, window_size, window_size, -1
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


def convert_shifted_windowmsa(
    shift_windowmsa: ShiftWindowMSA, quantize_mode: QuantizeMode
) -> LSQShiftWindowMSA:
    return LSQShiftWindowMSA(
        window_size=shift_windowmsa.window_size,
        shift_size=shift_windowmsa.shift_size,
        w_msa=convert_windowmsa(shift_windowmsa.w_msa, quantize_mode),
        drop=shift_windowmsa.drop,
    )


class LSQFFN(nn.Module):
    def __init__(
        self,
        layers,
        dropout_layer,
        add_identity,
    ):
        super().__init__()

        self.layers = layers
        self.dropout_layer = dropout_layer
        self.add_identity = add_identity

    def start_quantize(self):
        for layer in self.layers:
            if isinstance(layer, (QLinear, LSQLinear)):
                layer.start_quantize()
            elif isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, (QLinear, LSQLinear)):
                        sublayer.start_quantize()
            else:
                pass

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


def convert_ffn(ffn: FFN, mode: QuantizeMode) -> LSQFFN:

    def convert_sublayer(sublayer: nn.Module):
        if isinstance(sublayer, nn.Linear):
            return convert_linear(sublayer, mode)
        if isinstance(sublayer, nn.Sequential):
            return convert_layer(sublayer)
        else:
            return sublayer

    def convert_layer(layers: nn.Sequential):
        return nn.Sequential(*[convert_sublayer(layer) for layer in layers])

    return LSQFFN(
        layers=convert_layer(ffn.layers),
        dropout_layer=ffn.dropout_layer,
        add_identity=ffn.add_identity,
    )


class LSQSwinBlock(nn.Module):

    def __init__(self, norm1, attn, norm2, ffn):
        super().__init__()
        self.norm1 = norm1
        self.attn = attn
        self.norm2 = norm2
        self.ffn = ffn

    def start_quantize(self):
        if isinstance(self.attn, LSQShiftWindowMSA):
            self.attn.start_quantize()
        if isinstance(self.ffn, LSQFFN):
            self.ffn.start_quantize()

    def forward(self, x, hw_shape):

        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)

        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)

        return x


def convert_swinblock(
    swinblock: SwinBlock, quantize_mode: QuantizeMode
) -> LSQSwinBlock:
    return LSQSwinBlock(
        norm1=swinblock.norm1,
        attn=convert_shifted_windowmsa(swinblock.attn, quantize_mode),
        norm2=swinblock.norm2,
        ffn=convert_ffn(swinblock.ffn, quantize_mode),
    )


class LSQPatchMerging(nn.Module):
    def __init__(self, adap_padding, sampler, norm, reduction):
        super().__init__()
        self.adap_padding = adap_padding
        self.sampler = sampler
        self.norm = norm
        self.reduction = reduction

    def start_quantize(self):
        if isinstance(self.reduction, (QLinear, LSQLinear)):
            self.reduction.start_quantize()

    def forward(self, x, input_size):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (Merged_H, Merged_W).
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), (
            f"Expect " f"input_size is " f"`Sequence` " f"but get {input_size}"
        )

        H, W = input_size
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W
        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility

        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]

        x = self.sampler(x)
        # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)

        out_h = (
            H
            + 2 * self.sampler.padding[0]
            - self.sampler.dilation[0] * (self.sampler.kernel_size[0] - 1)
            - 1
        ) // self.sampler.stride[0] + 1
        out_w = (
            W
            + 2 * self.sampler.padding[1]
            - self.sampler.dilation[1] * (self.sampler.kernel_size[1] - 1)
            - 1
        ) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size


def convert_patch_merging(
    patch_merging: PatchMerging, quantize_mode: QuantizeMode
) -> LSQPatchMerging:
    return LSQPatchMerging(
        adap_padding=patch_merging.adap_padding,
        sampler=patch_merging.sampler,
        norm=patch_merging.norm,
        reduction=convert_linear(patch_merging.reduction, quantize_mode),
    )


class LSQSwinBlockSequence(nn.Module):

    def __init__(self, blocks, downsample):
        super().__init__()

        self.blocks = blocks
        self.downsample = downsample

    def start_quantize(self):
        for block in self.blocks:
            if isinstance(block, LSQSwinBlock):
                block.start_quantize()
        if isinstance(self.downsample, LSQPatchMerging):
            self.downsample.start_quantize()

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


def convert_swinblock_seq(
    swinblock_seq: SwinBlockSequence, quantize_mode: QuantizeMode
) -> LSQSwinBlockSequence:
    blocks = nn.ModuleList()

    for block in swinblock_seq.blocks:
        blocks.append(convert_swinblock(block, quantize_mode))

    downsample = None
    if isinstance(swinblock_seq.downsample, PatchMerging):
        downsample = convert_patch_merging(swinblock_seq.downsample, quantize_mode)
    elif downsample is None:
        pass
    else:
        raise NotImplementedError(f"please impl for {swinblock_seq.downsample}")

    return LSQSwinBlockSequence(blocks=blocks, downsample=downsample)


class LSQPatchEmbedSwin(nn.Module):

    def __init__(self, adap_padding, projection, norm):
        super().__init__()
        self.adap_padding = adap_padding
        self.projection = projection
        self.norm = norm

    def start_quantize(self):
        if isinstance(self.projection, (Qconv2d, LSQconv2d, QuantConv2dPlus)):
            self.projection.start_quantize()

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w).
        """

        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


def convert_patch_embed_swin(pes: PatchEmbed, mode: QuantizeMode) -> LSQPatchEmbedSwin:
    return LSQPatchEmbedSwin(
        adap_padding=pes.adap_padding,
        projection=convert_conv(pes.projection, mode),
        norm=pes.norm,
    )


class LSQSwinTransformer(nn.Module):

    def __init__(
        self,
        convert_weights,
        frozen_stages,
        out_indices,
        use_abs_pos_embed,
        patch_embed,
        absolute_pos_embed,
        drop_after_pos,
        stages,
        num_features,
        norm_layers,
    ):
        super().__init__()
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        self.patch_embed = patch_embed
        self.absolute_pos_embed = absolute_pos_embed
        self.drop_after_pos = drop_after_pos
        self.stages = stages
        self.num_features = num_features
        for name, layer in norm_layers:
            self.add_module(name, layer)

    def start_quantize(self):
        if isinstance(self.patch_embed, LSQPatchEmbedSwin):
            self.patch_embed.start_quantize()
        for stage in self.stages:
            if isinstance(stage, LSQSwinBlockSequence):
                stage.start_quantize()

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f"norm{i-1}")
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                out = norm_layer(out)
                out = (
                    out.view(-1, *out_hw_shape, self.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(out)

        return outs


def convert_swin_transformer(
    transformer: SwinTransformer, mode: QuantizeMode
) -> LSQSwinTransformer:

    norm_layers = []
    for i in transformer.out_indices:
        name = f"norm{i}"
        norm_layers.append((name, getattr(transformer, name)))

    return LSQSwinTransformer(
        convert_weights=transformer.convert_weights,
        frozen_stages=transformer.frozen_stages,
        out_indices=transformer.out_indices,
        use_abs_pos_embed=transformer.use_abs_pos_embed,
        absolute_pos_embed=(
            transformer.absolute_pos_embed if transformer.use_abs_pos_embed else None
        ),
        patch_embed=convert_patch_embed_swin(transformer.patch_embed, mode),
        drop_after_pos=transformer.drop_after_pos,
        stages=nn.ModuleList(
            convert_swinblock_seq(seq, mode) for seq in transformer.stages
        ),
        num_features=transformer.num_features,
        norm_layers=norm_layers,
    )
