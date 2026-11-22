from types import MethodType
import torch.nn as nn
from mmdet_custom.models.vit import Attention
from mmdet.models.backbones.swin import WindowMSA
import math

import torch.nn.functional as F


def attention_forward(self, x, H, W):
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


def deit_window_attn(self, x, H, W):
    B, N, C = x.shape
    N_ = self.window_size * self.window_size
    H_ = math.ceil(H / self.window_size) * self.window_size
    W_ = math.ceil(W / self.window_size) * self.window_size

    qkv = self.qkv(x)  # [B, N, C]
    qkv = qkv.transpose(1, 2).reshape(B, C * 3, H, W)  # [B, C, H, W]
    qkv = F.pad(qkv, [0, W_ - W, 0, H_ - H], mode="constant")

    qkv = F.unfold(
        qkv,
        kernel_size=(self.window_size, self.window_size),
        stride=(self.window_size, self.window_size),
    )
    B, C_kw_kw, L = qkv.shape  # L - the num of windows
    qkv = qkv.reshape(B, C * 3, N_, L).permute(0, 3, 2, 1)  # [B, L, N_, C]
    qkv = qkv.reshape(B, L, N_, 3, self.num_heads, C // self.num_heads).permute(
        3, 0, 1, 4, 2, 5
    )
    q, k, v = qkv.unbind(
        0
    )  # make torchscript happy (cannot use tensor as tuple)

    # q,k,v [B, L, num_head, N_, C/num_head]
    # attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, L, num_head, N_, N_]
    attn = (
        self.matmul1(q, k.transpose(-2, -1)) * self.scale
    )  # [B, L, num_head, N_, N_]

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)  # [B, L, num_head, N_, N_]
    # attn @ v = [B, L, num_head, N_, C/num_head]
    # x = (attn @ v).permute(0, 2, 4, 3, 1).reshape(B, C_kw_kw // 3, L)
    x = self.matmul2(attn, v).permute(0, 2, 4, 3, 1).reshape(B, C_kw_kw // 3, L)

    x = F.fold(
        x,
        output_size=(H_, W_),
        kernel_size=(self.window_size, self.window_size),
        stride=(self.window_size, self.window_size),
    )  # [B, C, H_, W_]
    x = x[:, :, :H, :W].reshape(B, C, N).transpose(-1, -2)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def windowmsa_forward(self, x, mask=None):
    """
    Args:

        x (tensor): input features with shape of (num_windows*B, N, C)
        mask (tensor | None, Optional): mask with shape of (num_windows,
            Wh*Ww, Wh*Ww), value should be between (-inf, 0].
    """
    B, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    # make torchscript happy (cannot use tensor as tuple)
    q, k, v = qkv[0], qkv[1], qkv[2]

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
        attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
            1
        ).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
    attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B


def annotate_model(net):
    """
    Get a vision transformer model.
    This will replace matrix multiplication operations with matmul modules in the model.
    """
    for _, module in net.named_modules():
        if isinstance(module, Attention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(attention_forward, module)
        if isinstance(module, WindowMSA):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(windowmsa_forward, module)

    net.eval()
    return net
