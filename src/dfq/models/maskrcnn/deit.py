"""The plain ViT (DeiT) backbone of ViT-Adapter's Mask R-CNN baselines, in plain PyTorch.

Ported from the old `vit/mmdet_custom/models/vit.py` and `vit_baseline.py` (ViT-Adapter,
Apache-2.0), which built on timm. Module and parameter names match those checkpoints
(`patch_embed.proj`, `blocks.{i}.attn.qkv`, `blocks.{i}.mlp.fc1`, `blocks.{i}.gamma1`, `norm1` to
`norm4`, `up1`, `up2`).
Blocks 2, 5, 8 and 11 attend globally; the others attend within 14 x 14 windows. The outputs of
blocks 2, 5, 8 and 11 are upsampled 4x and 2x, kept, and downsampled 2x into a four-level pyramid.
"""

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dfq.models.maskrcnn.swin import AttentionValues, DropPath, MatMul

GLOBAL_BLOCKS = (2, 5, 8, 11)


class Attention(nn.Module):
    """Self-attention over all tokens, or within `window x window` windows when `window` is set."""

    def __init__(self, dim: int, num_heads: int, window: int | None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.window = window
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
        self.matmul1 = MatMul()
        self.matmul2 = AttentionValues()

    def attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        weights = self.matmul1(q, k.transpose(-2, -1)) * self.scale
        weights = self.attn_drop(weights.softmax(dim=-1))
        return self.matmul2(weights, v)

    def global_attention(self, x: Tensor) -> Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        return self.attend(q, k, v).transpose(1, 2).reshape(batch, tokens, channels)

    def window_attention(self, x: Tensor, height: int, width: int, window: int) -> Tensor:
        batch, tokens, channels = x.shape
        size = window * window
        padded_height = math.ceil(height / window) * window
        padded_width = math.ceil(width / window) * window
        qkv = self.qkv(x).transpose(1, 2).reshape(batch, channels * 3, height, width)
        qkv = F.pad(qkv, [0, padded_width - width, 0, padded_height - height])
        qkv = F.unfold(qkv, kernel_size=(window, window), stride=(window, window))
        windows = qkv.shape[2]
        qkv = qkv.reshape(batch, channels * 3, size, windows).permute(0, 3, 2, 1)
        qkv = qkv.reshape(batch, windows, size, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv.permute(3, 0, 1, 4, 2, 5).unbind(0)
        mixed = self.attend(q, k, v).permute(0, 2, 4, 3, 1).reshape(batch, channels * size, windows)
        folded = F.fold(
            mixed, (padded_height, padded_width), (window, window), stride=(window, window)
        )
        return folded[:, :, :height, :width].reshape(batch, channels, tokens).transpose(-1, -2)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        if self.window is None:
            mixed = self.global_attention(x)
        else:
            mixed = self.window_attention(x, height, width, self.window)
        return self.proj_drop(self.proj(mixed))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.0)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(hidden))


class Block(nn.Module):
    """A transformer block with layer scale (`gamma1`, `gamma2`)."""

    def __init__(self, dim: int, num_heads: int, window: int | None, drop_path: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, window)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, 4 * dim)
        self.gamma1 = nn.Parameter(torch.ones(dim))
        self.gamma2 = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        attended = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x), height, width))
        return attended + self.drop_path(self.gamma2 * self.mlp(self.norm2(attended)))


class PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch_size: int = 16) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        projected = self.proj(x)
        height, width = projected.shape[2:]
        return projected.flatten(2).transpose(1, 2), height, width


class ViTBaseline(nn.Module):
    """Returns the four pyramid levels as `OrderedDict` `"0"` to `"3"`, each with `dim` channels."""

    def __init__(
        self, dim: int, num_heads: int, depth: int = 12, drop_path_rate: float = 0.1
    ) -> None:
        super().__init__()
        self.num_features = [dim] * 4
        self.patch_embed = PatchEmbed(dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 14 * 14 + 1, dim))
        self.pos_drop = nn.Dropout(0.0)
        rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *(
                Block(dim, num_heads, None if i in GLOBAL_BLOCKS else 14, rates[i])
                for i in range(depth)
            )
        )
        for i in range(1, 5):
            self.add_module(f"norm{i}", nn.LayerNorm(dim, eps=1e-6))
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 2, 2),
            nn.GroupNorm(32, dim),
            nn.GELU(),
            nn.ConvTranspose2d(dim, dim, 2, 2),
        )
        self.up2 = nn.ConvTranspose2d(dim, dim, 2, 2)
        self.up3 = nn.Identity()
        self.up4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def position_embedding(self, height: int, width: int) -> Tensor:
        """The pretrained 14 x 14 position embedding, resized bicubically to the token grid."""
        grid = self.pos_embed[:, 1:].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
        resized = F.interpolate(grid, size=(height, width), mode="bicubic", align_corners=False)
        return resized.reshape(1, -1, height * width).permute(0, 2, 1)

    def forward(self, images: Tensor) -> OrderedDict[str, Tensor]:
        x, height, width = self.patch_embed(images)
        x = self.pos_drop(x + self.position_embedding(height, width))
        kept = []
        for index, block in enumerate(self.blocks):
            x = block(x, height, width)
            if index in GLOBAL_BLOCKS:
                kept.append(x)
        samplers = [self.up1, self.up2, self.up3, self.up4]
        outputs: OrderedDict[str, Tensor] = OrderedDict()
        for level, (tokens, sampler) in enumerate(zip(kept, samplers, strict=True)):
            norm = self.get_submodule(f"norm{level + 1}")
            batch, _, channels = tokens.shape
            grid = norm(tokens).transpose(1, 2).reshape(batch, channels, height, width)
            outputs[str(level)] = sampler(grid).contiguous()
        return outputs
