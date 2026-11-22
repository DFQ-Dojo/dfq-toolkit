"""The Swin Transformer backbone, ported from MMDetection 2.22 to plain PyTorch.

Module and parameter names match MMDetection's `SwinTransformer`, so its checkpoints load
unchanged: `patch_embed.projection`, `stages.{i}.blocks.{j}.attn.w_msa.qkv`, `...ffn.layers.0.0`,
`stages.{i}.downsample.reduction`, `norm{i}`. The attention products run in `MatMul` modules
(`matmul1` for queries times keys, `matmul2` for attention times values), so that taps can read
the attention outputs. Source: mmdet/models/backbones/swin.py (Apache-2.0).
"""

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MatMul(nn.Module):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return a @ b


class AttentionValues(MatMul):
    """Attention weights times values: the output that the patch-similarity prior reads."""


class DropPath(nn.Module):
    """Drop whole residual branches per sample during training (stochastic depth)."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (keep + torch.rand(shape, dtype=x.dtype, device=x.device)).floor()
        return x.div(keep) * mask


def corner_padding(height: int, width: int, kernel: int, stride: int) -> tuple[int, int]:
    """Padding at the bottom and right so that a `kernel`/`stride` window covers the input."""
    out_height = math.ceil(height / stride)
    out_width = math.ceil(width / stride)
    pad_height = max((out_height - 1) * stride + kernel - height, 0)
    pad_width = max((out_width - 1) * stride + kernel - width, 0)
    return pad_height, pad_width


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dims: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels, embed_dims, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        pad_height, pad_width = corner_padding(
            x.shape[2], x.shape[3], self.patch_size, self.patch_size
        )
        padded = F.pad(x, [0, pad_width, 0, pad_height])
        projected = self.projection(padded)
        size = (projected.shape[2], projected.shape[3])
        tokens = projected.flatten(2).transpose(1, 2)
        return self.norm(tokens), size


class PatchMerging(nn.Module):
    """Join each 2x2 group of tokens and halve the resolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.sampler = nn.Unfold(kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(4 * in_channels)
        self.reduction = nn.Linear(4 * in_channels, out_channels, bias=False)

    def forward(self, x: Tensor, size: tuple[int, int]) -> tuple[Tensor, tuple[int, int]]:
        batch, _, channels = x.shape
        height, width = size
        grid = x.view(batch, height, width, channels).permute(0, 3, 1, 2)
        pad_height, pad_width = corner_padding(height, width, 2, 2)
        grid = F.pad(grid, [0, pad_width, 0, pad_height])
        out_size = (grid.shape[2] // 2, grid.shape[3] // 2)
        joined = self.sampler(grid).transpose(1, 2)
        return self.reduction(self.norm(joined)), out_size


def relative_position_index(window: int) -> Tensor:
    rows = torch.arange(0, (2 * window - 1) * window, 2 * window - 1)
    columns = torch.arange(0, window)
    coords = (rows[:, None] + columns[None, :]).reshape(1, -1)
    index = coords + coords.T
    return index.flip(1).contiguous()


class WindowMSA(nn.Module):
    """Multi-head self-attention inside one window, with a relative position bias."""

    def __init__(self, embed_dims: int, num_heads: int, window: int) -> None:
        super().__init__()
        self.window = window
        self.num_heads = num_heads
        self.scale = (embed_dims // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window - 1) ** 2, num_heads)
        )
        self.register_buffer("relative_position_index", relative_position_index(window))
        self.qkv = nn.Linear(embed_dims, embed_dims * 3)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(0.0)
        self.softmax = nn.Softmax(dim=-1)
        self.matmul1 = MatMul()
        self.matmul2 = AttentionValues()

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attention = self.matmul1(q * self.scale, k.transpose(-2, -1))
        index = self.relative_position_index
        assert isinstance(index, Tensor)
        bias = self.relative_position_bias_table[index.view(-1)].view(tokens, tokens, -1)
        attention = attention + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            windows = mask.shape[0]
            grouped = attention.view(batch // windows, windows, self.num_heads, tokens, tokens)
            attention = (grouped + mask.unsqueeze(1).unsqueeze(0)).view(
                -1, self.num_heads, tokens, tokens
            )
        attention = self.attn_drop(self.softmax(attention))
        mixed = self.matmul2(attention, v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(mixed))


def window_partition(x: Tensor, window: int) -> Tensor:
    batch, height, width, channels = x.shape
    grid = x.view(batch, height // window, window, width // window, window, channels)
    return grid.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window, window, channels)


def window_reverse(windows: Tensor, height: int, width: int, window: int) -> Tensor:
    batch = int(windows.shape[0] / (height * width / window / window))
    grid = windows.view(batch, height // window, width // window, window, window, -1)
    return grid.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


def shift_mask(height: int, width: int, window: int, shift: int, device: torch.device) -> Tensor:
    """-100 between tokens that a cyclic shift made neighbours, 0 elsewhere."""
    regions = torch.zeros((1, height, width, 1), device=device)
    slices = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
    count = 0
    for rows in slices:
        for columns in slices:
            regions[:, rows, columns, :] = count
            count += 1
    windows = window_partition(regions, window).view(-1, window * window)
    difference = windows.unsqueeze(1) - windows.unsqueeze(2)
    return difference.masked_fill(difference != 0, -100.0).masked_fill(difference == 0, 0.0)


class ShiftWindowMSA(nn.Module):
    def __init__(
        self, embed_dims: int, num_heads: int, window: int, shift: int, drop_path: float
    ) -> None:
        super().__init__()
        self.window = window
        self.shift = shift
        self.w_msa = WindowMSA(embed_dims, num_heads, window)
        self.drop = DropPath(drop_path)

    def forward(self, query: Tensor, size: tuple[int, int]) -> Tensor:
        batch, length, channels = query.shape
        height, width = size
        grid = query.view(batch, height, width, channels)
        pad_right = (self.window - width % self.window) % self.window
        pad_bottom = (self.window - height % self.window) % self.window
        grid = F.pad(grid, (0, 0, 0, pad_right, 0, pad_bottom))
        padded_height, padded_width = grid.shape[1], grid.shape[2]
        mask = None
        if self.shift > 0:
            grid = torch.roll(grid, shifts=(-self.shift, -self.shift), dims=(1, 2))
            mask = shift_mask(padded_height, padded_width, self.window, self.shift, query.device)
        windows = window_partition(grid, self.window).view(-1, self.window**2, channels)
        attended = self.w_msa(windows, mask).view(-1, self.window, self.window, channels)
        merged = window_reverse(attended, padded_height, padded_width, self.window)
        if self.shift > 0:
            merged = torch.roll(merged, shifts=(self.shift, self.shift), dims=(1, 2))
        cropped = merged[:, :height, :width, :].contiguous().view(batch, length, channels)
        return self.drop(cropped)


class FFN(nn.Module):
    """Two linear layers with GELU, laid out as MMCV's FFN: `layers.0.0` and `layers.1`."""

    def __init__(self, embed_dims: int, hidden: int, drop_path: float) -> None:
        super().__init__()
        first = nn.Sequential(nn.Linear(embed_dims, hidden), nn.GELU(), nn.Dropout(0.0))
        self.layers = nn.Sequential(first, nn.Linear(hidden, embed_dims), nn.Dropout(0.0))
        self.dropout_layer = DropPath(drop_path)

    def forward(self, x: Tensor, identity: Tensor) -> Tensor:
        return identity + self.dropout_layer(self.layers(x))


class SwinBlock(nn.Module):
    def __init__(
        self, embed_dims: int, num_heads: int, window: int, shift: bool, drop_path: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims)
        shift_size = window // 2 if shift else 0
        self.attn = ShiftWindowMSA(embed_dims, num_heads, window, shift_size, drop_path)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = FFN(embed_dims, 4 * embed_dims, drop_path)

    def forward(self, x: Tensor, size: tuple[int, int]) -> Tensor:
        attended = self.attn(self.norm1(x), size) + x
        return self.ffn(self.norm2(attended), identity=attended)


class SwinStage(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        depth: int,
        window: int,
        drop_paths: list[float],
        downsample: PatchMerging | None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            SwinBlock(embed_dims, num_heads, window, shift=i % 2 == 1, drop_path=drop_paths[i])
            for i in range(depth)
        )
        self.downsample = downsample

    def forward(self, x: Tensor, size: tuple[int, int]) -> tuple[Tensor, tuple[int, int], Tensor]:
        for block in self.blocks:
            x = block(x, size)
        if self.downsample is None:
            return x, size, x
        reduced, reduced_size = self.downsample(x, size)
        return reduced, reduced_size, x


class SwinTransformer(nn.Module):
    """Returns the normalized output of every stage as `OrderedDict` `"0"` to `"3"`, for an FPN."""

    def __init__(
        self,
        depths: tuple[int, ...],
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        embed_dims: int = 96,
        window: int = 7,
        drop_path_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(3, embed_dims, patch_size=4)
        self.drop_after_pos = nn.Dropout(0.0)
        rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.num_features = [embed_dims * 2**i for i in range(len(depths))]
        stages = []
        for i, depth in enumerate(depths):
            channels = self.num_features[i]
            last = i == len(depths) - 1
            downsample = None if last else PatchMerging(channels, 2 * channels)
            drop_paths = rates[sum(depths[:i]) : sum(depths[: i + 1])]
            stages.append(SwinStage(channels, num_heads[i], depth, window, drop_paths, downsample))
        self.stages = nn.ModuleList(stages)
        for i, channels in enumerate(self.num_features):
            self.add_module(f"norm{i}", nn.LayerNorm(channels))

    def forward(self, images: Tensor) -> OrderedDict[str, Tensor]:
        x, size = self.patch_embed(images)
        x = self.drop_after_pos(x)
        outputs: OrderedDict[str, Tensor] = OrderedDict()
        for i, stage in enumerate(self.stages):
            x, next_size, before = stage(x, size)
            norm = self.get_submodule(f"norm{i}")
            normalized = norm(before)
            grid = normalized.view(-1, *size, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
            outputs[str(i)] = grid
            size = next_size
        return outputs
