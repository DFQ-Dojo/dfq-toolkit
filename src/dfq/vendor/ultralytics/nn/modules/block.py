# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Blocks of YOLO11, from ultralytics 8.3.53 (`ultralytics/nn/modules/block.py`).

Class and attribute names match Ultralytics checkpoints, which pickle these classes by name.
`C3k2` and `C3k` stand alone here; upstream they subclass `C2f` and `C3`, which YOLO11 does
not use otherwise.
"""

import torch
from torch import Tensor, nn

from .conv import Conv


class DFL(nn.Module):
    """Distribution focal loss decoding: the expected value of each side's `c1`-bin distribution."""

    def __init__(self, c1: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        bins = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = bins.view(1, c1, 1, 1)
        self.c1 = c1

    def forward(self, x: Tensor) -> Tensor:
        batch, _, anchors = x.shape
        distributions = x.view(batch, 4, self.c1, anchors).transpose(2, 1).softmax(1)
        return self.conv(distributions).view(batch, 4, anchors)


class SPPF(nn.Module):
    """Spatial pyramid pooling: three chained max-pools, concatenated with their input."""

    def __init__(self, c1: int, c2: int, k: int = 5) -> None:
        super().__init__()
        hidden = c1 // 2
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(hidden * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: Tensor) -> Tensor:
        outputs = [self.cv1(x)]
        for _ in range(3):
            outputs.append(self.m(outputs[-1]))
        return self.cv2(torch.cat(outputs, 1))


class Bottleneck(nn.Module):
    """Two convolutions with kernels `kernels`, and a residual connection when shapes allow."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        kernels: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, kernels[0], 1)
        self.cv2 = Conv(hidden, c2, kernels[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: Tensor) -> Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k(nn.Module):
    """CSP bottleneck with three convolutions and `n` bottlenecks of kernel `k`."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
    ) -> None:
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, c2, 1)
        blocks = (Bottleneck(hidden, hidden, shortcut, g, (k, k), e=1.0) for _ in range(n))
        self.m = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        branch = self.m(self.cv1(x))
        return self.cv3(torch.cat((branch, self.cv2(x)), 1))


class C3k2(nn.Module):
    """CSP block that splits its input, runs `n` blocks on one half, and joins every output.

    The blocks are `C3k` when `c3k` is set (YOLO11 m, l and x), otherwise bottlenecks.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        blocks = (
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )
        self.m = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        outputs = list(self.cv1(x).chunk(2, 1))
        for block in self.m:
            outputs.append(block(outputs[-1]))
        return self.cv2(torch.cat(outputs, 1))


class Attention(nn.Module):
    """Multi-head self-attention over all positions, with a depth-wise positional encoding."""

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        channels = dim + 2 * self.key_dim * num_heads
        self.qkv = Conv(dim, channels, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        per_head = self.key_dim * 2 + self.head_dim
        qkv = self.qkv(x).view(batch, self.num_heads, per_head, height * width)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        weights = ((q.transpose(-2, -1) @ k) * self.scale).softmax(dim=-1)
        attended = (v @ weights.transpose(-2, -1)).view(batch, channels, height, width)
        encoded = self.pe(v.reshape(batch, channels, height, width))
        return self.proj(attended + encoded)


class PSABlock(nn.Module):
    """Attention, then a feed-forward network, each with a residual connection."""

    def __init__(
        self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True
    ) -> None:
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: Tensor) -> Tensor:
        attended = x + self.attn(x) if self.add else self.attn(x)
        return attended + self.ffn(attended) if self.add else self.ffn(attended)


class C2PSA(nn.Module):
    """Split the input, run `n` attention blocks on one half, and join the halves."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5) -> None:
        super().__init__()
        assert c1 == c2, "C2PSA keeps the channel count"
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        blocks = (PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))
        self.m = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        kept, attended = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((kept, self.m(attended)), 1))
