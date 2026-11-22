from copy import deepcopy
import torch
import torch.nn as nn

from quant.quant_mode import QuantizeMode, QuantizeKind

from quant.qatops import LSQconv2d  # type: ignore
from quant.lsqplus import QuantConv2dPlus  # type: ignore
from ultralytics.nn.modules import (  # type: ignore
    Conv,
)

from ultralytics.nn.modules.block import (  # type: ignore
    C3k2,
    SPPF,
    C2PSA,
    C2f,
    C3,
    C3k,
    Bottleneck,
    Attention,
    PSABlock,
)


def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = (
            d * (k - 1) + 1
            if isinstance(k, int)
            else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def _build_inner_conv(c1, c2, k, s, p, g, d, mode: QuantizeMode):
    if mode.kind == QuantizeKind.EXACT or (
        mode.weight_bits == 32 and mode.activation_bits == 32
    ):
        return nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            dilation=d,
            bias=False,
        )
    if mode.kind == QuantizeKind.SYMMETRIC_QUANTIZE:
        return LSQconv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            dilation=d,
            bias=False,
            weight_bits=mode.weight_bits,
            activation_bits=mode.activation_bits,
            mode="quantize_sym",
        )
    else:
        return QuantConv2dPlus(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            dilation=d,
            bias=False,
            a_bits=mode.activation_bits,
            w_bits=mode.weight_bits,
        )


class LSQConv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, conv, bn: nn.BatchNorm2d, act):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.act = act

    @staticmethod
    def new(
        self,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        d=1,
        act=True,
        num_bits=8,
        mode="exact",
    ):
        """Initializes a standard convolution layer with optional batch normalization and activation."""

        conv = _build_inner_conv(
            c1, c2, k, s, autopad(k, p, d), g, d, num_bits=num_bits, mode=mode
        )
        bn = nn.BatchNorm2d(c2)
        act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

        return LSQConv(conv, bn, act)

    def start_quantize(self):
        if isinstance(self.conv, (LSQconv2d, QuantConv2dPlus)):
            self.conv.start_quantize()

    def start_prune(self, sparsity=None, prune_mode=None):
        self.conv.start_prune(sparsity, prune_mode)

    def check_sparsity(self):
        self.conv.check_sparsity()

    def end_calibration(self):
        self.conv.end_calibration()

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        # if self.drop is not None:
        #     return self.drop(self.act(self.bn(self.conv(x))))
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies a fused convolution and activation function to the input tensor `x`."""
        return self.act(self.conv(x))


def lsqconv_from_conv(conv: Conv, mode: QuantizeMode) -> LSQConv:
    lsq_conv = _build_inner_conv(
        conv.conv.in_channels,
        conv.conv.out_channels,
        conv.conv.kernel_size,
        conv.conv.stride,
        conv.conv.padding,
        conv.conv.groups,
        conv.conv.dilation,
        mode=mode,
    )
    lsq_conv.state_dict()["weight"].copy_(conv.state_dict()["conv.weight"])
    lsq_bn = deepcopy(conv.bn)
    lsq_act = deepcopy(conv.act)
    return LSQConv(conv=lsq_conv, bn=lsq_bn, act=lsq_act)


class LSQBottleneck(nn.Module):
    def __init__(self, cv1, cv2, add: bool):
        super().__init__()
        self.cv1 = cv1
        self.cv2 = cv2
        self.add = add

    # Standard bottleneck
    @staticmethod
    def new(
        c1,
        c2,
        shortcut=True,
        g=1,
        k=(3, 3),
        e=0.5,
        num_bits=8,
        mode="exact",
    ):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        c_ = int(c2 * e)  # hidden channels
        cv1 = LSQConv.new(c1, c_, k[0], 1, num_bits=num_bits, mode=mode)
        cv2 = LSQConv.new(c_, c2, k[1], 1, g=g, num_bits=num_bits, mode=mode)
        add = shortcut and c1 == c2

        return LSQBottleneck(
            cv1=cv1,
            cv2=cv2,
            add=add,
        )

    def start_quantize(self):
        self.cv1.start_quantize()
        self.cv2.start_quantize()

    def start_prune(self, sparsity=None, prune_mode=None):
        self.cv1.start_prune(sparsity, prune_mode)
        self.cv2.start_prune(sparsity, prune_mode)

    def check_sparsity(self):
        self.cv1.check_sparsity()
        self.cv2.check_sparsity()

    def end_calibration(self):
        self.cv1.end_calibration()
        self.cv2.end_calibration()

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


def lsqbottleneck_from_bottleneck(
    bottleneck: Bottleneck, mode: QuantizeMode
) -> LSQBottleneck:
    cv1 = lsqconv_from_conv(bottleneck.cv1, mode=mode)
    cv2 = lsqconv_from_conv(bottleneck.cv2, mode=mode)
    return LSQBottleneck(cv1=cv1, cv2=cv2, add=bottleneck.add)


#
# C3k2 ------------------------------------------------
#


class LSQC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c, cv1, cv2, m):
        super().__init__()
        self.c = c
        self.cv1 = cv1
        self.cv2 = cv2
        self.m = m

    @staticmethod
    def new(c1, c2, n=1, shortcut=False, g=1, e=0.5, num_bits=8, mode="exact"):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        c = int(c2 * e)  # hidden channels
        cv1 = LSQConv.new(c1, 2 * c, 1, 1, num_bits=num_bits, mode=mode)
        cv2 = LSQConv.new(
            (2 + n) * c, c2, 1, num_bits=num_bits, mode=mode
        )  # optional act=FReLU(c2)
        m = nn.ModuleList(
            LSQBottleneck.new(
                c,
                c,
                shortcut,
                g,
                k=((3, 3), (3, 3)),
                e=1.0,
                num_bits=num_bits,
                mode=mode,
            )
            for _ in range(n)
        )

        return LSQC2f(c=c, cv1=cv1, cv2=cv2, m=m)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def start_quantize(self):
        self.cv1.start_quantize()
        self.cv2.start_quantize()
        for i in self.m:
            if isinstance(i, LSQBottleneck):
                i.start_quantize()

    def start_prune(self, sparsity=None, prune_mode=None):
        self.cv1.start_prune(sparsity, prune_mode)
        self.cv2.start_prune(sparsity, prune_mode)
        for i in self.m:
            if isinstance(i, LSQBottleneck):
                i.start_prune(sparsity, prune_mode)

    def check_sparsity(self):
        self.cv1.check_sparsity()
        self.cv2.check_sparsity()
        for i in self.m:
            if isinstance(i, LSQBottleneck):
                i.check_sparsity()

    def end_calibration(self):
        self.cv1.end_calibration()
        self.cv2.end_calibration()
        for i in self.m:
            if isinstance(i, LSQBottleneck):
                i.end_calibration()


def lsqc2f_from_c2f(c2f: C2f, mode: QuantizeMode) -> LSQC2f:
    c = c2f.c
    cv1 = lsqconv_from_conv(c2f.cv1, mode=mode)
    cv2 = lsqconv_from_conv(c2f.cv2, mode=mode)
    m = nn.ModuleList(
        lsqbottleneck_from_bottleneck(bottleneck, mode=mode)
        for bottleneck in c2f.m
    )
    return LSQC2f(c=c, cv1=cv1, cv2=cv2, m=m)


class LSQC3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, cv1, cv2, cv3, m):
        super().__init__()
        self.cv1 = cv1
        self.cv2 = cv2
        self.cv3 = cv3
        self.m = m

    @staticmethod
    def new(c1, c2, n=1, shortcut=True, g=1, e=0.5, num_bits=8, mode="exact"):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        c_ = int(c2 * e)  # hidden channels
        cv1 = LSQConv.new(c1, c_, 1, 1, num_bits=num_bits, mode=mode)
        cv2 = LSQConv.new(c1, c_, 1, 1, num_bits=num_bits, mode=mode)
        cv3 = LSQConv.new(
            2 * c_, c2, 1, num_bits=num_bits, mode=mode
        )  # optional act=FReLU(c2)
        m = nn.Sequential(
            *(
                LSQBottleneck.new(
                    c_,
                    c_,
                    shortcut,
                    g,
                    k=((1, 1), (3, 3)),
                    e=1.0,
                    num_bits=num_bits,
                    mode=mode,
                )
                for _ in range(n)
            )
        )

        return LSQC3(cv1=cv1, cv2=cv2, cv3=cv3, m=m)

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

    def start_quantize(self):
        self.cv1.start_quantize()
        self.cv2.start_quantize()
        self.cv3.start_quantize()
        for i in self.m:
            if isinstance(i, LSQBottleneck):
                i.start_quantize()

    def start_prune(self, sparsity=None, prune_mode=None):
        self.cv1.start_prune(sparsity, prune_mode)
        self.cv2.start_prune(sparsity, prune_mode)
        self.cv3.start_prune(sparsity, prune_mode)
        for i in self.m:
            if isinstance(i, LSQBottleneck):
                i.start_prune(sparsity, prune_mode)

    def check_sparsity(self):
        self.cv1.check_sparsity()
        self.cv2.check_sparsity()
        self.cv3.check_sparsity()
        for i in self.m:
            if isinstance(i, LSQBottleneck):
                i.check_sparsity()

    def end_calibration(self):
        self.cv1.end_calibration()
        self.cv2.end_calibration()
        self.cv3.end_calibration()
        for i in self.m:
            if isinstance(i, LSQBottleneck):
                i.end_calibration()


def lsqc3_from_c3(c3: C3, mode: QuantizeMode):
    cv1 = lsqconv_from_conv(c3.cv1, mode=mode)
    cv2 = lsqconv_from_conv(c3.cv2, mode=mode)
    cv3 = lsqconv_from_conv(c3.cv3, mode=mode)
    m = nn.Sequential(
        *(
            lsqbottleneck_from_bottleneck(bottleneck, mode=mode)
            for bottleneck in c3.m
        )
    )
    return LSQC3(cv1=cv1, cv2=cv2, cv3=cv3, m=m)


class LSQC3k(LSQC3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, cv1, cv2, cv3, m):
        super().__init__(cv1, cv2, cv3, m)

    @staticmethod
    def new(
        c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3, num_bits=8, mode="exact"
    ):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        c3 = super().new(
            c1, c2, n, shortcut, g, e, num_bits=num_bits, mode=mode
        )
        c_ = int(c2 * e)  # hidden channels
        m = nn.Sequential(
            *(
                LSQBottleneck.new(
                    c_,
                    c_,
                    shortcut,
                    g,
                    k=(k, k),
                    e=1.0,
                    num_bits=num_bits,
                    mode=mode,
                )
                for _ in range(n)
            )
        )
        return LSQC3k(cv1=c3.cv1, cv2=c3.cv2, cv3=c3.cv3, m=m)

    def start_quantize(self):
        super().start_quantize()

    def start_prune(self, sparsity=None, prune_mode=None):
        super().start_prune(sparsity, prune_mode)

    def check_sparsity(self):
        super().check_sparsity()

    def end_calibration(self):
        super().end_calibration()


def lsqc3k_from_c3k(c3k: C3k, mode: QuantizeMode) -> LSQC3k:
    cv1 = lsqconv_from_conv(c3k.cv1, mode=mode)
    cv2 = lsqconv_from_conv(c3k.cv2, mode=mode)
    cv3 = lsqconv_from_conv(c3k.cv3, mode=mode)
    m = nn.Sequential(
        *(
            lsqbottleneck_from_bottleneck(bottleneck, mode=mode)
            for bottleneck in c3k.m
        )
    )
    return LSQC3k(cv1=cv1, cv2=cv2, cv3=cv3, m=m)


class LSQC3k2(LSQC2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c, cv1, cv2, m):
        super().__init__(c, cv1, cv2, m)

    @staticmethod
    def new(
        c1,
        c2,
        n=1,
        c3k=False,
        e=0.5,
        g=1,
        shortcut=True,
        num_bits=8,
        mode="exact",
    ):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        c2k2 = super().new(
            c1, c2, n, shortcut, g, e, num_bits=num_bits, mode=mode
        )
        m = nn.ModuleList(
            (
                LSQC3k.new(
                    c2k2.c, c2k2.c, 2, shortcut, g, num_bits=num_bits, mode=mode
                )
                if c3k
                else LSQBottleneck.new(
                    c2k2.c, c2k2.c, shortcut, g, num_bits=num_bits, mode=mode
                )
            )
            for _ in range(n)
        )
        return LSQC3k2(c=c2k2.c, cv1=c2k2.cv1, cv2=c2k2.cv2, m=m)

    def start_quantize(self):
        self.cv1.start_quantize()
        self.cv2.start_quantize()
        for i in self.m:
            if isinstance(i, (LSQBottleneck, LSQC3k)):
                i.start_quantize()

    def start_prune(self, sparsity=None, prune_mode=None):
        self.cv1.start_prune(sparsity, prune_mode)
        self.cv2.start_prune(sparsity, prune_mode)
        for i in self.m:
            if isinstance(i, (LSQBottleneck, LSQC3k)):
                i.start_prune(sparsity, prune_mode)

    def check_sparsity(self):
        self.cv1.check_sparsity()
        self.cv2.check_sparsity()
        for i in self.m:
            if isinstance(i, (LSQBottleneck, LSQC3k)):
                i.check_sparsity()

    def end_calibration(self):
        self.cv1.end_calibration()
        self.cv2.end_calibration()
        for i in self.m:
            if isinstance(i, (LSQBottleneck, LSQC3k)):
                i.end_calibration()


def lsqc3k2_from_c3k2(c3k2: C3k2, mode: QuantizeMode) -> LSQC3k2:
    c = c3k2.c
    cv1 = lsqconv_from_conv(c3k2.cv1, mode=mode)
    cv2 = lsqconv_from_conv(c3k2.cv2, mode=mode)
    m = nn.ModuleList(
        (
            lsqc3k_from_c3k(module, mode=mode)
            if isinstance(module, C3k)
            else lsqbottleneck_from_bottleneck(module, mode=mode)
        )
        for module in c3k2.m
    )
    return LSQC3k2(c=c, cv1=cv1, cv2=cv2, m=m)


#
# LSQ SPPF
#


class LSQSPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, cv1, cv2, m):
        super().__init__()
        self.cv1 = cv1
        self.cv2 = cv2
        self.m = m

    @staticmethod
    def new(c1, c2, k=5, num_bits=8, mode="exact"):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        c_ = c1 // 2  # hidden channels
        cv1 = LSQConv.new(c1, c_, 1, 1, num_bits=num_bits, mode=mode)
        cv2 = LSQConv.new(c_ * 4, c2, 1, 1, num_bits=num_bits, mode=mode)
        m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        return LSQSPPF(cv1=cv1, cv2=cv2, m=m)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

    def start_quantize(self):
        self.cv1.start_quantize()
        self.cv2.start_quantize()

    def start_prune(self, sparsity=None, prune_mode=None):
        self.cv1.start_prune(sparsity, prune_mode)
        self.cv2.start_prune(sparsity, prune_mode)

    def check_sparsity(self):
        self.cv1.check_sparsity()
        self.cv2.check_sparsity()

    def end_calibration(self):
        self.cv1.end_calibration()
        self.cv2.end_calibration()


def lsqsppf_from_sppf(sppf: SPPF, mode: QuantizeMode) -> LSQSPPF:
    cv1 = lsqconv_from_conv(sppf.cv1, mode=mode)
    cv2 = lsqconv_from_conv(sppf.cv2, mode=mode)
    return LSQSPPF(cv1=cv1, cv2=cv2, m=deepcopy(sppf.m))


#
# LSQ C2PSA
#


class LSQAttention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, num_heads, head_dim, key_dim, scale, qkv, proj, pe):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.key_dim = key_dim
        self.scale = scale
        self.qkv = qkv
        self.proj = proj
        self.pe = pe

    @staticmethod
    def new(dim, num_heads=8, attn_ratio=0.5, num_bits=8, mode="exact"):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        head_dim = dim // num_heads
        key_dim = int(head_dim * attn_ratio)
        scale = key_dim**-0.5
        nh_kd = key_dim * num_heads
        h = dim + nh_kd * 2
        qkv = LSQConv.new(dim, h, 1, act=False, num_bits=num_bits, mode=mode)
        proj = LSQConv.new(dim, dim, 1, act=False, num_bits=num_bits, mode=mode)
        pe = LSQConv.new(
            dim, dim, 3, 1, g=dim, act=False, num_bits=num_bits, mode=mode
        )

        return LSQAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            key_dim=key_dim,
            scale=scale,
            qkv=qkv,
            proj=proj,
            pe=pe,
        )

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(
            B, self.num_heads, self.key_dim * 2 + self.head_dim, N
        ).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(
            v.reshape(B, C, H, W)
        )
        x = self.proj(x)
        return x

    def start_quantize(self):
        self.qkv.start_quantize()
        self.proj.start_quantize()
        self.pe.start_quantize()

    def start_prune(self, sparsity=None, prune_mode=None):
        self.qkv.start_prune(sparsity, prune_mode)
        self.proj.start_prune(sparsity, prune_mode)
        self.pe.start_prune(sparsity, prune_mode)

    def check_sparsity(self):
        self.qkv.check_sparsity()
        self.proj.check_sparsity()
        self.pe.check_sparsity()

    def end_calibration(self):
        self.qkv.end_calibration()
        self.proj.end_calibration()
        self.pe.end_calibration()


def lsqattention_from_attention(
    attention: Attention, mode: QuantizeMode
) -> LSQAttention:
    qkv = lsqconv_from_conv(attention.qkv, mode=mode)
    proj = lsqconv_from_conv(attention.proj, mode=mode)
    pe = lsqconv_from_conv(attention.pe, mode=mode)
    return LSQAttention(
        num_heads=attention.num_heads,
        head_dim=attention.head_dim,
        key_dim=attention.key_dim,
        scale=attention.scale,
        qkv=qkv,
        proj=proj,
        pe=pe,
    )


class LSQPSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, attn, ffn, add):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.add = add

    @staticmethod
    def new(
        c, attn_ratio=0.5, num_heads=4, shortcut=True, num_bits=8, mode="exact"
    ):
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        attn = LSQAttention.new(
            c,
            attn_ratio=attn_ratio,
            num_heads=num_heads,
            num_bits=num_bits,
            mode=mode,
        )
        ffn = nn.Sequential(
            LSQConv.new(c, c * 2, 1, num_bits=num_bits, mode=mode),
            LSQConv.new(c * 2, c, 1, act=False, num_bits=num_bits, mode=mode),
        )
        add = shortcut
        return LSQPSABlock(attn=attn, ffn=ffn, add=add)

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

    def start_quantize(self):
        self.attn.start_quantize()
        assert isinstance(self.ffn[0], LSQConv)
        self.ffn[0].start_quantize()
        assert isinstance(self.ffn[1], LSQConv)
        self.ffn[1].start_quantize()

    def start_prune(self, sparsity=None, prune_mode=None):
        self.attn.start_prune(sparsity, prune_mode)
        self.ffn[0].start_prune(sparsity, prune_mode)
        self.ffn[1].start_prune(sparsity, prune_mode)

    def check_sparsity(self):
        self.attn.check_sparsity()
        self.ffn[0].check_sparsity()
        self.ffn[1].check_sparsity()

    def end_calibration(self):
        self.attn.end_calibration()
        self.ffn[0].end_calibration()
        self.ffn[1].end_calibration()


def lsqpsablock_from_psablock(
    psablock: PSABlock, mode: QuantizeMode
) -> LSQPSABlock:
    attn = lsqattention_from_attention(psablock.attn, mode=mode)
    ffn = nn.Sequential(
        lsqconv_from_conv(psablock.ffn[0], mode=mode),
        lsqconv_from_conv(psablock.ffn[1], mode=mode),
    )
    return LSQPSABlock(attn=attn, ffn=ffn, add=psablock.add)


class LSQC2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c, cv1, cv2, m):
        super().__init__()
        self.c = c
        self.cv1 = cv1
        self.cv2 = cv2
        self.m = m

    @staticmethod
    def new(c1, c2, n=1, e=0.5, num_bits=8, mode="exact"):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        assert c1 == c2
        c = int(c1 * e)
        cv1 = LSQConv.new(c1, 2 * c, 1, 1, num_bits=num_bits, mode=mode)
        cv2 = LSQConv.new(2 * c, c1, 1, num_bits=num_bits, mode=mode)

        m = nn.Sequential(
            *(
                LSQPSABlock.new(
                    c,
                    attn_ratio=0.5,
                    num_heads=c // 64,
                    num_bits=num_bits,
                    mode=mode,
                )
                for _ in range(n)
            )
        )

        return LSQC2PSA(c=c, cv1=cv1, cv2=cv2, m=m)

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

    def start_quantize(self):
        self.cv1.start_quantize()
        self.cv2.start_quantize()
        for i in self.m:
            if isinstance(i, LSQPSABlock):
                i.start_quantize()

    def start_prune(self, sparsity=None, prune_mode=None):
        self.cv1.start_prune(sparsity, prune_mode)
        self.cv2.start_prune(sparsity, prune_mode)
        for i in self.m:
            if isinstance(i, LSQPSABlock):
                i.start_prune(sparsity, prune_mode)

    def check_sparsity(self):
        self.cv1.check_sparsity()
        self.cv2.check_sparsity()
        for i in self.m:
            if isinstance(i, LSQPSABlock):
                i.check_sparsity()

    def end_calibration(self):
        self.cv1.end_calibration()
        self.cv2.end_calibration()
        for i in self.m:
            if isinstance(i, LSQPSABlock):
                i.end_calibration()


def lsqc2psa_from_c2psa(c2psa: C2PSA, mode: QuantizeMode) -> LSQC2PSA:
    c = c2psa.c
    cv1 = lsqconv_from_conv(c2psa.cv1, mode=mode)
    cv2 = lsqconv_from_conv(c2psa.cv2, mode=mode)
    m = nn.Sequential(
        *(
            lsqpsablock_from_psablock(psablock, mode=mode)
            for psablock in c2psa.m
        )
    )
    return LSQC2PSA(c=c, cv1=cv1, cv2=cv2, m=m)
