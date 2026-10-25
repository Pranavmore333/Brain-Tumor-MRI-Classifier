"""
MedNeXt: 3D ConvNeXt with Depthwise Separable 7x7x7 Convolutions.
Blueprint Specifications:
- Pure 3D ConvNeXt design
- 7x7x7 Depthwise Separable 3D Convolutions
- Inverted Bottleneck (1x1x1 Conv expansion ratio = 4)
- GELU Non-linearity + GroupNorm / LayerScale
- Hierarchical Encoder-Decoder with residual skip connections
- Multi-Task Head: ET/TC/WT Segmentation + 3-Class Disease Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class MedNeXtBlock3D(nn.Module):
    """
    Core MedNeXt Block featuring 7x7x7 Depthwise Convolution and Inverted Bottleneck.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        expansion_ratio: int = 4,
        layer_scale_init_value: float = 1e-6
    ):
        super().__init__()
        padding = kernel_size // 2

        # 1. Depthwise 3D Convolution with large kernel
        self.dwconv = nn.Conv3d(
            dim, dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
            bias=True
        )

        # 2. Normalization
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)

        # 3. Inverted Bottleneck: 1x1x1 Conv -> GELU -> 1x1x1 Conv
        hidden_dim = dim * expansion_ratio
        self.pwconv1 = nn.Conv3d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(hidden_dim, dim, kernel_size=1)

        # 4. LayerScale
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim, 1, 1, 1)),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)

        if self.gamma is not None:
            out = self.gamma * out

        return shortcut + out


class MedNeXtDownsample3D(nn.Module):
    """Spatial downsampling by factor of 2 with 2x2x2 stride."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_dim)
        self.conv = nn.Conv3d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class MedNeXtUpsample3D(nn.Module):
    """Spatial upsampling by factor of 2 with transposed convolution."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_dim)
        self.conv = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class MedNeXt(nn.Module):
    """
    Complete Multi-Task 3D MedNeXt Architecture:
    - 4-channel input (T1, T1ce, T2, FLAIR)
    - 4 hierarchical stages with 7x7x7 depthwise convolutions
    - Residual decoder with skip connections
    - Multi-label segmentation output (ET, TC, WT)
    - Global multi-class classification head
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 32,
        num_classes: int = 3,
        kernel_size: int = 7,
        expansion_ratio: int = 4,
        blocks_per_stage: List[int] = [2, 2, 2, 2]
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_classes = num_classes

        dims = [base_channels * (2 ** i) for i in range(4)] # e.g. [32, 64, 128, 256]

        # Stem: maps input volume to base_channels
        self.stem = nn.Conv3d(in_channels, dims[0], kernel_size=1)

        # Stage 0
        self.stage0 = nn.Sequential(*[
            MedNeXtBlock3D(dims[0], kernel_size, expansion_ratio)
            for _ in range(blocks_per_stage[0])
        ])

        # Stage 1
        self.down0 = MedNeXtDownsample3D(dims[0], dims[1])
        self.stage1 = nn.Sequential(*[
            MedNeXtBlock3D(dims[1], kernel_size, expansion_ratio)
            for _ in range(blocks_per_stage[1])
        ])

        # Stage 2
        self.down1 = MedNeXtDownsample3D(dims[1], dims[2])
        self.stage2 = nn.Sequential(*[
            MedNeXtBlock3D(dims[2], kernel_size, expansion_ratio)
            for _ in range(blocks_per_stage[2])
        ])

        # Stage 3 (Bottleneck)
        self.down2 = MedNeXtDownsample3D(dims[2], dims[3])
        self.stage3 = nn.Sequential(*[
            MedNeXtBlock3D(dims[3], kernel_size, expansion_ratio)
            for _ in range(blocks_per_stage[3])
        ])

        # Global Classification Head
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(dims[3], dims[1]),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dims[1], num_classes)
        )

        # Decoder Path
        self.up2 = MedNeXtUpsample3D(dims[3], dims[2])
        self.dec_conv2 = nn.Conv3d(dims[2] * 2, dims[2], kernel_size=1)
        self.dec_stage2 = nn.Sequential(*[
            MedNeXtBlock3D(dims[2], kernel_size, expansion_ratio)
            for _ in range(blocks_per_stage[2])
        ])

        self.up1 = MedNeXtUpsample3D(dims[2], dims[1])
        self.dec_conv1 = nn.Conv3d(dims[1] * 2, dims[1], kernel_size=1)
        self.dec_stage1 = nn.Sequential(*[
            MedNeXtBlock3D(dims[1], kernel_size, expansion_ratio)
            for _ in range(blocks_per_stage[1])
        ])

        self.up0 = MedNeXtUpsample3D(dims[1], dims[0])
        self.dec_conv0 = nn.Conv3d(dims[0] * 2, dims[0], kernel_size=1)
        self.dec_stage0 = nn.Sequential(*[
            MedNeXtBlock3D(dims[0], kernel_size, expansion_ratio)
            for _ in range(blocks_per_stage[0])
        ])

        # Output Segmentation Head
        self.seg_head = nn.Conv3d(dims[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 4, H, W, D) multi-parametric volume
        Returns:
            seg_logits: (B, 3, H, W, D) segmentation logits for ET, TC, WT
            cls_logits: (B, 3) classification logits
        """
        # Encoder
        x0 = self.stem(x)
        s0 = self.stage0(x0)

        x1 = self.down0(s0)
        s1 = self.stage1(x1)

        x2 = self.down1(s1)
        s2 = self.stage2(x2)

        x3 = self.down2(s2)
        s3 = self.stage3(x3)

        # Classification from bottleneck
        pooled = self.global_pool(s3).flatten(1)
        cls_logits = self.classifier(pooled)

        # Decoder with skip connections
        u2 = self.up2(s3)
        u2 = self.dec_conv2(torch.cat([u2, s2], dim=1))
        d2 = self.dec_stage2(u2)

        u1 = self.up1(d2)
        u1 = self.dec_conv1(torch.cat([u1, s1], dim=1))
        d1 = self.dec_stage1(u1)

        u0 = self.up0(d1)
        u0 = self.dec_conv0(torch.cat([u0, s0], dim=1))
        d0 = self.dec_stage0(u0)

        seg_logits = self.seg_head(d0)
        return seg_logits, cls_logits
