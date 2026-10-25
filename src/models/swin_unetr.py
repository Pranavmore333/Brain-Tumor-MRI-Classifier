"""
Swin UNETR: 3D Swin Transformer for Volumetric Brain Tumor Segmentation & Classification.
Blueprint Specifications:
- 3D Patch Partition (2x2x2)
- Hierarchical 3D Swin Transformer Encoder with Window Multi-Head Self-Attention (W-MSA)
  and Shifted Window Multi-Head Self-Attention (SW-MSA)
- 3D Relative Position Bias B in R^{M^3 x M^3}
- 5 multi-scale feature resolution levels
- 3D CNN Residual Decoder with skip connections
- Joint Multi-Task Head: 3-channel nested segmentation (ET, TC, WT) + 3-class disease classifier
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any


def window_partition_3d(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Partitions 3D feature tensor into non-overlapping local 3D windows.
    Args:
        x: (B, D, H, W, C)
        window_size: M
    Returns:
        windows: (num_windows * B, M, M, M, C)
    """
    B, D, H, W, C = x.shape
    M = window_size
    pad_d = (M - D % M) % M
    pad_h = (M - H % M) % M
    pad_w = (M - W % M) % M

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))

    _, Dp, Hp, Wp, _ = x.shape
    num_d, num_h, num_w = Dp // M, Hp // M, Wp // M

    x = x.view(B, num_d, M, num_h, M, num_w, M, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, M, M, M, C)
    return windows, (num_d, num_h, num_w)


def window_reverse_3d(
    windows: torch.Tensor,
    window_size: int,
    grid_size: Tuple[int, int, int],
    orig_shape: Tuple[int, int, int, int, int]
) -> torch.Tensor:
    """
    Reverses window partitioning back into (B, D, H, W, C) tensor.
    """
    B, D, H, W, C = orig_shape
    M = window_size
    num_d, num_h, num_w = grid_size
    
    x = windows.view(B, num_d, num_h, num_w, M, M, M, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, num_d * M, num_h * M, num_w * M, C)
    return x[:, :D, :H, :W, :].contiguous()


class WindowAttention3D(nn.Module):
    """
    3D Local Window Multi-Head Self-Attention with 3D Relative Position Bias.
    Formula: Attention(Q, K, V) = Softmax(QK^T / sqrt(d_k) + B) * V
    """

    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 3D relative position bias table: (2M - 1)^3
        M = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * M - 1) * (2 * M - 1) * (2 * M - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Coordinate indexing grid
        coords_d = torch.arange(M)
        coords_h = torch.arange(M)
        coords_w = torch.arange(M)
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij")) # (3, M, M, M)
        coords_flatten = torch.flatten(coords, 1) # (3, M^3)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # (3, M^3, M^3)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += M - 1
        relative_coords[:, :, 1] += M - 1
        relative_coords[:, :, 2] += M - 1
        relative_coords[:, :, 0] *= (2 * M - 1) * (2 * M - 1)
        relative_coords[:, :, 1] *= (2 * M - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (num_windows * B, M^3, C)
        """
        B_w, N, C = x.shape
        qkv = self.qkv(x).reshape(B_w, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Relative position bias
        M3 = self.window_size ** 3
        rel_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(M3, M3, -1).permute(2, 0, 1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_w, N, C)
        return self.proj(out)


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block alternating between W-MSA and SW-MSA."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 7, shift_size: int = 0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, H, W, C)
        """
        B, D, H, W, C = x.shape
        shortcut = x
        x_norm = self.norm1(x)

        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x_norm,
                shifts=(-self.shift_size, -self.shift_size, -self.shift_size),
                dims=(1, 2, 3)
            )
        else:
            shifted_x = x_norm

        # Partition windows
        windows, grid_size = window_partition_3d(shifted_x, self.window_size)
        windows = windows.view(-1, self.window_size ** 3, C)

        attn_windows = self.attn(windows)

        # Reverse windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse_3d(attn_windows, self.window_size, grid_size, (B, D, H, W, C))

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size, self.shift_size),
                dims=(1, 2, 3)
            )
        else:
            x = shifted_x

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualConvBlock3D(nn.Module):
    """3D CNN Residual Block used in Decoder and Skip-projection connections."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.skip(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + res)


class SwinUNETR(nn.Module):
    """
    Complete Multi-Task 3D Swin UNETR Architecture:
    - 4-channel input (T1, T1ce, T2, FLAIR)
    - 5 hierarchical resolution levels
    - Residual 3D CNN decoder with skip connections
    - Multi-label segmentation output (3 channels: ET, TC, WT)
    - Global multi-class classification head (3 classes: Glioma II/III, GBM IV, Meningioma)
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_size: int = 48,
        num_classes: int = 3,
        window_size: int = 7,
        patch_size: int = 2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.num_classes = num_classes

        # Patch embedding: projects non-overlapping 2x2x2 patches to feature dimension
        self.patch_embed = nn.Conv3d(
            in_channels, feature_size,
            kernel_size=patch_size, stride=patch_size
        )
        self.norm0 = nn.LayerNorm(feature_size)

        # Level 1: Resolution 1/2
        self.stage1 = nn.Sequential(
            SwinTransformerBlock3D(feature_size, num_heads=3, window_size=window_size, shift_size=0),
            SwinTransformerBlock3D(feature_size, num_heads=3, window_size=window_size, shift_size=window_size // 2)
        )
        # Downsample to 1/4
        self.down1 = nn.Conv3d(feature_size, feature_size * 2, kernel_size=2, stride=2)

        # Level 2: Resolution 1/4
        self.stage2 = nn.Sequential(
            SwinTransformerBlock3D(feature_size * 2, num_heads=6, window_size=window_size, shift_size=0),
            SwinTransformerBlock3D(feature_size * 2, num_heads=6, window_size=window_size, shift_size=window_size // 2)
        )
        # Downsample to 1/8
        self.down2 = nn.Conv3d(feature_size * 2, feature_size * 4, kernel_size=2, stride=2)

        # Level 3: Resolution 1/8
        self.stage3 = nn.Sequential(
            SwinTransformerBlock3D(feature_size * 4, num_heads=12, window_size=window_size, shift_size=0),
            SwinTransformerBlock3D(feature_size * 4, num_heads=12, window_size=window_size, shift_size=window_size // 2)
        )
        # Downsample to 1/16 (Bottleneck)
        self.down3 = nn.Conv3d(feature_size * 4, feature_size * 8, kernel_size=2, stride=2)

        # Bottleneck: Resolution 1/16
        self.bottleneck = nn.Sequential(
            SwinTransformerBlock3D(feature_size * 8, num_heads=24, window_size=window_size, shift_size=0),
            SwinTransformerBlock3D(feature_size * 8, num_heads=24, window_size=window_size, shift_size=window_size // 2)
        )

        # Classification Head (from Bottleneck latent features)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(feature_size * 8, feature_size * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feature_size * 2, num_classes)
        )

        # Decoder: Transpose convs + Residual blocks
        self.up3 = nn.ConvTranspose3d(feature_size * 8, feature_size * 4, kernel_size=2, stride=2)
        self.dec_block3 = ResidualConvBlock3D(feature_size * 8, feature_size * 4)

        self.up2 = nn.ConvTranspose3d(feature_size * 4, feature_size * 2, kernel_size=2, stride=2)
        self.dec_block2 = ResidualConvBlock3D(feature_size * 4, feature_size * 2)

        self.up1 = nn.ConvTranspose3d(feature_size * 2, feature_size, kernel_size=2, stride=2)
        self.dec_block1 = ResidualConvBlock3D(feature_size * 2, feature_size)

        self.up0 = nn.ConvTranspose3d(feature_size, feature_size, kernel_size=2, stride=2)
        self.dec_block0 = ResidualConvBlock3D(feature_size + in_channels, feature_size)

        # Final Segmentation Head (ET, TC, WT)
        self.seg_head = nn.Conv3d(feature_size, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 4, H, W, D) multi-channel volume
        Returns:
            seg_logits: (B, 3, H, W, D) segmentation logits for ET, TC, WT
            cls_logits: (B, 3) classification logits
        """
        # Patch embedding
        p0 = self.patch_embed(x) # (B, C, H/2, W/2, D/2)
        p0_perm = p0.permute(0, 2, 3, 4, 1) # (B, D, H, W, C)
        p0_norm = self.norm0(p0_perm)

        # Level 1
        s1 = self.stage1(p0_norm) # (B, D, H, W, C)
        s1_t = s1.permute(0, 4, 1, 2, 3) # (B, C, H, W, D)

        # Level 2
        d1 = self.down1(s1_t).permute(0, 2, 3, 4, 1)
        s2 = self.stage2(d1)
        s2_t = s2.permute(0, 4, 1, 2, 3)

        # Level 3
        d2 = self.down2(s2_t).permute(0, 2, 3, 4, 1)
        s3 = self.stage3(d2)
        s3_t = s3.permute(0, 4, 1, 2, 3)

        # Bottleneck (1/16)
        d3 = self.down3(s3_t).permute(0, 2, 3, 4, 1)
        bn = self.bottleneck(d3)
        bn_t = bn.permute(0, 4, 1, 2, 3)

        # Classification Logits
        pooled = self.global_pool(bn_t).flatten(1)
        cls_logits = self.classifier(pooled)

        # Decoder path with skip connections
        u3 = self.up3(bn_t)
        u3 = torch.cat([u3, s3_t], dim=1)
        d_out3 = self.dec_block3(u3)

        u2 = self.up2(d_out3)
        u2 = torch.cat([u2, s2_t], dim=1)
        d_out2 = self.dec_block2(u2)

        u1 = self.up1(d_out2)
        u1 = torch.cat([u1, s1_t], dim=1)
        d_out1 = self.dec_block1(u1)

        u0 = self.up0(d_out1)
        u0 = torch.cat([u0, x], dim=1) # skip to input volume
        d_out0 = self.dec_block0(u0)

        seg_logits = self.seg_head(d_out0)
        return seg_logits, cls_logits
