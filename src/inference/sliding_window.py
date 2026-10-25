"""
Volumetric Patch-Based Sliding Window Inference Engine.
Blueprint Specifications:
- 128x128x128 ROI Patches with 50% overlap ratio (sigma = 0.5)
- Spatial 3D Gaussian Weighting Matrix W_Gauss
- Boundary artifact suppression and seam-free blending:
  P_final(x) = sum(W_gauss * Y_hat) / sum(W_gauss)
- Memory-efficient batching preventing GPU VRAM exhaustion
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, List


def create_3d_gaussian_weight_matrix(
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    sigma_scale: float = 0.125
) -> np.ndarray:
    """
    Computes a 3D Gaussian kernel weighting window W_Gauss.
    Values peak at 1.0 at the center of the patch and decay toward the boundaries.
    """
    H, W, D = patch_size
    sigmas = [H * sigma_scale, W * sigma_scale, D * sigma_scale]

    x = np.arange(H) - (H - 1) / 2.0
    y = np.arange(W) - (W - 1) / 2.0
    z = np.arange(D) - (D - 1) / 2.0

    gx = np.exp(-0.5 * (x / sigmas[0]) ** 2)
    gy = np.exp(-0.5 * (y / sigmas[1]) ** 2)
    gz = np.exp(-0.5 * (z / sigmas[2]) ** 2)

    # Outer product across 3 dimensions
    weight = gx[:, None, None] * gy[None, :, None] * gz[None, None, :]
    # Normalize peak to 1.0 and clip minimum to preserve numerical stability
    if np.max(weight) > 0:
        weight = weight / np.max(weight)
    weight = np.clip(weight, 1e-4, 1.0).astype(np.float32)
    return weight


class SlidingWindowInferer:
    """
    Executes patch-based sliding window inference on whole-brain 3D volumes.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        overlap: float = 0.5,
        device: str = "cpu"
    ):
        self.patch_size = patch_size
        self.overlap = overlap
        self.device = device
        self.gaussian_weight_np = create_3d_gaussian_weight_matrix(patch_size)
        self.gaussian_weight = torch.from_numpy(self.gaussian_weight_np).to(device)

    def get_patch_coordinates(
        self,
        volume_shape: Tuple[int, int, int]
    ) -> List[Tuple[slice, slice, slice]]:
        """
        Calculates slice indices for all sliding window patches covering the volume.
        """
        slices = []
        H, W, D = volume_shape
        pH, pW, pD = self.patch_size
        sH = max(1, int(pH * (1.0 - self.overlap)))
        sW = max(1, int(pW * (1.0 - self.overlap)))
        sD = max(1, int(pD * (1.0 - self.overlap)))

        def generate_1d_starts(dim_len: int, patch_len: int, stride_len: int) -> List[int]:
            if dim_len <= patch_len:
                return [0]
            starts = list(range(0, dim_len - patch_len + 1, stride_len))
            if starts[-1] + patch_len < dim_len:
                starts.append(dim_len - patch_len)
            return starts

        h_starts = generate_1d_starts(H, pH, sH)
        w_starts = generate_1d_starts(W, pW, sW)
        d_starts = generate_1d_starts(D, pD, sD)

        for h in h_starts:
            for w in w_starts:
                for d in d_starts:
                    slices.append((
                        slice(h, h + pH),
                        slice(w, w + pW),
                        slice(d, d + pD)
                    ))
        return slices

    @torch.no_grad()
    def infer(
        self,
        model: nn.Module,
        volume: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes sliding window inference across the full input tensor.
        Args:
            model: PyTorch model returning (seg_logits, cls_logits)
            volume: (B, C, H, W, D) tensor on target device
        Returns:
            final_seg_probs: (B, 3, H, W, D) blended multi-label probabilities
            avg_cls_logits: (B, 3) aggregated classification logits
        """
        model.eval()
        B, C, H, W, D = volume.shape
        out_channels = 3 # ET, TC, WT
        pH, pW, pD = self.patch_size

        # Accumulator buffers for weighted predictions and weights sum
        output_buffer = torch.zeros((B, out_channels, H, W, D), device=self.device, dtype=torch.float32)
        weight_sum = torch.zeros((1, 1, H, W, D), device=self.device, dtype=torch.float32)

        cls_logits_list = []
        patch_coords = self.get_patch_coordinates((H, W, D))

        for sl_h, sl_w, sl_d in patch_coords:
            patch = volume[:, :, sl_h, sl_w, sl_d]
            seg_logits, cls_logits = model(patch)
            seg_probs = torch.sigmoid(seg_logits)

            # Apply Gaussian spatial weighting
            weighted_patch = seg_probs * self.gaussian_weight

            output_buffer[:, :, sl_h, sl_w, sl_d] += weighted_patch
            weight_sum[:, :, sl_h, sl_w, sl_d] += self.gaussian_weight

            cls_logits_list.append(cls_logits)

        # Normalize by total accumulated Gaussian weights: P_final(x) = sum(W * Y) / sum(W)
        final_seg_probs = output_buffer / torch.clamp(weight_sum, min=1e-5)

        # Average classification logits across all sampled patches
        avg_cls_logits = torch.stack(cls_logits_list, dim=0).mean(dim=0)

        return final_seg_probs, avg_cls_logits
