"""
Active Learning & Uncertainty Estimation Engine.
Blueprint Specifications:
- Epistemic Uncertainty (Model Uncertainty):
  M=10 stochastic Monte Carlo Dropout passes:
  Var_epistemic(x) = (1 / M) * sum_{m=1}^M (p_m(x) - \bar{p}(x))^2
- Aleatoric Uncertainty (Data Noise):
  Test-Time Augmentation (TTA) stability across 3D spatial transformations
- Active Learning Prioritization Queue for Human-in-the-Loop Clinician Refinement
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List, Optional
from src.inference.sliding_window import SlidingWindowInferer


def enable_dropout(model: nn.Module) -> None:
    """Activates dropout layers during inference for Monte Carlo sampling."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


class EpistemicUncertaintyEstimator:
    """
    Computes voxel-wise epistemic uncertainty using Monte Carlo Dropout forward passes.
    """

    def __init__(
        self,
        num_mc_samples: int = 10,
        inferer: Optional[SlidingWindowInferer] = None
    ):
        self.num_mc_samples = num_mc_samples
        self.inferer = inferer or SlidingWindowInferer()

    @torch.no_grad()
    def estimate_uncertainty(
        self,
        model: nn.Module,
        volume: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Executes M=10 stochastic passes with active dropout.
        Returns:
            mean_probs: (B, 3, H, W, D) ensemble mean prediction
            voxel_variance: (B, 3, H, W, D) epistemic variance map
            volume_uncertainty_score: float, scalar ranking score for active learning
        """
        # Enable dropout while keeping batchnorm/instancenorm in eval mode
        model.eval()
        enable_dropout(model)

        mc_predictions = []
        mc_cls_logits = []

        for _ in range(self.num_mc_samples):
            probs, cls_logits = self.inferer.infer(model, volume)
            mc_predictions.append(probs)
            mc_cls_logits.append(cls_logits)

        # Stack predictions: (M, B, 3, H, W, D)
        mc_stack = torch.stack(mc_predictions, dim=0)

        # Mean probability: \bar{p}(x)
        mean_probs = torch.mean(mc_stack, dim=0)

        # Voxel-wise variance: Var(x) = (1/M) * sum((p_m(x) - \bar{p}(x))^2)
        voxel_variance = torch.var(mc_stack, dim=0, unbiased=False)

        # Average classification logits
        avg_cls_logits = torch.stack(mc_cls_logits, dim=0).mean(dim=0)

        # Compute volume-level uncertainty score (mean variance in regions where tumor is predicted)
        # Using Whole Tumor channel (index 2)
        wt_mask = mean_probs[:, 2:3] > 0.1
        if torch.any(wt_mask):
            volume_score = float(torch.mean(voxel_variance[:, 2:3][wt_mask]).cpu().item())
        else:
            volume_score = float(torch.max(voxel_variance).cpu().item())

        return {
            "mean_probs": mean_probs,
            "voxel_variance": voxel_variance,
            "volume_uncertainty_score": volume_score,
            "cls_logits": avg_cls_logits,
            "mc_samples": self.num_mc_samples
        }


class TestTimeAugmentationEstimator:
    """
    Computes Aleatoric Uncertainty via Test-Time Augmentation (TTA).
    Evaluates prediction stability under 3D flips (axial, sagittal, coronal).
    """

    def __init__(self, inferer: Optional[SlidingWindowInferer] = None):
        self.inferer = inferer or SlidingWindowInferer()

    @torch.no_grad()
    def evaluate_tta(
        self,
        model: nn.Module,
        volume: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Applies identity, flip-X, flip-Y, flip-Z, and inverts predictions back to native space.
        """
        model.eval()
        transformations = [
            ("identity", None),
            ("flip_x", 2),
            ("flip_y", 3),
            ("flip_z", 4)
        ]

        predictions = []

        for name, flip_dim in transformations:
            transformed_vol = volume.clone()
            if flip_dim is not None:
                transformed_vol = torch.flip(transformed_vol, dims=(flip_dim,))

            probs, _ = self.inferer.infer(model, transformed_vol)

            if flip_dim is not None:
                probs = torch.flip(probs, dims=(flip_dim,))

            predictions.append(probs)

        tta_stack = torch.stack(predictions, dim=0)
        mean_tta_probs = torch.mean(tta_stack, dim=0)
        aleatoric_variance = torch.var(tta_stack, dim=0, unbiased=False)

        return {
            "mean_tta_probs": mean_tta_probs,
            "aleatoric_variance": aleatoric_variance,
            "mean_aleatoric_score": float(torch.mean(aleatoric_variance).cpu().item())
        }
