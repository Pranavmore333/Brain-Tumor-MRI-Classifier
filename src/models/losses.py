"""
Clinical-Grade Objective Functions for Severe Class Imbalance.
Implements:
1. Multi-label Soft Dice Loss evaluated across K overlapping regions (ET, TC, WT)
2. Weighted Binary Cross-Entropy
3. Categorical Cross-Entropy for Global Disease Classification
4. Multi-Task Composite Loss: L_composite = L_dice + lambda_ce * L_ce + lambda_cls * L_cls
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class MultiLabelSoftDiceLoss(nn.Module):
    """
    Multi-label Soft Dice Loss across K overlapping clinical sub-regions.
    Formula:
        L_Dice = 1 - (2 / K) * sum_{k=1}^K [ (sum(p_k * g_k) + eps) / (sum(p_k^2) + sum(g_k^2) + eps) ]
    where eps = 1e-5 preserves numerical stability.
    """

    def __init__(self, smooth: float = 1e-5, p_squared: bool = True):
        super().__init__()
        self.smooth = smooth
        self.p_squared = p_squared

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: Tensor of shape (B, K, H, W, D) - unnormalized network logits
            target: Ground truth binary tensor of shape (B, K, H, W, D) in {0, 1}
        Returns:
            Scalar Dice loss value
        """
        probs = torch.sigmoid(pred_logits)
        K = probs.shape[1]

        # Flatten spatial dimensions: (B, K, V)
        probs_flat = probs.view(probs.shape[0], K, -1)
        target_flat = target.view(target.shape[0], K, -1).float()

        intersection = torch.sum(probs_flat * target_flat, dim=-1)

        if self.p_squared:
            denominator = torch.sum(probs_flat ** 2, dim=-1) + torch.sum(target_flat ** 2, dim=-1)
        else:
            denominator = torch.sum(probs_flat, dim=-1) + torch.sum(target_flat, dim=-1)

        dice_per_channel = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        # Average across channels and batch
        loss = 1.0 - torch.mean(dice_per_channel)
        return loss


class MultiTaskCompositeLoss(nn.Module):
    """
    Combines multi-label Soft Dice loss, weighted Binary Cross Entropy (BCE) for
    segmentation, and Categorical Cross Entropy for global disease classification.
    """

    def __init__(
        self,
        lambda_ce: float = 1.0,
        lambda_cls: float = 0.5,
        dice_smooth: float = 1e-5
    ):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_cls = lambda_cls
        self.dice_loss = MultiLabelSoftDiceLoss(smooth=dice_smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_cls_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        seg_logits: torch.Tensor,
        seg_targets: torch.Tensor,
        cls_logits: Optional[torch.Tensor] = None,
        cls_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes composite loss and individual components for monitoring.
        """
        l_dice = self.dice_loss(seg_logits, seg_targets)
        l_bce = self.bce_loss(seg_logits, seg_targets.float())
        l_seg = l_dice + self.lambda_ce * l_bce

        total_loss = l_seg
        l_cls = torch.tensor(0.0, device=seg_logits.device)

        if cls_logits is not None and cls_targets is not None:
            l_cls = self.ce_cls_loss(cls_logits, cls_targets)
            total_loss = total_loss + self.lambda_cls * l_cls

        return {
            "total_loss": total_loss,
            "dice_loss": l_dice,
            "bce_loss": l_bce,
            "cls_loss": l_cls
        }
