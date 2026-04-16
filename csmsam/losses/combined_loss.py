"""
Combined loss for CSM-SAM training.

L_total = L_dice(pred_mask, mid_mask)
        + λ_ce * L_bce(pred_mask, mid_mask)
        + λ_change * L_ce(pred_change, change_label)

Defaults: λ_ce = 1.0, λ_change = 0.3
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from csmsam.modeling.change_head import ChangeMapLoss, build_change_labels


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    L_dice = 1 - (2 * |P ∩ G| + ε) / (|P| + |G| + ε)
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W) — raw predictions (before sigmoid)
            targets : (B, 1, H, W) — binary masks {0, 1}
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.shape[0], -1)
        targets_flat = targets.reshape(targets.shape[0], -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in sparse tumor masks."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class CombinedLoss(nn.Module):
    """
    CSM-SAM training loss.

    Components:
        1. Dice loss      — primary segmentation (λ = 1.0)
        2. BCE loss       — pixel-wise calibration (λ = 1.0)
        3. Change map CE  — auxiliary change supervision (λ = 0.3)

    The change map loss is optional (only computed when change_logits is provided).
    """

    def __init__(
        self,
        lambda_dice: float = 1.0,
        lambda_bce: float = 1.0,
        lambda_change: float = 0.3,
        change_loss_weights: list[float] | None = None,
    ):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.lambda_change = lambda_change

        self.dice_loss = DiceLoss(smooth=1.0)
        self.change_loss = ChangeMapLoss(weight=change_loss_weights)

    def forward(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        change_logits: torch.Tensor | None = None,
        pre_masks: torch.Tensor | None = None,
        mid_masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred_masks   : (B, 1, H, W) — segmentation logits
            target_masks : (B, 1, H, W) — binary mid-RT masks
            change_logits: (B, 4, H, W) — change map logits (optional)
            pre_masks    : (B, 1, H, W) — pre-RT masks (for change label generation)
            mid_masks    : (B, 1, H, W) — mid-RT masks (same as target_masks usually)

        Returns:
            dict with 'total', 'dice', 'bce', 'change' loss tensors
        """
        # Segmentation losses
        l_dice = self.dice_loss(pred_masks, target_masks)
        l_bce = F.binary_cross_entropy_with_logits(pred_masks, target_masks)

        total = self.lambda_dice * l_dice + self.lambda_bce * l_bce

        losses = {
            "total": total,
            "dice": l_dice.detach(),
            "bce": l_bce.detach(),
        }

        # Change map loss (auxiliary)
        if change_logits is not None and pre_masks is not None:
            mid_for_change = mid_masks if mid_masks is not None else target_masks

            # Resize change logits to mask resolution if needed
            if change_logits.shape[-2:] != target_masks.shape[-2:]:
                change_logits = F.interpolate(
                    change_logits,
                    size=target_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            change_labels = build_change_labels(pre_masks, mid_for_change)
            l_change = self.change_loss(change_logits, change_labels)
            total = total + self.lambda_change * l_change

            losses["change"] = l_change.detach()
            losses["total"] = total

        return losses
