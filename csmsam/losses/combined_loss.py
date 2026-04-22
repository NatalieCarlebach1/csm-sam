"""
Combined loss for CSM-SAM training.

L_total = L_dice(pred_mask, mid_mask)
        + λ_ce * L_bce(pred_mask, mid_mask)
        + λ_change * L_ce(pred_change, change_label)

Defaults: λ_ce = 1.0, λ_change = 0.3

Supports per-structure predictions (B, 2, H, W): channel 0 = GTVp, channel 1 = GTVn.
Single-channel (combined tumor) predictions (B, 1, H, W) remain supported.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from csmsam.modeling.change_head import ChangeMapLoss, build_change_labels
from csmsam.losses.consistency import FeatureConsistencyLoss


def _batch_pos_weight(targets: torch.Tensor, cap: float = 20.0) -> torch.Tensor:
    """Per-channel bg/fg pixel ratio for BCE pos_weight, capped to avoid instability."""
    B, C, H, W = targets.shape
    flat = targets.reshape(B, C, -1)
    n_pos = flat.sum(dim=(0, 2)).clamp_min(1.0)
    n_neg = (1.0 - flat).sum(dim=(0, 2)).clamp_min(1.0)
    return (n_neg / n_pos).clamp_max(cap)


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    L_dice = 1 - (2 * |P ∩ G| + ε) / (|P| + |G| + ε)

    Accepts (B, C, H, W). Dice is computed per channel per batch element,
    then averaged across the batch and channels.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C, H, W) — raw predictions (before sigmoid)
            targets : (B, C, H, W) — binary masks {0, 1}

        Returns:
            scalar dice loss averaged across all (batch, channel) pairs.
        """
        if logits.dim() == 4:
            probs = torch.sigmoid(logits)
            B, C = probs.shape[0], probs.shape[1]
            probs_flat = probs.reshape(B, C, -1)
            targets_flat = targets.reshape(B, C, -1)

            intersection = (probs_flat * targets_flat).sum(dim=2)  # (B, C)
            denom = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)
            dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)  # (B, C)
            return 1.0 - dice.mean()
        else:
            # Fallback: treat as single-channel legacy path
            probs = torch.sigmoid(logits)
            probs_flat = probs.reshape(probs.shape[0], -1)
            targets_flat = targets.reshape(targets.shape[0], -1)
            intersection = (probs_flat * targets_flat).sum(dim=1)
            dice = (2.0 * intersection + self.smooth) / (
                probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
            )
            return 1.0 - dice.mean()

    def per_channel(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dice loss per channel (averaged across batch only).

        Returns:
            (C,) tensor of per-channel dice losses.
        """
        probs = torch.sigmoid(logits)
        B, C = probs.shape[0], probs.shape[1]
        probs_flat = probs.reshape(B, C, -1)
        targets_flat = targets.reshape(B, C, -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)  # (B, C)
        denom = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)  # (B, C)
        return 1.0 - dice.mean(dim=0)  # (C,)


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

    Supports both single-channel (B, 1, H, W) and two-channel (B, 2, H, W)
    predictions. For two-channel predictions, channel 0 is GTVp and
    channel 1 is GTVn. Per-channel dice is reported for monitoring.

    Change map supervision is structure-agnostic: pre_masks and mid_masks
    for the change head are single-channel combined masks.
    """

    def __init__(
        self,
        lambda_dice: float = 1.0,
        lambda_bce: float = 1.0,
        lambda_change: float = 0.3,
        change_loss_weights: list[float] | None = None,
        gtvn_weight: float = 1.0,
        lambda_consistency: float = 0.2,
        consistency_fg: float = 1.0,
        consistency_bg: float = 0.1,
    ):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.lambda_change = lambda_change
        self.gtvn_weight = gtvn_weight
        self.lambda_consistency = lambda_consistency

        self.dice_loss = DiceLoss(smooth=1.0)
        self.change_loss = ChangeMapLoss(weight=change_loss_weights)

        # Optional feature-consistency auxiliary loss. Only instantiated when
        # the weight is positive to avoid silent computation.
        self.consistency_loss: FeatureConsistencyLoss | None
        if lambda_consistency > 0:
            self.consistency_loss = FeatureConsistencyLoss(
                lambda_fg=consistency_fg,
                lambda_bg=consistency_bg,
            )
        else:
            self.consistency_loss = None

    def forward(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        change_logits: torch.Tensor | None = None,
        pre_masks: torch.Tensor | None = None,
        mid_masks: torch.Tensor | None = None,
        predicted_mid_features: torch.Tensor | None = None,
        actual_mid_features: torch.Tensor | None = None,
        union_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred_masks   : (B, C, H, W) — segmentation logits (C==1 or C==2)
            target_masks : (B, C', H, W) — binary mid-RT masks;
                           if C==2 and C'==1, target is broadcast to both channels.
            change_logits: (B, 4, H, W) — change map logits (optional)
            pre_masks    : (B, 1, H, W) — pre-RT masks (combined, for change labels)
            mid_masks    : (B, 1, H, W) — mid-RT masks (combined, for change labels)
            predicted_mid_features: (B, C_feat, h, w) — output of FeatureEvolutionPredictor
            actual_mid_features:    (B, C_feat, h, w) — SAM2 encoder features on mid-RT slice
            union_mask:             (B, 1, H, W) or (B, 1, h, w) — (pre | mid) tumor mask

        Returns:
            dict with 'total', 'dice', 'bce', optional 'change', 'consistency',
            and per-channel 'dice_gtvp', 'dice_gtvn' when C == 2.
        """
        C_pred = pred_masks.shape[1]

        # Broadcast target to match pred channels if needed
        if C_pred == 2 and target_masks.shape[1] == 1:
            target_seg = target_masks.expand_as(pred_masks)
        elif pred_masks.shape[1] == target_masks.shape[1]:
            target_seg = target_masks
        else:
            raise ValueError(
                f"Incompatible pred/target channels: pred={pred_masks.shape}, "
                f"target={target_masks.shape}"
            )

        if C_pred == 2:
            # Per-channel dice for reporting and optional weighting
            per_ch_dice = self.dice_loss.per_channel(pred_masks, target_seg)  # (2,)
            dice_gtvp = per_ch_dice[0]
            dice_gtvn = per_ch_dice[1]

            # Weighted sum over channels: GTVp has implicit weight 1.0
            l_dice = dice_gtvp + self.gtvn_weight * dice_gtvn

            # Per-channel BCE with dynamic pos_weight for class imbalance
            pw = _batch_pos_weight(target_seg).view(1, C_pred, 1, 1)
            bce_per_ch = F.binary_cross_entropy_with_logits(
                pred_masks, target_seg, pos_weight=pw, reduction="none"
            ).mean(dim=(0, 2, 3))  # (2,)
            l_bce = bce_per_ch[0] + self.gtvn_weight * bce_per_ch[1]
        else:
            # Single-channel path
            l_dice = self.dice_loss(pred_masks, target_seg)
            pw = _batch_pos_weight(target_seg).view(1, 1, 1, 1)
            l_bce = F.binary_cross_entropy_with_logits(pred_masks, target_seg, pos_weight=pw)

        total = self.lambda_dice * l_dice + self.lambda_bce * l_bce

        losses: dict[str, torch.Tensor] = {
            "total": total,
            "dice": l_dice.detach(),
            "bce": l_bce.detach(),
        }
        if C_pred == 2:
            losses["dice_gtvp"] = dice_gtvp.detach()
            losses["dice_gtvn"] = dice_gtvn.detach()

        # Change map loss (auxiliary) — structure-agnostic, always single-channel
        if change_logits is not None and pre_masks is not None:
            # Choose the single-channel mid mask for change labels
            if mid_masks is not None:
                mid_for_change = mid_masks
            elif target_masks.shape[1] == 1:
                mid_for_change = target_masks
            else:
                # Collapse multi-channel target to combined single-channel
                mid_for_change = target_masks.amax(dim=1, keepdim=True)

            # Resize change logits to mask resolution if needed
            if change_logits.shape[-2:] != mid_for_change.shape[-2:]:
                change_logits = F.interpolate(
                    change_logits,
                    size=mid_for_change.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            change_labels = build_change_labels(pre_masks, mid_for_change)
            l_change = self.change_loss(change_logits, change_labels)
            total = total + self.lambda_change * l_change

            losses["change"] = l_change.detach()
            losses["total"] = total

        # Feature consistency (auxiliary) — only when all three tensors are provided
        # and the weight is positive.
        if (
            self.consistency_loss is not None
            and self.lambda_consistency > 0
            and predicted_mid_features is not None
            and actual_mid_features is not None
            and union_mask is not None
        ):
            l_consistency = self.consistency_loss(
                predicted_mid_features, actual_mid_features, union_mask
            )
            total = total + self.lambda_consistency * l_consistency
            losses["consistency"] = l_consistency.detach()
            losses["total"] = total

        return losses
