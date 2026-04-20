"""
ChangeHead — lightweight 3-class change map prediction head.

Predicts, for each spatial location in the mid-RT scan:
  Class 0: background (no tumor in pre-RT or mid-RT)
  Class 1: stable tumor (tumor in both pre-RT and mid-RT)
  Class 2: grown region  (tumor in mid-RT but NOT in pre-RT)
  Class 3: shrunk region (tumor in pre-RT but NOT in mid-RT)

Labels are derived automatically from pre-RT and mid-RT masks:
  label = 0 if pre==0 and mid==0
  label = 1 if pre==1 and mid==1
  label = 2 if pre==0 and mid==1
  label = 3 if pre==1 and mid==0

No extra annotation is required — this is free supervision.
The change map head is supervised with cross-entropy (weight 0.3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_change_labels(
    pre_mask: torch.Tensor,
    mid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Build 4-class change labels from binary pre-RT and mid-RT masks.

    Args:
        pre_mask : (B, 1, H, W) binary mask from pre-RT scan (0 or 1)
        mid_mask : (B, 1, H, W) binary mask from mid-RT scan  (0 or 1)

    Returns:
        change_label : (B, H, W) long tensor with values in {0, 1, 2, 3}
    """
    pre = (pre_mask > 0.5).squeeze(1).long()  # (B, H, W)
    mid = (mid_mask > 0.5).squeeze(1).long()  # (B, H, W)

    # 0: background, 1: stable, 2: grown, 3: shrunk
    change_label = torch.zeros_like(pre)  # background
    change_label[(pre == 1) & (mid == 1)] = 1  # stable
    change_label[(pre == 0) & (mid == 1)] = 2  # grown
    change_label[(pre == 1) & (mid == 0)] = 3  # shrunk
    return change_label


class ChangeHead(nn.Module):
    """
    Lightweight 4-class change map prediction head.

    Input: concatenation of [pre-RT features, mid-RT features] after alignment.
    Output: (B, 4, H, W) logits for change classification.

    Architecture:
        Conv(2C → C) → BN → ReLU
        Conv(C → C//2) → BN → ReLU
        Conv(C//2 → 4)
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 4):
        super().__init__()
        mid = in_channels // 2

        self.head = nn.Sequential(
            # Fuse pre-RT and mid-RT features
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            # Reduce
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=False),
            # Classify
            nn.Conv2d(mid, num_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        pre_features: torch.Tensor,
        mid_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pre_features : (B, C, H, W) — spatial features from pre-RT slice
            mid_features : (B, C, H, W) — spatial features from mid-RT slice

        Returns:
            logits : (B, 4, H, W) — change map logits
        """
        # Align spatial resolutions if needed
        if pre_features.shape[-2:] != mid_features.shape[-2:]:
            pre_features = F.interpolate(
                pre_features,
                size=mid_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([pre_features, mid_features], dim=1)  # (B, 2C, H, W)
        return self.head(x)


class ChangeMapLoss(nn.Module):
    """
    Cross-entropy loss for change map prediction.

    Background class (0) is down-weighted because it dominates.
    """

    def __init__(self, weight: list[float] | None = None):
        super().__init__()
        if weight is None:
            # Background is ~90% of pixels; up-weight tumor-change classes
            weight = [0.1, 1.0, 2.0, 2.0]
        self.register_buffer("weight", torch.tensor(weight))

    def forward(
        self,
        logits: torch.Tensor,
        change_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits       : (B, 4, H, W)
            change_labels: (B, H, W) long tensor

        Returns:
            loss : scalar
        """
        weight = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(logits, change_labels, weight=weight)
