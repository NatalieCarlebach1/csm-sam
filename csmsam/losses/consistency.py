"""
Feature consistency auxiliary loss.

Trains a small predictor g(pre_features, t) -> predicted_mid_features and
minimizes ||predicted_mid - actual_mid||^2 on tumor regions. Provides a
longitudinal self-supervised signal: the model must learn HOW tumor
representations evolve over time, not just pattern-match.

Separate class weights for inside vs outside the union tumor mask so the
loss focuses on the clinically relevant regions.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_time_encoding(t: torch.Tensor, n_frequencies: int = 6, base: float = 2.0) -> torch.Tensor:
    """Encode scalar time (B,) as sin/cos features of dimension 2 * n_frequencies.

    Frequencies are base^0, base^1, ..., base^(n_frequencies - 1).
    """
    if t.dim() == 0:
        t = t.unsqueeze(0)
    t = t.float().unsqueeze(-1)  # (B, 1)
    freqs = torch.pow(
        base,
        torch.arange(n_frequencies, device=t.device, dtype=t.dtype),
    )  # (n_frequencies,)
    angles = t * freqs * math.pi  # (B, n_frequencies)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, 2*n_freq)


class _FiLMResBlock(nn.Module):
    """Bottleneck residual conv block with FiLM conditioning.

    Conv3x3 (C -> hidden) -> FiLM (gamma * x + beta) -> GELU -> Conv3x3 (hidden -> C) -> + residual

    The bottleneck keeps the parameter count low while preserving the C-channel
    I/O shape. FiLM is applied at the bottleneck (hidden) channels.
    """

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        # bias=False: FiLM's beta term in block1 supplies the bias for conv1,
        # and conv2 is followed by a residual add so a bias is redundant.
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv1(x)
        # gamma, beta: (B, hidden) -> (B, hidden, 1, 1)
        h = gamma.unsqueeze(-1).unsqueeze(-1) * h + beta.unsqueeze(-1).unsqueeze(-1)
        h = self.act(h)
        h = self.conv2(h)
        return h + residual


class FeatureEvolutionPredictor(nn.Module):
    """Small MLP/CNN that takes pre-RT spatial features + scalar time and
    predicts mid-RT features.

    Input:  pre_features (B, C, h, w), t (B,)
    Output: predicted_mid_features (B, C, h, w)

    Architecture:
      - Sinusoidal time encoding (n_frequencies, base 2) -> 2-layer MLP -> (2*C,) FiLM params
      - 2 residual conv blocks with FiLM conditioning
    """

    def __init__(self, d_model: int = 256, hidden: int = 24, n_frequencies: int = 6):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.n_frequencies = n_frequencies

        time_dim = 2 * n_frequencies
        # Time MLP -> FiLM params for both blocks: 2 blocks * (gamma, beta) at bottleneck width
        time_mlp_hidden = max(32, hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_mlp_hidden),
            nn.GELU(),
            nn.Linear(time_mlp_hidden, 4 * hidden),
        )

        self.block1 = _FiLMResBlock(d_model, hidden)
        self.block2 = _FiLMResBlock(d_model, hidden)

    def forward(self, pre_features: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pre_features: (B, C, h, w)
            t:            (B,) scalar weeks elapsed
        Returns:
            predicted_mid_features: (B, C, h, w)
        """
        B, C, _, _ = pre_features.shape
        if C != self.d_model:
            raise ValueError(
                f"FeatureEvolutionPredictor expected d_model={self.d_model}, got C={C}"
            )

        t_enc = _sinusoidal_time_encoding(t.to(pre_features.dtype), self.n_frequencies)  # (B, 2*n_freq)
        film = self.time_mlp(t_enc)  # (B, 4*C)
        gamma1, beta1, gamma2, beta2 = film.chunk(4, dim=-1)  # each (B, C)
        # Initialize gamma around 1 for stability (identity FiLM at init).
        gamma1 = 1.0 + gamma1
        gamma2 = 1.0 + gamma2

        x = self.block1(pre_features, gamma1, beta1)
        x = self.block2(x, gamma2, beta2)
        return x


class FeatureConsistencyLoss(nn.Module):
    """Masked MSE between predicted and actual mid-RT features.

    L = lambda_fg * mean(MSE[in mask]) + lambda_bg * mean(MSE[out of mask])

    Args:
        lambda_fg: weight for tumor-region pixels (default 1.0)
        lambda_bg: weight for background pixels (default 0.1)
    """

    def __init__(self, lambda_fg: float = 1.0, lambda_bg: float = 0.1):
        super().__init__()
        self.lambda_fg = lambda_fg
        self.lambda_bg = lambda_bg

    def forward(
        self,
        predicted_mid: torch.Tensor,
        actual_mid: torch.Tensor,
        union_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted_mid: (B, C, h, w) from FeatureEvolutionPredictor
            actual_mid:    (B, C, h, w) from SAM2 encoder on the mid-RT slice
            union_mask:    (B, 1, H, W) in feature OR image resolution.
                           Will be resized (area) to feature resolution and
                           thresholded at 0.5.
        Returns:
            scalar loss
        """
        # Defensive detach on the target (frozen encoder output).
        actual_mid = actual_mid.detach()

        B, C, h, w = predicted_mid.shape

        # Resize union mask to feature resolution if needed.
        if union_mask.shape[-2:] != (h, w):
            # area mode for downsampling; bilinear fallback if upsampling.
            if union_mask.shape[-2] >= h and union_mask.shape[-1] >= w:
                mask_resized = F.interpolate(
                    union_mask.float(), size=(h, w), mode="area"
                )
            else:
                mask_resized = F.interpolate(
                    union_mask.float(), size=(h, w), mode="bilinear", align_corners=False
                )
        else:
            mask_resized = union_mask.float()

        # Threshold to boolean mask.
        fg_mask = (mask_resized > 0.5).float()  # (B, 1, h, w)

        # Per-pixel squared error, averaged across channels -> (B, 1, h, w)
        sq_err = (predicted_mid - actual_mid).pow(2).mean(dim=1, keepdim=True)

        fg_count = fg_mask.sum().clamp_min(1.0)
        bg_mask = 1.0 - fg_mask
        bg_count = bg_mask.sum().clamp_min(1.0)

        fg_loss = (sq_err * fg_mask).sum() / fg_count
        bg_loss = (sq_err * bg_mask).sum() / bg_count

        return self.lambda_fg * fg_loss + self.lambda_bg * bg_loss


if __name__ == "__main__":
    pred_net = FeatureEvolutionPredictor(d_model=64, hidden=64)
    pre = torch.randn(2, 64, 16, 16)
    t = torch.tensor([2.0, 5.5])
    predicted = pred_net(pre, t)
    actual = torch.randn_like(predicted)
    mask = torch.rand(2, 1, 16, 16).round()
    loss = FeatureConsistencyLoss()(predicted, actual, mask)
    print("OK", predicted.shape, loss.item(), sum(p.numel() for p in pred_net.parameters()))
