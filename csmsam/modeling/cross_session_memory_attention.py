"""
CrossSessionMemoryAttention — the core novel module of CSM-SAM.

Extends SAM2's within-session memory attention to cross-visit temporal propagation.
Given:
  - curr_features : features of current mid-RT slice (B, HW, C)
  - M_pre         : pre-RT memory bank (B, N_pre, C)  — encoded from week-0 scan
  - M_mid         : within-session mid-RT memory (B, N_mid, C) — prior slices of same scan
  - weeks_elapsed : integer or float tensor (B,) — weeks between pre-RT and mid-RT scans

The module learns to gate between:
  (1) Cross-session context  — what the pre-RT scan tells us about tumor location
  (2) Within-session context — what earlier slices of the mid-RT scan look like

Both attention streams are summed via a learned sigmoid gate, allowing the model
to shift from pre-RT guidance (early slices, no mid-RT context yet) to within-session
context (later slices, rich mid-RT context available).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DiscreteTemporalEmbedding(nn.Module):
    """Additive embedding conditioned on weeks elapsed (0–max_weeks supported).

    This is the ABLATION baseline: a discrete ``nn.Embedding`` lookup over
    integer weeks. It is kept alongside :class:`ContinuousTimeEncoder` so the
    two variants can be compared head-to-head in experiments.
    """

    def __init__(self, d_model: int, max_weeks: int = 12):
        super().__init__()
        self.embedding = nn.Embedding(max_weeks + 1, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, weeks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            weeks: (B,) integer tensor of weeks elapsed
        Returns:
            (B, 1, d_model) temporal bias added to memory tokens
        """
        return self.embedding(weeks).unsqueeze(1)  # (B, 1, C)


class ContinuousTimeEncoder(nn.Module):
    """Continuous-time temporal encoder.

    Accepts a scalar float tensor of weeks elapsed (need not be integer) and
    produces an additive bias for memory tokens. Uses sinusoidal features at
    ``n_frequencies`` log-spaced (base 2) frequencies, followed by a small MLP.

    Initialization is intentionally small (std=0.02 on the final projection
    with zero bias) so that the encoder's contribution is near-zero at init
    and training starts from a neutral point.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 128,
        n_frequencies: int = 6,
        normalize_by: float = 12.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_frequencies = n_frequencies
        self.normalize_by = float(normalize_by) if normalize_by else 1.0

        in_features = 2 * n_frequencies
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

        # Small init on the final projection so the bias is near-zero at start.
        final_linear = self.mlp[-1]
        nn.init.normal_(final_linear.weight, std=0.02)
        if final_linear.bias is not None:
            nn.init.zeros_(final_linear.bias)

    def forward(self, weeks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            weeks: (B,) float tensor of weeks elapsed (need not be integer)
        Returns:
            (B, 1, d_model) temporal bias added to memory tokens
        """
        t = weeks.to(dtype=torch.float32)
        if self.normalize_by and self.normalize_by != 1.0:
            t = t / self.normalize_by

        freqs = 2.0 ** torch.arange(
            self.n_frequencies, device=t.device, dtype=torch.float32
        )  # (n_freq,)
        ang = t.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, n_freq)
        features = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, 2*n_freq)

        out = self.mlp(features)  # (B, d_model)
        return out.unsqueeze(1)  # (B, 1, d_model)


def build_temporal_encoder(
    kind: str,
    d_model: int,
    max_weeks: int = 12,
    hidden_dim: int = 128,
    n_frequencies: int = 6,
) -> nn.Module:
    """Dispatch to the requested temporal encoder variant."""
    if kind == "continuous":
        return ContinuousTimeEncoder(
            d_model, hidden_dim=hidden_dim, n_frequencies=n_frequencies
        )
    elif kind == "discrete":
        return DiscreteTemporalEmbedding(d_model, max_weeks=max_weeks)
    else:
        raise ValueError(f"Unknown temporal encoder: {kind}")


class MultiHeadCrossAttention(nn.Module):
    """Standard multi-head cross-attention with optional key/value memory."""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, Nq, C)
            key:   (B, Nk, C)
            value: (B, Nk, C)
            key_padding_mask: (B, Nk) bool mask, True = pad token
        Returns:
            out:   (B, Nq, C)
            attn:  (B, num_heads, Nq, Nk)
        """
        B, Nq, C = query.shape
        Nk = key.shape[1]

        Q = self.q_proj(query).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / self.scale  # (B, H, Nq, Nk)

        if key_padding_mask is not None:
            # Expand mask to (B, 1, 1, Nk) and mask out pad tokens
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ V).transpose(1, 2).reshape(B, Nq, C)
        out = self.out_proj(out)
        return out, attn_weights


class CrossSessionMemoryAttention(nn.Module):
    """
    Cross-Session Memory Attention module.

    Replaces SAM2's standard MemoryAttention in the mid-RT forward pass.
    Operates on flattened spatial features (B, HW, C) from the SAM2 image encoder.

    Architecture:
        1. Cross-session attention: curr ← M_pre + temporal_embed(weeks)
        2. Within-session attention: curr ← M_mid  (standard SAM2 memory)
        3. Gate: learned sigmoid that blends the two contexts
        4. LayerNorm + residual connection

    Parameters trained: ~2M (cross_attn + within_attn + gate + norm layers)
    SAM2 encoder/decoder: frozen
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        max_weeks: int = 12,
        norm_eps: float = 1e-6,
        temporal_encoder_type: str = "continuous",
        temporal_hidden_dim: int = 128,
        temporal_n_frequencies: int = 6,
    ):
        super().__init__()

        # Cross-session attention (pre-RT memory → current features)
        self.cross_session_attn = MultiHeadCrossAttention(d_model, num_heads, dropout)

        # Within-session attention (mid-RT prior slices → current features)
        self.within_session_attn = MultiHeadCrossAttention(d_model, num_heads, dropout)

        # Temporal encoder for weeks elapsed (continuous by default; discrete ablation available)
        self.temporal_embed = build_temporal_encoder(
            temporal_encoder_type,
            d_model,
            max_weeks=max_weeks,
            hidden_dim=temporal_hidden_dim,
            n_frequencies=temporal_n_frequencies,
        )

        # Gate: learned linear combination of cross- and within-session contexts
        # Input: concat of cross-session context + within-session context
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        # Feedforward after attention
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=norm_eps)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Re-apply the small init on the continuous-time encoder's final
        # projection so the temporal bias stays near-zero at start (the
        # sweeping _init_weights above would otherwise overwrite it).
        if isinstance(self.temporal_embed, ContinuousTimeEncoder):
            final_linear = self.temporal_embed.mlp[-1]
            nn.init.normal_(final_linear.weight, std=0.02)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)

    def _prepare_weeks(self, weeks: torch.Tensor) -> torch.Tensor:
        """Cast ``weeks`` to the dtype expected by the active temporal encoder."""
        if isinstance(self.temporal_embed, ContinuousTimeEncoder):
            return weeks.float()
        return weeks.long()

    def forward(
        self,
        curr_features: torch.Tensor,
        M_pre: torch.Tensor,
        M_mid: torch.Tensor | None,
        weeks_elapsed: torch.Tensor,
        M_pre_mask: torch.Tensor | None = None,
        M_mid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            curr_features : (B, HW, C)  — SAM2 image features for current mid-RT slice
            M_pre         : (B, N_pre, C) — pre-RT memory bank (from pre-RT forward pass)
            M_mid         : (B, N_mid, C) or None — within-session mid-RT memory
                            (None at the first mid-RT slice, gradually grows)
            weeks_elapsed : (B,) tensor — weeks between pre-RT and mid-RT.
                            Cast internally to long (discrete) or float (continuous).
            M_pre_mask    : (B, N_pre) bool — optional padding mask for pre memory
            M_mid_mask    : (B, N_mid) bool — optional padding mask for mid memory

        Returns:
            out           : (B, HW, C) — updated features with cross-session context
            cross_attn_w  : (B, H, HW, N_pre) — cross-session attention weights (for visualization)
            gate_vals     : (B, HW, 1)  — gate values (0=within-session, 1=cross-session)
        """
        B, HW, C = curr_features.shape

        # ------------------------------------------------------------------
        # 1. Cross-session attention: attend to pre-RT memory
        # ------------------------------------------------------------------
        # Add temporal embedding to pre-RT memory to inform the model about
        # how many weeks have elapsed (tumor shrinkage scales with time)
        weeks_typed = self._prepare_weeks(weeks_elapsed)
        temp_bias = self.temporal_embed(weeks_typed)  # (B, 1, C)
        M_pre_temp = M_pre + temp_bias  # (B, N_pre, C)

        cross_context, cross_attn_w = self.cross_session_attn(
            query=curr_features,
            key=M_pre_temp,
            value=M_pre,  # value is un-perturbed memory
            key_padding_mask=M_pre_mask,
        )  # (B, HW, C)

        # ------------------------------------------------------------------
        # 2. Within-session attention: attend to mid-RT prior slices
        # ------------------------------------------------------------------
        if M_mid is not None and M_mid.shape[1] > 0:
            within_context, _ = self.within_session_attn(
                query=curr_features,
                key=M_mid,
                value=M_mid,
                key_padding_mask=M_mid_mask,
            )  # (B, HW, C)
        else:
            # No within-session memory yet (first mid-RT slice)
            # Use zeros so the gate falls back entirely on cross-session context
            within_context = torch.zeros_like(curr_features)

        # ------------------------------------------------------------------
        # 3. Gated fusion
        # ------------------------------------------------------------------
        gate_vals = self.gate(
            torch.cat([cross_context, within_context], dim=-1)
        )  # (B, HW, C) — note: gate is per-channel, not scalar
        # Clamp gate to avoid extreme values during early training
        fused = gate_vals * cross_context + (1.0 - gate_vals) * within_context

        # ------------------------------------------------------------------
        # 4. Residual + norm + FFN
        # ------------------------------------------------------------------
        x = self.norm1(curr_features + fused)
        x = self.norm2(x + self.ffn(x))

        # Return gate values averaged over channels for visualization
        gate_vis = gate_vals.mean(dim=-1, keepdim=True)  # (B, HW, 1)

        return x, cross_attn_w, gate_vis


class CrossSessionMemoryEncoder(nn.Module):
    """
    Encodes a pre-RT scan into a memory bank M_pre.

    This runs once per patient before inference and caches the result.
    It pools SAM2 image features across selected slices of the pre-RT volume.

    The memory bank has shape (1, N_frames * HW_pooled, C) where N_frames is
    the number of pre-RT slices encoded into memory.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_memory_frames: int = 8,
        spatial_pool_size: int = 16,
    ):
        super().__init__()
        self.n_memory_frames = n_memory_frames
        self.spatial_pool_size = spatial_pool_size

        # Project encoder features to memory dimensionality
        self.memory_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable mask-conditioned memory refinement
        # Given the pre-RT mask, we up-weight features near the tumor
        self.mask_modulator = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        pre_features: torch.Tensor,
        pre_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pre_features : (B, N_slices, C, H, W) — SAM2 features for each pre-RT slice
            pre_mask     : (B, N_slices, 1, H, W) — pre-RT segmentation mask (optional)

        Returns:
            M_pre : (B, N_memory, C) — memory bank ready for CrossSessionMemoryAttention
        """
        B, N, C, H, W = pre_features.shape

        # Uniformly sample N_memory_frames from the pre-RT volume
        if N > self.n_memory_frames:
            indices = torch.linspace(0, N - 1, self.n_memory_frames, dtype=torch.long, device=pre_features.device)
            pre_features = pre_features[:, indices]
            if pre_mask is not None:
                pre_mask = pre_mask[:, indices]
            N = self.n_memory_frames

        # Spatial pooling to reduce sequence length
        feat = pre_features.reshape(B * N, C, H, W)
        feat = F.adaptive_avg_pool2d(feat, self.spatial_pool_size)  # (B*N, C, P, P)
        P = self.spatial_pool_size

        # Mask-conditioned modulation: up-weight features near the tumor
        if pre_mask is not None:
            mask = pre_mask.reshape(B * N, 1, H, W)
            mask = F.interpolate(mask, size=(P, P), mode="bilinear", align_corners=False)
            modulation = self.mask_modulator(mask)  # (B*N, C, P, P)
            feat = feat * (1.0 + modulation)

        # Flatten spatial dims and project to memory
        feat = feat.permute(0, 2, 3, 1).reshape(B * N, P * P, C)  # (B*N, P², C)
        feat = self.memory_proj(feat)

        # Reshape to (B, N*P², C)
        M_pre = feat.reshape(B, N * P * P, C)
        return M_pre


# Back-compat alias: older code imports ``TemporalEmbedding`` directly.
TemporalEmbedding = DiscreteTemporalEmbedding


if __name__ == "__main__":
    # Smoke test for both temporal encoders.
    d = 256
    t = torch.tensor([0.0, 3.5, 7.0])
    cont = ContinuousTimeEncoder(d)
    disc = DiscreteTemporalEmbedding(d)
    out_c = cont(t)
    out_d = disc(t.long())
    assert out_c.shape == (3, 1, d), out_c.shape
    assert out_d.shape == (3, 1, d), out_d.shape
    print("OK", out_c.shape, out_d.shape)
