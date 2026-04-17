"""
Variational Change Latent for CSM-SAM (Emergent Treatment Response Phenotyping).

Core idea: learn a low-dimensional latent z that summarises each patient's
treatment response phenotype.  If trained purely on the segmentation objective
(no ΔV labels), z should still organise into clinically meaningful clusters —
fast responders, slow responders, non-responders — as an *emergent* property
of the representation.  This is verified post-hoc (Spearman ρ vs ΔV), not
supervised.

Sub-modules
-----------
ChangeLatentEncoder  — posterior q(z | f_pre, f_mid, t): used at training time
                        when both sessions are available.
ChangePrior          — prior p(z | f_pre, t): used at inference.  Trained via
                        KL(q || p) so it learns what changes are plausible given
                        the pre-session scan.
FiLMConditioner      — z → (gamma, beta) for FiLM-conditioning of the
                        cross-session attention query.

Loss addition
-------------
  L_KL = 0.5 * Σ_j [ (lv_p_j − lv_q_j) + exp(lv_q_j − lv_p_j)
                      + (μ_q_j − μ_p_j)² exp(−lv_p_j) − 1 ]

  where lv = log(σ²).  Annealed: β(e) = β_max · min(1, e / warmup).

Inference-time z
----------------
  z = (1 − α) · μ_prior + α · Σ_i w_i z_i
  where z_i are posterior means stored in the cross-patient bank and
  w_i ∝ softmax(cosine_sim / τ).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sinusoidal_encoding(
    t: torch.Tensor,
    d_out: int,
    n_freq: int = 8,
    normalize_by: float = 12.0,
) -> torch.Tensor:
    """Map scalar weeks_elapsed (B,) to sinusoidal features (B, d_out).

    Uses n_freq log₂-spaced frequencies, then a small linear projection to
    d_out.  Separate from the ContinuousTimeEncoder in attention so the latent
    encoder learns its own temporal embedding.
    """
    t = t.float() / normalize_by                      # (B,)
    freqs = 2.0 ** torch.arange(n_freq, device=t.device, dtype=torch.float32)
    ang = t.unsqueeze(-1) * freqs.unsqueeze(0)        # (B, n_freq)
    feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, 2*n_freq)
    if 2 * n_freq == d_out:
        return feats
    # Linear projection to d_out (no bias; the MLP downstream adds bias)
    if not hasattr(_sinusoidal_encoding, "_proj") or \
            _sinusoidal_encoding._proj.weight.shape != (d_out, 2 * n_freq):
        # Lazily-created linear; not ideal but avoids carrying extra state here.
        # In practice callers use SinusoidalTimeEncoder instead.
        proj = nn.Linear(2 * n_freq, d_out, bias=False).to(t.device)
        nn.init.normal_(proj.weight, std=0.02)
        _sinusoidal_encoding._proj = proj
    return _sinusoidal_encoding._proj(feats)


class SinusoidalTimeEncoder(nn.Module):
    """Shared sinusoidal → MLP temporal encoder for the latent modules.

    Separate weights from the ContinuousTimeEncoder in CrossSessionMemoryAttention
    so the two encoders can specialise to their respective tasks.
    """

    def __init__(self, d_out: int, n_freq: int = 8, normalize_by: float = 12.0):
        super().__init__()
        self.normalize_by = normalize_by
        self.n_freq = n_freq
        in_dim = 2 * n_freq
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, weeks: torch.Tensor) -> torch.Tensor:
        """weeks: (B,) float or long  →  (B, d_out)."""
        t = weeks.float() / self.normalize_by
        freqs = 2.0 ** torch.arange(
            self.n_freq, device=t.device, dtype=torch.float32
        )
        ang = t.unsqueeze(-1) * freqs.unsqueeze(0)    # (B, n_freq)
        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.mlp(feats)                         # (B, d_out)


def kl_divergence(
    mu_q: torch.Tensor,
    log_var_q: torch.Tensor,
    mu_p: torch.Tensor,
    log_var_p: torch.Tensor,
) -> torch.Tensor:
    """Analytical KL(q || p) for two diagonal Gaussians.

    KL = 0.5 * Σ_j [ (lv_p_j − lv_q_j)
                    + exp(lv_q_j − lv_p_j)
                    + (μ_q_j − μ_p_j)² exp(−lv_p_j)
                    − 1 ]

    Args:
        mu_q, log_var_q : (B, d_z) posterior parameters
        mu_p, log_var_p : (B, d_z) prior parameters

    Returns:
        scalar — mean KL per batch element summed over d_z.
    """
    # Clamp for numerical stability
    lv_q = log_var_q.clamp(-10.0, 10.0)
    lv_p = log_var_p.clamp(-10.0, 10.0)

    kl = 0.5 * (
        (lv_p - lv_q)
        + (lv_q - lv_p).exp()
        + (mu_q - mu_p).pow(2) * (-lv_p).exp()
        - 1.0
    )  # (B, d_z)
    return kl.sum(dim=-1).mean()   # scalar


def kl_beta(epoch: int, beta_max: float, warmup_epochs: int) -> float:
    """Linear β annealing schedule.

    β starts at 0 (epoch 0) and ramps to beta_max at warmup_epochs.
    Prevents the KL from collapsing the posterior before the segmentation
    loss has shaped the latent space.
    """
    if warmup_epochs <= 0:
        return float(beta_max)
    return float(beta_max) * min(1.0, epoch / warmup_epochs)


# ---------------------------------------------------------------------------
# Posterior encoder  q(z | f_pre, f_mid, t)
# ---------------------------------------------------------------------------

class ChangeLatentEncoder(nn.Module):
    """Amortised posterior q(z | f_pre, f_mid, t).

    Inputs:
        f_pre   : (B, d_model)  — global-average-pooled pre-RT memory bank
        f_mid   : (B, d_model)  — global-average-pooled mid-RT SAM2 features
        weeks   : (B,)          — weeks elapsed (float or long)

    Outputs:
        mu       : (B, d_z)
        log_var  : (B, d_z)   — log σ²

    Architecture: concat [f_pre | f_mid | t_embed] → LayerNorm → MLP → μ/log_var heads.
    Final heads are zero-initialised so that at the start of training z ≈ 0
    and the KL term is small.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_z: int = 64,
        t_n_freq: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.d_z = d_z
        self.time_enc = SinusoidalTimeEncoder(d_model, n_freq=t_n_freq)

        in_dim = d_model * 3  # f_pre + f_mid + t_embed
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.mu_head      = nn.Linear(hidden_dim // 2, d_z)
        self.log_var_head = nn.Linear(hidden_dim // 2, d_z)

        # Zero init → z ≈ 0, log_var ≈ 0 (σ≈1) at start
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_var_head.weight)
        nn.init.zeros_(self.log_var_head.bias)

    def forward(
        self,
        f_pre: torch.Tensor,
        f_mid: torch.Tensor,
        weeks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.time_enc(weeks)                       # (B, d_model)
        h = self.net(torch.cat([f_pre, f_mid, t], dim=-1))
        return self.mu_head(h), self.log_var_head(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick: z = μ + σ·ε,  ε ~ N(0,I)."""
        std = (0.5 * log_var.clamp(-10.0, 10.0)).exp()
        return mu + std * torch.randn_like(std)


# ---------------------------------------------------------------------------
# Prior network  p(z | f_pre, t)
# ---------------------------------------------------------------------------

class ChangePrior(nn.Module):
    """Learned prior p(z | f_pre, t).

    Used at inference time when mid-RT features are not yet available.
    Trained jointly via the KL term: KL(q(z|f_pre,f_mid,t) || p(z|f_pre,t)).
    This forces the prior to learn what changes are plausible given the
    pre-session scan and elapsed time.

    Inputs:
        f_pre  : (B, d_model)
        weeks  : (B,)

    Outputs:
        mu_p      : (B, d_z)
        log_var_p : (B, d_z)
    """

    def __init__(
        self,
        d_model: int = 256,
        d_z: int = 64,
        t_n_freq: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.d_z = d_z
        self.time_enc = SinusoidalTimeEncoder(d_model, n_freq=t_n_freq)

        in_dim = d_model * 2  # f_pre + t_embed
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.mu_head      = nn.Linear(hidden_dim // 2, d_z)
        self.log_var_head = nn.Linear(hidden_dim // 2, d_z)

        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_var_head.weight)
        nn.init.zeros_(self.log_var_head.bias)

    def forward(
        self,
        f_pre: torch.Tensor,
        weeks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.time_enc(weeks)                       # (B, d_model)
        h = self.net(torch.cat([f_pre, t], dim=-1))
        return self.mu_head(h), self.log_var_head(h)


# ---------------------------------------------------------------------------
# FiLM conditioner   z → (γ, β)
# ---------------------------------------------------------------------------

class FiLMConditioner(nn.Module):
    """Map latent z to FiLM scale/shift for conditioning the attention query.

    Applied to the cross-session attention query inside
    CrossSessionMemoryAttention:
        query' = query * (1 + γ.unsqueeze(1)) + β.unsqueeze(1)

    This makes the cross-session attention pattern depend on the inferred
    change phenotype: a fast-responder z retrieves different memory tokens
    than a non-responder z.

    Inputs:  z : (B, d_z)
    Outputs: gamma : (B, d_model)
             beta  : (B, d_model)

    Both heads are zero-initialised → identity transform at init (no effect
    on the base model at the start of training).
    """

    def __init__(self, d_z: int = 64, d_model: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(d_z, d_model),
            nn.GELU(),
        )
        self.gamma_head = nn.Linear(d_model, d_model)
        self.beta_head  = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(z)
        return self.gamma_head(h), self.beta_head(h)   # (B, d_model), (B, d_model)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, d_z = 4, 256, 64

    enc   = ChangeLatentEncoder(d_model=C, d_z=d_z)
    prior = ChangePrior(d_model=C, d_z=d_z)
    film  = FiLMConditioner(d_z=d_z, d_model=C)

    f_pre  = torch.randn(B, C)
    f_mid  = torch.randn(B, C)
    weeks  = torch.tensor([2.0, 3.0, 2.5, 4.0])

    mu_q, lv_q = enc(f_pre, f_mid, weeks)
    mu_p, lv_p = prior(f_pre, weeks)
    z = ChangeLatentEncoder.reparameterize(mu_q, lv_q)
    kl = kl_divergence(mu_q, lv_q, mu_p, lv_p)
    gamma, beta = film(z)

    assert mu_q.shape == (B, d_z),   mu_q.shape
    assert lv_q.shape == (B, d_z),   lv_q.shape
    assert z.shape    == (B, d_z),   z.shape
    assert kl.ndim    == 0,          kl.shape
    assert gamma.shape == (B, C),    gamma.shape
    assert beta.shape  == (B, C),    beta.shape

    # KL should be near 0 at init (zero-init heads → μ≈0, σ≈1 for both)
    assert kl.item() < 0.1, f"Expected small KL at init, got {kl.item():.4f}"

    # Beta schedule
    assert kl_beta(0, 0.1, 10)  == 0.0
    assert kl_beta(5, 0.1, 10)  == 0.05
    assert kl_beta(10, 0.1, 10) == 0.1
    assert kl_beta(20, 0.1, 10) == 0.1

    print("change_latent.py smoke test OK")
    print(f"  mu_q:  {mu_q.shape}, kl: {kl.item():.6f}")
    print(f"  gamma: {gamma.shape}, beta: {beta.shape}")
