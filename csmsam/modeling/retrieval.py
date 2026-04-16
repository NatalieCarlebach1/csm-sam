"""
Cross-Patient Retrieval for CSM-SAM.

At inference time we look up the K most similar *training* patients (by pre-RT
feature summary) and inject their pre→mid change "template" tokens as additional
cross-session memory.

Pieces:
  - CrossPatientBank: persistent store of {patient_id -> (pre_summary,
    change_template, weeks_elapsed)}. Retrieval is cosine top-K on
    L2-normalized pre summaries.
  - CrossPatientRetrieval: projection head that prepares retrieved templates
    for concatenation onto M_pre, plus a learned scalar gate (initialized
    near 0) so the model starts by ignoring retrieval and learns to trust it.
  - compute_pre_summary / compute_change_template: helpers that turn encoder
    features + masks into bank entries.

Bank entries are stored detached on CPU (frozen reference tokens). The
projection MLP + gate are gradient-flowing.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_pre_summary(
    pre_features: torch.Tensor,
    pre_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask-weighted mean of pre-RT encoder features.

    Args:
        pre_features: (B, N_slices, C, h, w)
        pre_mask:     (B, N_slices, 1, h, w) in feature resolution (optional)
    Returns:
        (B, C) — falls back to unweighted mean if the mask is empty.
    """
    B, N, C, h, w = pre_features.shape
    feats = pre_features.reshape(B, N * h * w, C)
    if pre_mask is None:
        return feats.mean(dim=1)
    m = pre_mask.reshape(B, N * h * w, 1).float()
    denom = m.sum(dim=1).clamp_min(1e-6)
    weighted = (feats * m).sum(dim=1) / denom
    empty = (m.sum(dim=1) < 1e-6).squeeze(-1)
    if empty.any():
        weighted = torch.where(empty.unsqueeze(-1), feats.mean(dim=1), weighted)
    return weighted


def compute_change_template(
    pre_features: torch.Tensor,
    mid_features: torch.Tensor,
    pre_mask: torch.Tensor | None,
    mid_mask: torch.Tensor | None,
    n_tokens: int = 16,
) -> torch.Tensor:
    """Spatial-pool (mid − pre) at tumor locations into (B, n_tokens, C).

    Uses the UNION mask (pre ∪ mid) to zero non-tumor regions, masked-mean
    across slices, and adaptive-avg-pool to sqrt(n_tokens)².
    """
    assert pre_features.shape == mid_features.shape
    B, N, C, h, w = pre_features.shape
    diff = mid_features - pre_features  # (B, N, C, h, w)

    union = None
    if pre_mask is not None or mid_mask is not None:
        parts = [t.float() for t in (pre_mask, mid_mask) if t is not None]
        union = torch.stack(parts, dim=0).max(dim=0).values  # (B, N, 1, h, w)
        empty = union.sum(dim=(1, 2, 3, 4)) < 1e-6
        if empty.any():
            union = torch.where(empty.view(B, 1, 1, 1, 1), torch.ones_like(union), union)
        diff = diff * union

    if union is not None:
        denom = union.sum(dim=1).clamp_min(1e-6)        # (B, 1, h, w)
        diff_2d = diff.sum(dim=1) / denom               # (B, C, h, w)
    else:
        diff_2d = diff.mean(dim=1)                      # (B, C, h, w)

    side = max(int(round(math.sqrt(n_tokens))), 1)
    pooled = F.adaptive_avg_pool2d(diff_2d, (side, side))       # (B, C, s, s)
    template = pooled.flatten(2).transpose(1, 2).contiguous()    # (B, s*s, C)

    t = template.shape[1]
    if t < n_tokens:
        template = torch.cat([template, template.new_zeros(B, n_tokens - t, C)], dim=1)
    elif t > n_tokens:
        template = template[:, :n_tokens]
    return template


# ---------------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------------

class CrossPatientBank:
    """Cosine top-K store over per-patient (pre_summary, change_template).

    Entries are stored detached on CPU (frozen reference tokens).
    """

    def __init__(self):
        self.ids: list[str] = []
        self.summaries: list[torch.Tensor] = []       # each: (C,)
        self.templates: list[torch.Tensor] = []       # each: (N_tokens, C)
        self.weeks: list[int] = []
        self._summaries_t: torch.Tensor | None = None
        self._templates_t: torch.Tensor | None = None
        self._dirty: bool = True

    def __len__(self) -> int:
        return len(self.ids)

    def add(
        self,
        patient_id: str,
        pre_summary: torch.Tensor,
        change_template: torch.Tensor,
        weeks_elapsed: int,
    ) -> None:
        if pre_summary.dim() != 1:
            raise ValueError(f"pre_summary must be (C,), got {tuple(pre_summary.shape)}")
        if change_template.dim() != 2:
            raise ValueError(f"change_template must be (N_tokens, C), got {tuple(change_template.shape)}")
        self.ids.append(str(patient_id))
        self.summaries.append(pre_summary.detach().cpu().float())
        self.templates.append(change_template.detach().cpu().float())
        self.weeks.append(int(weeks_elapsed))
        self._dirty = True

    def _rebuild(self) -> None:
        if not self.summaries:
            self._summaries_t = None
            self._templates_t = None
        else:
            self._summaries_t = torch.stack(self.summaries, dim=0)   # (N, C)
            self._templates_t = torch.stack(self.templates, dim=0)   # (N, N_tokens, C)
        self._dirty = False

    def topk(
        self,
        query_pre_summary: torch.Tensor,
        k: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cosine top-K. Returns (indices, sims, templates) with sims/templates
        on the query's device; shapes (B, K), (B, K), (B, K, N_tokens, C).
        """
        if len(self) == 0:
            raise RuntimeError("CrossPatientBank is empty; cannot retrieve.")
        if self._dirty:
            self._rebuild()

        device, dtype = query_pre_summary.device, query_pre_summary.dtype
        bank_s = self._summaries_t.to(device=device, dtype=dtype)            # (N, C)
        bank_t = self._templates_t.to(device=device, dtype=dtype)            # (N, N_tokens, C)

        k_eff = min(k, bank_s.shape[0])
        q = F.normalize(query_pre_summary, dim=-1)
        s = F.normalize(bank_s, dim=-1)
        sims = q @ s.t()                                                      # (B, N)
        top_sims, top_idx = sims.topk(k_eff, dim=-1)

        B = q.shape[0]
        Nt, C = bank_t.shape[1], bank_t.shape[2]
        gathered = bank_t.index_select(0, top_idx.reshape(-1)).reshape(B, k_eff, Nt, C)

        # Pad by repeating last neighbor if requested k > bank size.
        if k_eff < k:
            pad = k - k_eff
            top_idx = torch.cat([top_idx, top_idx[:, -1:].expand(B, pad)], dim=1)
            top_sims = torch.cat([top_sims, top_sims[:, -1:].expand(B, pad)], dim=1)
            gathered = torch.cat([gathered, gathered[:, -1:].expand(B, pad, Nt, C)], dim=1)
        return top_idx, top_sims, gathered

    def save(self, path: str | Path) -> None:
        if self._dirty:
            self._rebuild()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ids": self.ids,
                "summaries_t": self._summaries_t,
                "templates_t": self._templates_t,
                "weeks": self.weeks,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CrossPatientBank":
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        bank = cls()
        bank.ids = list(obj["ids"])
        bank.weeks = list(obj["weeks"])
        st, tt = obj["summaries_t"], obj["templates_t"]
        bank.summaries = list(st.unbind(0)) if st is not None else []
        bank.templates = list(tt.unbind(0)) if tt is not None else []
        bank._summaries_t = st
        bank._templates_t = tt
        bank._dirty = False
        return bank


# ---------------------------------------------------------------------------
# Retrieval module
# ---------------------------------------------------------------------------

class CrossPatientRetrieval(nn.Module):
    """Retrieve top-K templates from a bank and emit projected, gated tokens
    to concatenate onto M_pre.

    Gate = sigmoid(gate_logit + gate_init). Initialized so sigmoid(gate_init)
    is the starting scale (gate_init=0 → 0.5).
    """

    def __init__(
        self,
        d_model: int = 256,
        k: int = 5,
        gate_init: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.k = k

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("gate_init", torch.tensor(float(gate_init)))

        nn.init.xavier_uniform_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)

    def current_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit + self.gate_init)

    def forward(
        self,
        query_pre_summary: torch.Tensor,
        bank: CrossPatientBank,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_pre_summary: (B, C)
            bank: CrossPatientBank (non-empty)
        Returns:
            retrieved_memory: (B, K*N_tokens, C)
            gate: (B, 1)
        """
        if len(bank) == 0:
            raise RuntimeError("CrossPatientBank is empty; cannot retrieve.")

        B, C = query_pre_summary.shape
        _, _, templates = bank.topk(query_pre_summary, k=self.k)     # (B, K, Nt, C)
        _, K, Nt, _ = templates.shape

        tokens = self.proj(templates.reshape(B, K * Nt, C))
        gate = self.current_gate().expand(B, 1).to(tokens.dtype)
        return tokens * gate.unsqueeze(-1), gate


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    bank = CrossPatientBank()
    for i in range(10):
        bank.add(f"p{i}", torch.randn(256), torch.randn(16, 256), weeks_elapsed=3)
    assert len(bank) == 10

    retr = CrossPatientRetrieval(256, k=3)
    q = torch.randn(2, 256)
    tokens, gate = retr(q, bank)
    print("tokens:", tokens.shape, "gate:", gate.shape)
    assert tokens.shape == (2, 3 * 16, 256)
    assert gate.shape == (2, 1)

    pre_feat = torch.randn(1, 4, 256, 8, 8)
    mid_feat = torch.randn(1, 4, 256, 8, 8)
    pre_mask = (torch.rand(1, 4, 1, 8, 8) > 0.7).float()
    mid_mask = (torch.rand(1, 4, 1, 8, 8) > 0.7).float()
    summ = compute_pre_summary(pre_feat, pre_mask)
    tmpl = compute_change_template(pre_feat, mid_feat, pre_mask, mid_mask, n_tokens=16)
    print("summary:", summ.shape, "template:", tmpl.shape)
    assert summ.shape == (1, 256) and tmpl.shape == (1, 16, 256)

    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "bank.pt")
        bank.save(p)
        bank2 = CrossPatientBank.load(p)
        assert len(bank2) == len(bank)
        tokens2, _ = retr(q, bank2)
        assert tokens2.shape == tokens.shape

    print("retrieval.py smoke test OK")
