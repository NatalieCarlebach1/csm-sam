"""
SCanNet Baseline — Joint Spatio-Temporal Transformer for SCD.

Paper:
    Ding, L., Zhang, J., Guo, H., Zhang, K., Liu, B., & Bruzzone, L. (2024).
    "Joint Spatio-Temporal Modeling for Semantic Change Detection in Remote
    Sensing Images." IEEE TGRS, 62, 1-14.
    https://arxiv.org/abs/2212.05245

Typical reported metrics on SECOND (as reported in paper):
    mIoU (semantic): ~38.5
    Sek            : ~23.9
    F_scd          : ~59.8   (SOTA at time of publication)

Architecture idea:
    - Shared siamese encoder produces per-time feature pyramid.
    - A Spatio-Temporal Transformer (ST-Transformer) jointly attends over
      BOTH time steps concatenated along the token axis, enabling joint
      modeling instead of two-pass reasoning.
    - Per-time semantic heads + a shared change head operate on the fused
      tokens, projected back to dense feature maps.

This is a paraphrased reimplementation following the high-level design:
    encoder -> ST-Transformer -> dual semantic heads + change head.
It is NOT bit-exact with the official SCanNet release.

Uniqueness note
---------------
Bi-SRNet / SCanNet specialize to bitemporal semantic change with small
encoders. CSM-SAM's cross-session memory module is ARCHITECTURALLY GENERAL:
it slots atop any SAM2 backbone and uses token-level attention between
sessions, not branch-differencing. On SECOND, this means the same module
that solves HNTS-MRG medical tumor propagation also propagates land-cover
semantics.

Install / weights:
    pip install torch torchvision
    No official weights are redistributed here. Train with Dice + CE on
    SECOND, or use paper-released weights placed at --checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kw):
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
if HAS_TORCH:

    def _cbr(i: int, o: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(i, o, k, s, p, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
        )

    class _Encoder(nn.Module):
        """Compact siamese CNN encoder — 4 levels, /16 at the deepest."""

        def __init__(self, base: int = 48):
            super().__init__()
            self.s1 = _cbr(3, base, k=7, s=2, p=3)          # /2
            self.s2 = nn.Sequential(_cbr(base, base * 2, s=2), _cbr(base * 2, base * 2))       # /4
            self.s3 = nn.Sequential(_cbr(base * 2, base * 4, s=2), _cbr(base * 4, base * 4))   # /8
            self.s4 = nn.Sequential(_cbr(base * 4, base * 8, s=2), _cbr(base * 8, base * 8))   # /16

        def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
            f1 = self.s1(x)
            f2 = self.s2(f1)
            f3 = self.s3(f2)
            f4 = self.s4(f3)
            return [f1, f2, f3, f4]

    class _PosEnc2D(nn.Module):
        """Fixed 2D sin-cos positional encoding."""

        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim

        def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
            y = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)
            x = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)
            div = torch.exp(torch.arange(0, self.dim // 2, 2, device=device) *
                            -(math.log(10000.0) / (self.dim // 2)))
            pe = torch.zeros(self.dim, H, W, device=device)
            pe[0::4] = torch.sin(x.unsqueeze(0) * div.view(-1, 1, 1))[: self.dim // 4]
            pe[1::4] = torch.cos(x.unsqueeze(0) * div.view(-1, 1, 1))[: self.dim // 4]
            pe[2::4] = torch.sin(y.unsqueeze(0) * div.view(-1, 1, 1))[: self.dim // 4]
            pe[3::4] = torch.cos(y.unsqueeze(0) * div.view(-1, 1, 1))[: self.dim // 4]
            return pe

    class _STTransformerLayer(nn.Module):
        """One Spatio-Temporal transformer block: self-attn over concat tokens."""

        def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 2.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                nn.GELU(),
                nn.Linear(int(dim * mlp_ratio), dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm1(x)
            a, _ = self.attn(h, h, h, need_weights=False)
            x = x + a
            x = x + self.mlp(self.norm2(x))
            return x

    class SCanNet(nn.Module):
        """
        SCanNet paraphrased reimplementation: siamese encoder -> stacked
        ST-Transformer over concatenated (pre, mid) tokens -> decoders.
        """

        def __init__(self, n_classes: int = 6, base: int = 48, n_layers: int = 4, heads: int = 4):
            super().__init__()
            self.encoder = _Encoder(base=base)
            self.proj = nn.Conv2d(base * 8, base * 8, 1)
            self.pos_enc = _PosEnc2D(base * 8)
            self.time_embed = nn.Parameter(torch.randn(2, base * 8))
            self.layers = nn.ModuleList([_STTransformerLayer(base * 8, heads) for _ in range(n_layers)])

            # Decoder: 1x1 project + upsample/concat with encoder skips
            self.dec4 = _cbr(base * 8, base * 4)
            self.dec3 = _cbr(base * 4 + base * 4, base * 2)
            self.dec2 = _cbr(base * 2 + base * 2, base)
            self.dec1 = _cbr(base + base, base)

            self.head_pre = nn.Conv2d(base, n_classes, 1)
            self.head_mid = nn.Conv2d(base, n_classes, 1)
            self.head_change = nn.Conv2d(base * 2, 1, 1)

        def _decode(self, f4: torch.Tensor, feats: list[torch.Tensor]) -> torch.Tensor:
            x = self.dec4(f4)
            x = F.interpolate(x, size=feats[2].shape[-2:], mode="bilinear", align_corners=False)
            x = self.dec3(torch.cat([x, feats[2]], dim=1))
            x = F.interpolate(x, size=feats[1].shape[-2:], mode="bilinear", align_corners=False)
            x = self.dec2(torch.cat([x, feats[1]], dim=1))
            x = F.interpolate(x, size=feats[0].shape[-2:], mode="bilinear", align_corners=False)
            x = self.dec1(torch.cat([x, feats[0]], dim=1))
            return x

        def forward(self, pre: torch.Tensor, mid: torch.Tensor) -> dict:
            fp = self.encoder(pre)
            fm = self.encoder(mid)

            p4, m4 = self.proj(fp[-1]), self.proj(fm[-1])
            B, C, H, W = p4.shape

            pe = self.pos_enc(H, W, p4.device).unsqueeze(0)                # (1, C, H, W)
            p_tok = (p4 + pe).flatten(2).transpose(1, 2) + self.time_embed[0]
            m_tok = (m4 + pe).flatten(2).transpose(1, 2) + self.time_embed[1]
            toks = torch.cat([p_tok, m_tok], dim=1)                        # (B, 2HW, C)

            for layer in self.layers:
                toks = layer(toks)

            p_back = toks[:, : H * W].transpose(1, 2).reshape(B, C, H, W)
            m_back = toks[:, H * W:].transpose(1, 2).reshape(B, C, H, W)

            dec_pre = self._decode(p_back, fp)
            dec_mid = self._decode(m_back, fm)

            sem_pre = F.interpolate(self.head_pre(dec_pre), size=pre.shape[-2:],
                                    mode="bilinear", align_corners=False)
            sem_mid = F.interpolate(self.head_mid(dec_mid), size=mid.shape[-2:],
                                    mode="bilinear", align_corners=False)
            change = F.interpolate(
                self.head_change(torch.cat([dec_pre, dec_mid], dim=1)),
                size=pre.shape[-2:], mode="bilinear", align_corners=False,
            )
            return {"sem_pre": sem_pre, "sem_mid": sem_mid, "change": change}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _binary_iou_f1(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    p = pred.astype(bool).reshape(-1)
    t = target.astype(bool).reshape(-1)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    iou = inter / union if union > 0 else 0.0
    tp = inter
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return float(iou), float(f1)


def _semantic_miou(pred: np.ndarray, target: np.ndarray, n_classes: int) -> float:
    ious = []
    for c in range(n_classes):
        p = (pred == c)
        t = (target == c)
        union = np.logical_or(p, t).sum()
        if union == 0:
            continue
        ious.append(np.logical_and(p, t).sum() / union)
    return float(np.mean(ious)) if ious else 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_scannet_cd_baseline(
    data_dir: str,
    dataset_name: str = "second",
    output_dir: str = "results/baselines/scannet_cd",
    split: str = "test",
    device: str = "cuda",
    n_classes: int = 6,
    checkpoint: str | None = None,
    image_size: int = 512,
    batch_size: int = 1,
    max_samples: int | None = None,
):
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SCanNet Baseline (paraphrased reimplementation)")
    print(f"Dataset: {dataset_name}   Split: {split}")
    print("=" * 60)

    if not HAS_TORCH:
        print("torch not available — writing placeholder metrics.")
        stub = {"aggregate": {"status": "torch_unavailable"}, "per_sample": []}
        with open(output_dir_p / "metrics.json", "w") as f:
            json.dump(stub, f, indent=2)
        return stub["aggregate"]

    if dataset_name != "second":
        raise ValueError("SCanNet baseline only supports the SECOND dataset.")

    from csmsam.datasets.second import SECONDDataset
    ds = SECONDDataset(data_dir=data_dir, split=split, image_size=image_size, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SCanNet(n_classes=n_classes).to(device)
    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("No checkpoint — running with random init (baseline floor).")
    model.eval()

    per_sample: list[dict] = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="SCanNet")):
            if max_samples is not None and i >= max_samples:
                break
            pre = batch["pre_image"].to(device)
            mid = batch["mid_image"].to(device)
            out = model(pre, mid)

            sem_pre_pred = out["sem_pre"].argmax(1).cpu().numpy()[0]
            sem_mid_pred = out["sem_mid"].argmax(1).cpu().numpy()[0]
            change_pred = (torch.sigmoid(out["change"]) > 0.5).cpu().numpy()[0, 0]

            sem_pre_gt = batch["pre_mask_semantic"].numpy()[0]
            sem_mid_gt = batch["mid_mask_semantic"].numpy()[0]
            change_gt = batch["change_mask"].numpy()[0, 0] > 0.5

            miou_pre = _semantic_miou(sem_pre_pred, sem_pre_gt, n_classes)
            miou_mid = _semantic_miou(sem_mid_pred, sem_mid_gt, n_classes)
            bin_iou, _ = _binary_iou_f1((sem_mid_pred > 0).astype(np.uint8),
                                        (sem_mid_gt > 0).astype(np.uint8))
            ch_iou, ch_f1 = _binary_iou_f1(change_pred, change_gt)

            per_sample.append({
                "image_name": batch["image_name"][0],
                "miou_pre": miou_pre,
                "miou_mid": miou_mid,
                "binary_iou_mid": bin_iou,
                "change_iou": ch_iou,
                "change_f1": ch_f1,
            })

    def _mean(k: str) -> float:
        vals = [s[k] for s in per_sample]
        return float(np.mean(vals)) if vals else 0.0

    agg = {
        "n_samples": len(per_sample),
        "miou_pre_mean": _mean("miou_pre"),
        "miou_mid_mean": _mean("miou_mid"),
        "miou_mean": 0.5 * (_mean("miou_pre") + _mean("miou_mid")),
        "binary_iou_mid_mean": _mean("binary_iou_mid"),
        "change_iou_mean": _mean("change_iou"),
        "change_f1_mean": _mean("change_f1"),
    }

    print(f"\nSCanNet Results ({split}):")
    for k, v in agg.items():
        print(f"  {k:<22} : {v:.4f}" if isinstance(v, float) else f"  {k:<22} : {v}")

    results = {"aggregate": agg, "per_sample": per_sample}
    with open(output_dir_p / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {output_dir_p}/metrics.json")
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw/SECOND/data")
    parser.add_argument("--dataset_name", type=str, default="second")
    parser.add_argument("--output_dir", type=str, default="results/baselines/scannet_cd")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    if HAS_TORCH and args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_scannet_cd_baseline(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        split=args.split,
        device=args.device,
        n_classes=args.n_classes,
        checkpoint=args.checkpoint,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
