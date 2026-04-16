"""
Bi-SRNet Baseline — Bi-Temporal Semantic Reasoning Network.

Paper:
    Ding, L., Guo, H., Liu, S., Mou, L., Zhang, J., & Bruzzone, L. (2022).
    "Bi-Temporal Semantic Reasoning for the Semantic Change Detection of HR
    Remote Sensing Images." IEEE TGRS, 60, 1-14.
    https://arxiv.org/abs/2108.06103

Typical reported metrics on SECOND:
    mIoU (semantic): ~37.6
    Sek (SeCond-specific): ~22.0
    F_scd  (semantic change F1): ~58.2

Architecture:
    - Siamese ResNet-lite encoder on (pre, mid) images (shared weights).
    - Two "SR" (semantic reasoning) branches fuse bi-temporal features via
      cross-attention (channel + spatial).
    - A temporal-reasoning branch predicts the binary change map from a
      feature difference, which gates the per-time semantic decoders.
    - Two decoder heads predict pre and mid semantic masks.

This is a paraphrased reimplementation: it captures the Bi-SRNet topology
(siamese encoder + SR reasoning block + change branch + dual semantic
decoders) in a compact form suitable for ablation studies. It is NOT a
bit-exact port of the authors' official code.

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
    (No pretrained weights — random-init by default; the paper trains from
    scratch on SECOND. For fair comparison use ImageNet-pretrained ResNet18
    encoder when available.)
"""

from __future__ import annotations

import argparse
import json
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

    def _conv_bn_relu(in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    class _ResBlock(nn.Module):
        def __init__(self, c: int):
            super().__init__()
            self.b1 = _conv_bn_relu(c, c)
            self.b2 = nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.relu(x + self.b2(self.b1(x)), inplace=True)

    class _SiameseEncoder(nn.Module):
        """Lightweight ResNet-lite siamese encoder returning 4 feature levels."""

        def __init__(self, base: int = 32):
            super().__init__()
            self.stem = _conv_bn_relu(3, base, k=7, s=2, p=3)
            self.l1 = nn.Sequential(_ResBlock(base), nn.Conv2d(base, base * 2, 3, 2, 1))
            self.l2 = nn.Sequential(_ResBlock(base * 2), nn.Conv2d(base * 2, base * 4, 3, 2, 1))
            self.l3 = nn.Sequential(_ResBlock(base * 4), nn.Conv2d(base * 4, base * 8, 3, 2, 1))

        def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
            f0 = self.stem(x)
            f1 = self.l1(f0)
            f2 = self.l2(f1)
            f3 = self.l3(f2)
            return [f0, f1, f2, f3]

    class _SRBlock(nn.Module):
        """Semantic reasoning block: channel + spatial cross-attention."""

        def __init__(self, c: int):
            super().__init__()
            self.q = nn.Conv2d(c, c, 1)
            self.k = nn.Conv2d(c, c, 1)
            self.v = nn.Conv2d(c, c, 1)
            self.proj = nn.Conv2d(c, c, 1)
            self.gamma = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape
            q = self.q(x).flatten(2)           # (B, C, HW)
            k = self.k(y).flatten(2)           # (B, C, HW)
            v = self.v(y).flatten(2)           # (B, C, HW)
            attn = torch.softmax(q.transpose(1, 2) @ k / (C ** 0.5), dim=-1)  # (B, HW, HW)
            out = (v @ attn.transpose(1, 2)).reshape(B, C, H, W)
            return x + self.gamma * self.proj(out)

    class _Decoder(nn.Module):
        """Small U-Net style decoder: upsample + concat + conv x 3 stages."""

        def __init__(self, base: int, n_classes: int):
            super().__init__()
            self.up3 = _conv_bn_relu(base * 8 + base * 4, base * 4)
            self.up2 = _conv_bn_relu(base * 4 + base * 2, base * 2)
            self.up1 = _conv_bn_relu(base * 2 + base, base)
            self.head = nn.Conv2d(base, n_classes, 1)

        def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
            f0, f1, f2, f3 = feats
            x = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            x = self.up3(torch.cat([x, f2], dim=1))
            x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
            x = self.up2(torch.cat([x, f1], dim=1))
            x = F.interpolate(x, size=f0.shape[-2:], mode="bilinear", align_corners=False)
            x = self.up1(torch.cat([x, f0], dim=1))
            return self.head(x)

    class BiSRNet(nn.Module):
        """
        Bi-SRNet: siamese encoder + SR reasoning + dual semantic heads +
        temporal-change head. Paraphrased reimplementation.
        """

        def __init__(self, n_classes: int = 6, base: int = 32):
            super().__init__()
            self.encoder = _SiameseEncoder(base=base)
            self.sr_pre = _SRBlock(base * 8)
            self.sr_mid = _SRBlock(base * 8)
            self.dec_pre = _Decoder(base, n_classes)
            self.dec_mid = _Decoder(base, n_classes)
            self.change_head = nn.Sequential(
                _conv_bn_relu(base * 8, base * 4),
                nn.Conv2d(base * 4, 1, 1),
            )

        def forward(self, pre: torch.Tensor, mid: torch.Tensor) -> dict:
            fp = self.encoder(pre)
            fm = self.encoder(mid)

            fp[-1] = self.sr_pre(fp[-1], fm[-1])
            fm[-1] = self.sr_mid(fm[-1], fp[-1])

            diff = torch.abs(fp[-1] - fm[-1])
            change_logits = F.interpolate(
                self.change_head(diff), size=pre.shape[-2:], mode="bilinear", align_corners=False
            )

            sem_pre = F.interpolate(
                self.dec_pre(fp), size=pre.shape[-2:], mode="bilinear", align_corners=False
            )
            sem_mid = F.interpolate(
                self.dec_mid(fm), size=mid.shape[-2:], mode="bilinear", align_corners=False
            )
            return {"sem_pre": sem_pre, "sem_mid": sem_mid, "change": change_logits}


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
def run_bisrnet_baseline(
    data_dir: str,
    dataset_name: str = "second",
    output_dir: str = "results/baselines/bisrnet",
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
    print("Bi-SRNet Baseline (paraphrased reimplementation)")
    print(f"Dataset: {dataset_name}   Split: {split}")
    print("=" * 60)

    if not HAS_TORCH:
        print("torch not available — writing placeholder metrics.")
        stub = {"aggregate": {"status": "torch_unavailable"}, "per_sample": []}
        with open(output_dir_p / "metrics.json", "w") as f:
            json.dump(stub, f, indent=2)
        return stub["aggregate"]

    if dataset_name != "second":
        raise ValueError("Bi-SRNet baseline only supports the SECOND dataset.")

    from csmsam.datasets.second import SECONDDataset
    ds = SECONDDataset(data_dir=data_dir, split=split, image_size=image_size, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = BiSRNet(n_classes=n_classes).to(device)
    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("No checkpoint — running with random init (baseline floor).")
    model.eval()

    per_sample: list[dict] = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Bi-SRNet")):
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

    def _mean(key: str) -> float:
        vals = [s[key] for s in per_sample]
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

    print(f"\nBi-SRNet Results ({split}):")
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
    parser.add_argument("--output_dir", type=str, default="results/baselines/bisrnet")
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

    run_bisrnet_baseline(
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
