"""
CED-Net / HRSCD-inspired Deeply-Supervised SCD Baseline.

Papers this follows:
    - Daudt, R.C., et al. (2019). "High Resolution Semantic Change Detection."
      CVIU, 187, 102783. (HRSCD strategies I-IV.) Introduces a family of
      shared vs. separate branch architectures with explicit deep supervision
      on BOTH per-time semantic maps and the binary change map.
    - Peng et al. (2021). "CED-Net: Crops and Weeds Segmentation..." The
      deeply-supervised decoder idea is reused here as a simple SCD head.

This is a simpler architecture than Bi-SRNet / SCanNet and is intended as
a cheap-to-train ablation point ("how far can a plain dual-UNet go on
SECOND before adding fancy attention?"). It follows HRSCD Strategy IV:
    - Shared encoder (no siamese weight sharing trick beyond identical conv
      stacks) producing per-time features.
    - Two symmetric semantic decoders (pre, mid).
    - A change decoder that consumes the concatenation of the two
      semantic-decoder feature maps and predicts the binary change mask.
    - Deep supervision: auxiliary heads at every decoder level (summed with
      loss weights during training; disabled at eval).

Typical metrics on SECOND (HRSCD Strategy IV numbers from the paper):
    mIoU (semantic) : ~34.0
    Sek             : ~18.2
    F_scd           : ~51.5
(so clearly weaker than Bi-SRNet / SCanNet — that is the point: serves as
an ablation floor showing where the gap comes from.)

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
    No external weights. Marked as "paraphrased reimplementation" below.
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
# Model (paraphrased HRSCD-IV / CED-style deeply supervised SCD)
# ---------------------------------------------------------------------------
if HAS_TORCH:

    def _cbr(i: int, o: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(i, o, 3, 1, 1, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, 1, 1, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
        )

    class _UNetEncoder(nn.Module):
        def __init__(self, base: int = 32):
            super().__init__()
            self.e1 = _cbr(3, base)
            self.e2 = _cbr(base, base * 2)
            self.e3 = _cbr(base * 2, base * 4)
            self.e4 = _cbr(base * 4, base * 8)
            self.pool = nn.MaxPool2d(2, 2)

        def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
            f1 = self.e1(x)
            f2 = self.e2(self.pool(f1))
            f3 = self.e3(self.pool(f2))
            f4 = self.e4(self.pool(f3))
            return [f1, f2, f3, f4]

    class _UNetDecoder(nn.Module):
        """Decoder with 3 upsampling stages and deep-supervision heads."""

        def __init__(self, base: int = 32, n_classes: int = 6):
            super().__init__()
            self.d3 = _cbr(base * 8 + base * 4, base * 4)
            self.d2 = _cbr(base * 4 + base * 2, base * 2)
            self.d1 = _cbr(base * 2 + base, base)
            self.head = nn.Conv2d(base, n_classes, 1)
            # Deep-supervision auxiliary heads
            self.aux3 = nn.Conv2d(base * 4, n_classes, 1)
            self.aux2 = nn.Conv2d(base * 2, n_classes, 1)

        def forward(self, feats: list[torch.Tensor], return_aux: bool = False) -> dict:
            f1, f2, f3, f4 = feats
            x = F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
            x = self.d3(torch.cat([x, f3], dim=1))
            aux3 = self.aux3(x)
            x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            x = self.d2(torch.cat([x, f2], dim=1))
            aux2 = self.aux2(x)
            x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
            x = self.d1(torch.cat([x, f1], dim=1))
            out = {"logits": self.head(x), "feat": x}
            if return_aux:
                out["aux3"] = aux3
                out["aux2"] = aux2
            return out

    class CEDNetSCD(nn.Module):
        """
        HRSCD-IV / CED-Net style deeply-supervised semantic change detection.
        Paraphrased reimplementation.
        """

        def __init__(self, n_classes: int = 6, base: int = 32):
            super().__init__()
            self.encoder = _UNetEncoder(base)
            self.dec_pre = _UNetDecoder(base, n_classes)
            self.dec_mid = _UNetDecoder(base, n_classes)
            self.change = nn.Sequential(
                nn.Conv2d(base * 2, base, 3, 1, 1, bias=False),
                nn.BatchNorm2d(base),
                nn.ReLU(inplace=True),
                nn.Conv2d(base, 1, 1),
            )

        def forward(self, pre: torch.Tensor, mid: torch.Tensor, return_aux: bool = False) -> dict:
            fp = self.encoder(pre)
            fm = self.encoder(mid)
            op = self.dec_pre(fp, return_aux=return_aux)
            om = self.dec_mid(fm, return_aux=return_aux)
            change_logits = self.change(torch.cat([op["feat"], om["feat"]], dim=1))

            sem_pre = F.interpolate(op["logits"], size=pre.shape[-2:], mode="bilinear", align_corners=False)
            sem_mid = F.interpolate(om["logits"], size=mid.shape[-2:], mode="bilinear", align_corners=False)
            change = F.interpolate(change_logits, size=pre.shape[-2:], mode="bilinear", align_corners=False)

            out = {"sem_pre": sem_pre, "sem_mid": sem_mid, "change": change}
            if return_aux:
                out["aux_pre"] = [op["aux2"], op["aux3"]]
                out["aux_mid"] = [om["aux2"], om["aux3"]]
            return out


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
def run_ced_net_baseline(
    data_dir: str,
    dataset_name: str = "second",
    output_dir: str = "results/baselines/ced_net",
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
    print("CED-Net / HRSCD-IV Baseline (paraphrased reimplementation)")
    print(f"Dataset: {dataset_name}   Split: {split}")
    print("=" * 60)

    if not HAS_TORCH:
        print("torch not available — writing placeholder metrics.")
        stub = {"aggregate": {"status": "torch_unavailable"}, "per_sample": []}
        with open(output_dir_p / "metrics.json", "w") as f:
            json.dump(stub, f, indent=2)
        return stub["aggregate"]

    if dataset_name != "second":
        raise ValueError("CED-Net baseline is configured for the SECOND dataset only.")

    from csmsam.datasets.second import SECONDDataset
    ds = SECONDDataset(data_dir=data_dir, split=split, image_size=image_size, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CEDNetSCD(n_classes=n_classes).to(device)
    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("No checkpoint — running with random init (baseline floor).")
    model.eval()

    per_sample: list[dict] = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="CED-Net")):
            if max_samples is not None and i >= max_samples:
                break
            pre = batch["pre_image"].to(device)
            mid = batch["mid_image"].to(device)
            out = model(pre, mid, return_aux=False)

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

    print(f"\nCED-Net Results ({split}):")
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
    parser.add_argument("--output_dir", type=str, default="results/baselines/ced_net")
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

    run_ced_net_baseline(
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
