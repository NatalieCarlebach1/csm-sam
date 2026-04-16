"""
TinyCD Baseline — Compact, Efficient Change Detection.

Paper      : "TINYCD: A (Not So) Deep Learning Model for Change Detection"
             A. Codegoni, G. Lombardi, A. Ferrari — arXiv 2022.
Typical F1 : ~89 on LEVIR-CD (256x256) with <0.3M parameters.
Install    : No external weights; paraphrased reimplementation. Canonical
             code at https://github.com/AndreaCodegoni/Tiny_model_4_CD.

Architecture summary
--------------------
A MobileNetV2-style siamese backbone extracts multi-scale features. A
Mixing and Attention Mask Block (MAMB) on each scale blends the pair:
concatenation → 1x1 conv → channel attention, producing a "temporal
mix" feature. The mix features from all scales are progressively
upsampled and summed, then a 1x1 conv head outputs the change map.
Designed to be tiny (<0.3M params) yet competitive with heavyweight
CD models.

Uniqueness vs CSM-SAM
---------------------
CSM-SAM inherits SAM2's 1B-mask-pretrained ViT-H and trains only ~2M
parameters of cross-session memory attention on top of a frozen encoder;
TinyCD trains a sub-1M-parameter network end-to-end from scratch. The
design philosophies are opposite: TinyCD achieves efficiency by
avoiding pretraining and using lightweight attention on tiny features,
while CSM-SAM leverages a massive pretrained encoder with per-token
cross-time query/key attention — yielding richer spatial routing
between pre- and mid-timepoint features than TinyCD's channel-wise
mixing blocks.

Paraphrased reimplementation from the paper — not an exact reproduction
of the authors' code. Intended as a reference baseline for CSM-SAM.
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
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


if _HAS_TORCH:

    class _DWBlock(nn.Module):
        """MobileNetV2-lite depthwise-separable + pointwise block."""

        def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
            super().__init__()
            self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                                groups=in_ch, bias=False)
            self.bn1 = nn.BatchNorm2d(in_ch)
            self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.act = nn.ReLU6(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.act(self.bn1(self.dw(x)))
            return self.act(self.bn2(self.pw(x)))


    class _TinyBackbone(nn.Module):
        """Ultra-light siamese trunk returning 3 multi-scale feature maps."""

        def __init__(self, in_channels: int = 3):
            super().__init__()
            self.stage1 = nn.Sequential(
                _DWBlock(in_channels, 16, stride=2),
                _DWBlock(16, 16),
            )
            self.stage2 = nn.Sequential(
                _DWBlock(16, 24, stride=2),
                _DWBlock(24, 24),
            )
            self.stage3 = nn.Sequential(
                _DWBlock(24, 32, stride=2),
                _DWBlock(32, 32),
            )
            self.channels = (16, 24, 32)

        def forward(self, x: torch.Tensor):
            s1 = self.stage1(x)
            s2 = self.stage2(s1)
            s3 = self.stage3(s2)
            return s1, s2, s3


    class _MAMB(nn.Module):
        """Mixing and Attention Mask Block: concat → 1x1 → channel attention."""

        def __init__(self, dim: int, reduction: int = 4):
            super().__init__()
            hidden = max(1, dim // reduction)
            self.mix = nn.Sequential(
                nn.Conv2d(dim * 2, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            )
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, hidden, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, dim, 1, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            m = self.mix(torch.cat([a, b], dim=1))
            return m * self.gate(m)


    class TinyCD(nn.Module):
        """
        TinyCD change detection network (<1M params).

        Paraphrased reimplementation of Codegoni et al. 2022. Input: two
        RGB images (B, 3, H, W). Output: change logits (B, 1, H, W).
        """

        def __init__(self, in_channels: int = 3):
            super().__init__()
            self.backbone = _TinyBackbone(in_channels)
            cs = self.backbone.channels

            self.mambs = nn.ModuleList([_MAMB(c) for c in cs])

            # Top-down decoder: upsample + sum with finer mix features
            self.reduce = nn.ModuleList([
                nn.Conv2d(cs[-1], cs[-2], 1),
                nn.Conv2d(cs[-2], cs[-3], 1),
            ])
            self.head = nn.Sequential(
                nn.Conv2d(cs[0], cs[0], 3, padding=1),
                nn.BatchNorm2d(cs[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(cs[0], 1, 1),
            )

        def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            H, W = x1.shape[-2:]
            a1, a2, a3 = self.backbone(x1)
            b1, b2, b3 = self.backbone(x2)

            m1 = self.mambs[0](a1, b1)  # high-res
            m2 = self.mambs[1](a2, b2)
            m3 = self.mambs[2](a3, b3)  # low-res

            # Top-down merge
            u = F.interpolate(self.reduce[0](m3), size=m2.shape[-2:],
                              mode="bilinear", align_corners=False) + m2
            u = F.interpolate(self.reduce[1](u), size=m1.shape[-2:],
                              mode="bilinear", align_corners=False) + m1

            logits = self.head(u)
            return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _build_dataset(dataset_name: str, data_dir: str, split: str, image_size: int):
    name = dataset_name.lower()
    if name == "levir_cd":
        from csmsam.datasets import LEVIRCDDataset
        return LEVIRCDDataset(data_dir, split=split, image_size=image_size, augment=False)
    if name == "s2looking":
        from csmsam.datasets import S2LookingDataset
        return S2LookingDataset(data_dir, split=split, image_size=image_size, augment=False)
    if name == "second":
        from csmsam.datasets import SECONDDataset
        return SECONDDataset(data_dir, split=split, image_size=image_size, augment=False)
    if name == "xbd":
        from csmsam.datasets import XBDDataset
        return XBDDataset(data_dir, split=split, image_size=image_size, augment=False)
    raise ValueError(
        f"Unknown dataset '{dataset_name}'. Expected one of: "
        "levir_cd, s2looking, second, xbd."
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _binary_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    p = (pred > 0).astype(np.uint8).reshape(-1)
    t = (target > 0).astype(np.uint8).reshape(-1)
    tp = int(((p == 1) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    tn = int(((p == 0) & (t == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "iou": iou}


def _aggregate(per_sample: list[dict]) -> dict:
    if not per_sample:
        return {}
    tp = sum(m["tp"] for m in per_sample)
    fp = sum(m["fp"] for m in per_sample)
    fn = sum(m["fn"] for m in per_sample)
    tn = sum(m["tn"] for m in per_sample)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {
        "precision": prec, "recall": rec, "f1": f1, "iou": iou,
        "n_samples": len(per_sample), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_tinycd_baseline(
    data_dir: str,
    dataset_name: str = "levir_cd",
    output_dir: str = "results/baselines/tinycd",
    split: str = "test",
    device: str = "cuda",
    image_size: int = 256,
    threshold: float = 0.5,
    checkpoint: str | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"TinyCD Baseline | dataset={dataset_name} | split={split}")
    print("=" * 60)

    dataset = _build_dataset(dataset_name, data_dir, split, image_size)

    if not _HAS_TORCH:
        print("Warning: torch not available — emitting random predictions.")
        model = None
    else:
        model = TinyCD().to(device)
        if checkpoint and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location=device)
            model.load_state_dict(state.get("model", state), strict=False)
            print(f"Loaded weights from {checkpoint}")
        else:
            print("No checkpoint provided: running with randomly initialized weights.")
        model.eval()

    per_sample_metrics = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False) if _HAS_TORCH else range(len(dataset))

    for idx, batch in enumerate(loader):
        if _HAS_TORCH:
            pre = batch["pre_image"].to(device)
            mid = batch["mid_image"].to(device)
            gt = batch["change_mask"].squeeze().cpu().numpy()
            with torch.no_grad():
                logits = model(pre, mid)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred = (prob > threshold).astype(np.uint8)
        else:
            sample = dataset[idx]
            gt = sample["change_mask"].squeeze().numpy()
            pred = (np.random.rand(*gt.shape) > 0.95).astype(np.uint8)

        m = _binary_metrics(pred, gt)
        m["sample_index"] = int(idx)
        per_sample_metrics.append(m)

    agg = _aggregate(per_sample_metrics)
    print(f"\nTinyCD Results ({dataset_name}/{split}):")
    print(f"  F1        : {agg.get('f1', 0):.4f}")
    print(f"  IoU       : {agg.get('iou', 0):.4f}")
    print(f"  Precision : {agg.get('precision', 0):.4f}")
    print(f"  Recall    : {agg.get('recall', 0):.4f}")

    results = {
        "model": "tinycd",
        "dataset": dataset_name,
        "split": split,
        "aggregate": agg,
        "per_sample": per_sample_metrics,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {output_dir}/metrics.json")
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="levir_cd",
                        choices=["levir_cd", "s2looking", "second", "xbd"])
    parser.add_argument("--output_dir", type=str, default="results/baselines/tinycd")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if _HAS_TORCH and args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_tinycd_baseline(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        split=args.split,
        device=args.device,
        image_size=args.image_size,
        threshold=args.threshold,
        checkpoint=args.checkpoint,
    )
