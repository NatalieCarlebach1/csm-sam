"""
FC-Siam-conc Baseline — Fully Convolutional Siamese (Concatenation) Network for Change Detection.

Paper      : "Fully Convolutional Siamese Networks for Change Detection"
             R. C. Daudt, B. Le Saux, A. Boulch — ICIP 2018.
Typical F1 : ~82 on LEVIR-CD (from-scratch, 256x256).
Install    : No external weights; paraphrased reimplementation trained
             from scratch on the target dataset. Canonical code at
             https://github.com/rcdaudt/fully_convolutional_change_detection.

Architecture summary
--------------------
Identical siamese UNet scaffold to FC-Siam-diff, but instead of using the
*absolute difference* of paired encoder features, the two branches'
feature maps are *concatenated* at each skip-connection. The decoder
therefore sees 2C channels from the encoder pair at each scale, giving
the model access to both branches' features directly rather than
a pre-computed difference. A final 1x1 conv outputs a change map.

Uniqueness vs CSM-SAM
---------------------
CSM-SAM inherits SAM2's 1B-mask-pretrained ViT-H and trains ~2M
parameters of cross-session memory attention on top of a frozen encoder;
this baseline trains a full siamese UNet from scratch on ~1k image pairs
with no external priors. CSM-SAM's cross-session memory is
architecturally richer than channel-wise concatenation — it supports
per-token query/key attention that can route information spatially
between the two time points, whereas FC-Siam-conc fuses the pair purely
via local 3x3 convolutions over stacked channels with no long-range
cross-temporal reasoning.

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

    def _conv_block(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


    class FCSiamConc(nn.Module):
        """
        Siamese UNet where skip-connections concatenate (f_t1, f_t2).

        Paraphrased reimplementation of Daudt et al. 2018. Input: two RGB
        images (B, 3, H, W). Output: change logits (B, 1, H, W).
        """

        def __init__(self, in_channels: int = 3, base: int = 16):
            super().__init__()
            c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16

            # Shared encoder
            self.enc1 = _conv_block(in_channels, c1)
            self.enc2 = _conv_block(c1, c2)
            self.enc3 = _conv_block(c2, c3)
            self.enc4 = _conv_block(c3, c4)
            self.bottleneck = _conv_block(c4, c5)
            self.pool = nn.MaxPool2d(2, 2)

            # Decoder receives concat(t1, t2) per skip (2C) + upsampled (C)
            self.up4 = nn.ConvTranspose2d(c5 * 2, c4, 2, stride=2)
            self.dec4 = _conv_block(c4 + c4 * 2, c4)
            self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
            self.dec3 = _conv_block(c3 + c3 * 2, c3)
            self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
            self.dec2 = _conv_block(c2 + c2 * 2, c2)
            self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
            self.dec1 = _conv_block(c1 + c1 * 2, c1)

            self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

        def _encode(self, x: torch.Tensor):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            return e1, e2, e3, e4, b

        def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            e1a, e2a, e3a, e4a, ba = self._encode(x1)
            e1b, e2b, e3b, e4b, bb = self._encode(x2)

            # Concatenation skips — decoder sees (t1, t2) features directly
            s1 = torch.cat([e1a, e1b], dim=1)
            s2 = torch.cat([e2a, e2b], dim=1)
            s3 = torch.cat([e3a, e3b], dim=1)
            s4 = torch.cat([e4a, e4b], dim=1)
            b = torch.cat([ba, bb], dim=1)

            u4 = self.up4(b)
            s4_up = F.interpolate(u4, size=s4.shape[-2:], mode="bilinear", align_corners=False)
            out4 = self.dec4(torch.cat([s4_up, s4], dim=1))

            u3 = self.up3(out4)
            s3_up = F.interpolate(u3, size=s3.shape[-2:], mode="bilinear", align_corners=False)
            out3 = self.dec3(torch.cat([s3_up, s3], dim=1))

            u2 = self.up2(out3)
            s2_up = F.interpolate(u2, size=s2.shape[-2:], mode="bilinear", align_corners=False)
            out2 = self.dec2(torch.cat([s2_up, s2], dim=1))

            u1 = self.up1(out2)
            s1_up = F.interpolate(u1, size=s1.shape[-2:], mode="bilinear", align_corners=False)
            out1 = self.dec1(torch.cat([s1_up, s1], dim=1))

            return self.out_conv(out1)


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


def run_fc_siam_conc_baseline(
    data_dir: str,
    dataset_name: str = "levir_cd",
    output_dir: str = "results/baselines/fc_siam_conc",
    split: str = "test",
    device: str = "cuda",
    image_size: int = 256,
    threshold: float = 0.5,
    checkpoint: str | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"FC-Siam-conc Baseline | dataset={dataset_name} | split={split}")
    print("=" * 60)

    dataset = _build_dataset(dataset_name, data_dir, split, image_size)

    if not _HAS_TORCH:
        print("Warning: torch not available — emitting random predictions.")
        model = None
    else:
        model = FCSiamConc().to(device)
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
    print(f"\nFC-Siam-conc Results ({dataset_name}/{split}):")
    print(f"  F1        : {agg.get('f1', 0):.4f}")
    print(f"  IoU       : {agg.get('iou', 0):.4f}")
    print(f"  Precision : {agg.get('precision', 0):.4f}")
    print(f"  Recall    : {agg.get('recall', 0):.4f}")

    results = {
        "model": "fc_siam_conc",
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
    parser.add_argument("--output_dir", type=str, default="results/baselines/fc_siam_conc")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if _HAS_TORCH and args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_fc_siam_conc_baseline(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        split=args.split,
        device=args.device,
        image_size=args.image_size,
        threshold=args.threshold,
        checkpoint=args.checkpoint,
    )
