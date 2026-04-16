"""
SNUNet-CD Baseline — Siamese Nested-UNet for Change Detection.

Paper      : "SNUNet-CD: A Densely Connected Siamese Network for Change
              Detection of VHR Images"
             S. Fang, K. Li, J. Shao, Z. Li — IEEE GRSL 2022.
Typical F1 : ~88 on LEVIR-CD (256x256).
Install    : No external weights; paraphrased reimplementation. Canonical
             code at https://github.com/likyoo/Siam-NestedUNet.

Architecture summary
--------------------
A compact siamese encoder with a NestedUNet-style (UNet++) decoder: every
decoder node takes as input all finer-scale decoder nodes along the same
level plus the upsampled coarser node. A final channel-attention (ECAM)
module reweighs the multi-scale decoder outputs before a 1x1 conv
produces the change map. The dense nested skips are the main driver of
the strong small-object performance reported in the paper.

Uniqueness vs CSM-SAM
---------------------
CSM-SAM inherits SAM2's 1B-mask-pretrained ViT-H encoder and trains only
~2M parameters of cross-session memory attention; SNUNet-CD trains an
entire nested siamese UNet from scratch on ~1k image pairs. CSM-SAM's
cross-session memory is architecturally richer than NestedUNet skips —
it supports per-token query/key attention routing between pre- and
mid-timepoint tokens, whereas SNUNet-CD fuses temporal information by
concatenating paired encoder outputs at each nested node and relies on
3x3 convolutions to mix them.

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

    class _ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)


    class _ECAM(nn.Module):
        """Ensemble Channel Attention Module for multi-output fusion."""

        def __init__(self, in_ch: int, ratio: int = 4):
            super().__init__()
            hidden = max(1, in_ch // ratio)
            self.avg = nn.AdaptiveAvgPool2d(1)
            self.mlp = nn.Sequential(
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, in_ch, 1, bias=False),
            )
            self.act = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            w = self.act(self.mlp(self.avg(x)))
            return x * w


    class SNUNetCD(nn.Module):
        """
        Siamese NestedUNet for change detection.

        Paraphrased reimplementation of Fang et al. 2022. Input: two RGB
        images (B, 3, H, W). Output: change logits (B, 1, H, W).
        """

        def __init__(self, in_channels: int = 3, base: int = 32):
            super().__init__()
            c = [base, base * 2, base * 4, base * 8, base * 16]

            # Shared siamese encoder (5 levels)
            self.conv00 = _ConvBlock(in_channels, c[0])
            self.conv10 = _ConvBlock(c[0], c[1])
            self.conv20 = _ConvBlock(c[1], c[2])
            self.conv30 = _ConvBlock(c[2], c[3])
            self.conv40 = _ConvBlock(c[3], c[4])
            self.pool = nn.MaxPool2d(2, 2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

            # Nested decoder. Paired (t1, t2) features are concatenated at
            # the same level; deeper decoder nodes also consume already
            # computed finer decoder nodes along their row (NestedUNet).
            self.conv01 = _ConvBlock(c[0] * 2 + c[1] * 2, c[0])
            self.conv11 = _ConvBlock(c[1] * 2 + c[2] * 2, c[1])
            self.conv21 = _ConvBlock(c[2] * 2 + c[3] * 2, c[2])
            self.conv31 = _ConvBlock(c[3] * 2 + c[4] * 2, c[3])

            self.conv02 = _ConvBlock(c[0] * 2 + c[0] + c[1], c[0])
            self.conv12 = _ConvBlock(c[1] * 2 + c[1] + c[2], c[1])
            self.conv22 = _ConvBlock(c[2] * 2 + c[2] + c[3], c[2])

            self.conv03 = _ConvBlock(c[0] * 2 + c[0] * 2 + c[1], c[0])
            self.conv13 = _ConvBlock(c[1] * 2 + c[1] * 2 + c[2], c[1])

            self.conv04 = _ConvBlock(c[0] * 2 + c[0] * 3 + c[1], c[0])

            # ECAM over the 4 nested outputs (L=1..4), followed by 1x1 conv
            self.ecam = _ECAM(c[0] * 4)
            self.out_conv = nn.Conv2d(c[0] * 4, 1, kernel_size=1)

        def _encode(self, x: torch.Tensor):
            x0 = self.conv00(x)
            x1 = self.conv10(self.pool(x0))
            x2 = self.conv20(self.pool(x1))
            x3 = self.conv30(self.pool(x2))
            x4 = self.conv40(self.pool(x3))
            return x0, x1, x2, x3, x4

        @staticmethod
        def _match_cat(*tensors: torch.Tensor) -> torch.Tensor:
            """Resize everything to the first tensor's HxW and concat."""
            ref = tensors[0]
            aligned = [ref] + [
                F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
                for t in tensors[1:]
            ]
            return torch.cat(aligned, dim=1)

        def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            a0, a1, a2, a3, a4 = self._encode(x1)
            b0, b1, b2, b3, b4 = self._encode(x2)

            # Row 1
            x01 = self.conv01(self._match_cat(torch.cat([a0, b0], 1), self.up(torch.cat([a1, b1], 1))))
            x11 = self.conv11(self._match_cat(torch.cat([a1, b1], 1), self.up(torch.cat([a2, b2], 1))))
            x21 = self.conv21(self._match_cat(torch.cat([a2, b2], 1), self.up(torch.cat([a3, b3], 1))))
            x31 = self.conv31(self._match_cat(torch.cat([a3, b3], 1), self.up(torch.cat([a4, b4], 1))))

            # Row 2
            x02 = self.conv02(self._match_cat(torch.cat([a0, b0], 1), x01, self.up(x11)))
            x12 = self.conv12(self._match_cat(torch.cat([a1, b1], 1), x11, self.up(x21)))
            x22 = self.conv22(self._match_cat(torch.cat([a2, b2], 1), x21, self.up(x31)))

            # Row 3
            x03 = self.conv03(self._match_cat(torch.cat([a0, b0], 1), x01, x02, self.up(x12)))
            x13 = self.conv13(self._match_cat(torch.cat([a1, b1], 1), x11, x12, self.up(x22)))

            # Row 4
            x04 = self.conv04(self._match_cat(torch.cat([a0, b0], 1), x01, x02, x03, self.up(x13)))

            # Ensemble-attention over the 4 nested outputs
            fused = torch.cat([x01, x02, x03, x04], dim=1)
            fused = self.ecam(fused)
            return self.out_conv(fused)


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


def run_snunet_baseline(
    data_dir: str,
    dataset_name: str = "levir_cd",
    output_dir: str = "results/baselines/snunet",
    split: str = "test",
    device: str = "cuda",
    image_size: int = 256,
    threshold: float = 0.5,
    checkpoint: str | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SNUNet-CD Baseline | dataset={dataset_name} | split={split}")
    print("=" * 60)

    dataset = _build_dataset(dataset_name, data_dir, split, image_size)

    if not _HAS_TORCH:
        print("Warning: torch not available — emitting random predictions.")
        model = None
    else:
        model = SNUNetCD().to(device)
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
    print(f"\nSNUNet-CD Results ({dataset_name}/{split}):")
    print(f"  F1        : {agg.get('f1', 0):.4f}")
    print(f"  IoU       : {agg.get('iou', 0):.4f}")
    print(f"  Precision : {agg.get('precision', 0):.4f}")
    print(f"  Recall    : {agg.get('recall', 0):.4f}")

    results = {
        "model": "snunet_cd",
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
    parser.add_argument("--output_dir", type=str, default="results/baselines/snunet")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if _HAS_TORCH and args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_snunet_baseline(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        split=args.split,
        device=args.device,
        image_size=args.image_size,
        threshold=args.threshold,
        checkpoint=args.checkpoint,
    )
