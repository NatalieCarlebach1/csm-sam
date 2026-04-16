"""
BIT Baseline — Bitemporal Image Transformer for Change Detection.

Paper      : "Remote Sensing Image Change Detection with Transformers"
             H. Chen, Z. Qi, Z. Shi — IEEE TGRS 2021 (CVPR 2021 track).
Typical F1 : ~89 on LEVIR-CD (256x256, ResNet-18 backbone).
Install    : No external weights; paraphrased reimplementation. Canonical
             code at https://github.com/justchenhao/BIT_CD.

Architecture summary
--------------------
A ResNet-style CNN encodes both timepoints into dense feature maps. A
small semantic-tokenizer (spatial attention + weighted pooling) distills
each feature map into a short sequence of L tokens, then a Transformer
encoder jointly reasons over the two token sets. A Transformer decoder
broadcasts the refined tokens back onto each dense feature map, and the
pixel-wise distance between the two refined feature maps is fed through
a shallow prediction head to produce the change map.

Uniqueness vs CSM-SAM
---------------------
CSM-SAM inherits SAM2's 1B-mask-pretrained ViT-H encoder and trains only
~2M parameters of cross-session memory attention; BIT trains a small
ResNet-18 encoder plus transformer from scratch on ~1k image pairs.
Architecturally, BIT reasons over a compact L-token (≈4-token) bottleneck
that forces temporal interaction through a low-rank summary, whereas
CSM-SAM's cross-session memory operates over the full SAM2 token grid
with per-token query/key attention between pre- and mid-timepoint
tokens — richer spatial routing at the cost of more compute.

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

    class _BasicBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
            super().__init__()
            self.c1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.b1 = nn.BatchNorm2d(out_ch)
            self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
            self.b2 = nn.BatchNorm2d(out_ch)
            self.downsample = None
            if stride != 1 or in_ch != out_ch:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            identity = x if self.downsample is None else self.downsample(x)
            out = F.relu(self.b1(self.c1(x)), inplace=True)
            out = self.b2(self.c2(out))
            return F.relu(out + identity, inplace=True)


    class _MiniResNet(nn.Module):
        """Small ResNet-18-ish trunk that yields 1/4-resolution features."""

        def __init__(self, in_channels: int = 3, feat_dim: int = 64):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, feat_dim, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
            )
            self.layer1 = nn.Sequential(_BasicBlock(feat_dim, feat_dim),
                                        _BasicBlock(feat_dim, feat_dim))
            self.layer2 = nn.Sequential(_BasicBlock(feat_dim, feat_dim),
                                        _BasicBlock(feat_dim, feat_dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer2(self.layer1(self.stem(x)))


    class _TokenEncoder(nn.Module):
        """1x1 spatial attention head that extracts L semantic tokens."""

        def __init__(self, feat_dim: int, num_tokens: int = 4):
            super().__init__()
            self.num_tokens = num_tokens
            self.attn = nn.Conv2d(feat_dim, num_tokens, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape
            a = self.attn(x)  # (B, L, H, W)
            a = a.view(B, self.num_tokens, -1)
            a = F.softmax(a, dim=-1)
            xf = x.view(B, C, -1).transpose(1, 2)  # (B, HW, C)
            tokens = torch.bmm(a, xf)  # (B, L, C)
            return tokens


    class _Transformer(nn.Module):
        def __init__(self, dim: int, depth: int = 1, heads: int = 8, mlp_ratio: int = 2):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * mlp_ratio,
                batch_first=True, activation="gelu",
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=depth)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.enc(x)


    class _TransformerDecoder(nn.Module):
        """Broadcasts tokens back onto a feature map via cross-attention."""

        def __init__(self, dim: int, depth: int = 1, heads: int = 8, mlp_ratio: int = 2):
            super().__init__()
            layer = nn.TransformerDecoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * mlp_ratio,
                batch_first=True, activation="gelu",
            )
            self.dec = nn.TransformerDecoder(layer, num_layers=depth)

        def forward(self, feat: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
            B, C, H, W = feat.shape
            q = feat.view(B, C, -1).transpose(1, 2)  # (B, HW, C)
            out = self.dec(q, tokens)                # (B, HW, C)
            return out.transpose(1, 2).view(B, C, H, W)


    class BIT(nn.Module):
        """
        Bitemporal Image Transformer.

        Paraphrased reimplementation of Chen et al. 2021. Input: two RGB
        images (B, 3, H, W). Output: change logits (B, 1, H, W).
        """

        def __init__(self, in_channels: int = 3, feat_dim: int = 64,
                     num_tokens: int = 4, enc_depth: int = 1, dec_depth: int = 1):
            super().__init__()
            self.backbone = _MiniResNet(in_channels, feat_dim)
            self.tokenizer = _TokenEncoder(feat_dim, num_tokens)
            self.transformer = _Transformer(feat_dim, depth=enc_depth)
            self.trans_dec = _TransformerDecoder(feat_dim, depth=dec_depth)

            self.head = nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_dim, 1, 1),
            )

        def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            H, W = x1.shape[-2:]
            f1 = self.backbone(x1)
            f2 = self.backbone(x2)

            t1 = self.tokenizer(f1)
            t2 = self.tokenizer(f2)

            # Joint transformer reasoning over concatenated token sets
            joint = torch.cat([t1, t2], dim=1)
            joint = self.transformer(joint)
            L = t1.size(1)
            t1r, t2r = joint[:, :L], joint[:, L:]

            # Broadcast refined tokens back onto the dense features
            g1 = self.trans_dec(f1, t1r)
            g2 = self.trans_dec(f2, t2r)

            diff = torch.abs(g1 - g2)
            logits = self.head(diff)
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


def run_bit_baseline(
    data_dir: str,
    dataset_name: str = "levir_cd",
    output_dir: str = "results/baselines/bit",
    split: str = "test",
    device: str = "cuda",
    image_size: int = 256,
    threshold: float = 0.5,
    checkpoint: str | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"BIT Baseline | dataset={dataset_name} | split={split}")
    print("=" * 60)

    dataset = _build_dataset(dataset_name, data_dir, split, image_size)

    if not _HAS_TORCH:
        print("Warning: torch not available — emitting random predictions.")
        model = None
    else:
        model = BIT().to(device)
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
    print(f"\nBIT Results ({dataset_name}/{split}):")
    print(f"  F1        : {agg.get('f1', 0):.4f}")
    print(f"  IoU       : {agg.get('iou', 0):.4f}")
    print(f"  Precision : {agg.get('precision', 0):.4f}")
    print(f"  Recall    : {agg.get('recall', 0):.4f}")

    results = {
        "model": "bit",
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
    parser.add_argument("--output_dir", type=str, default="results/baselines/bit")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if _HAS_TORCH and args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_bit_baseline(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        split=args.split,
        device=args.device,
        image_size=args.image_size,
        threshold=args.threshold,
        checkpoint=args.checkpoint,
    )
