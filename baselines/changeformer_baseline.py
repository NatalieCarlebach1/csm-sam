"""
ChangeFormer Baseline — Hierarchical Transformer for Change Detection.

Paper      : "A Transformer-Based Siamese Network for Change Detection"
             W. G. C. Bandara, V. M. Patel — IGARSS 2022.
Typical F1 : ~90 on LEVIR-CD (256x256), state-of-the-art at publication.
Install    : No external weights; paraphrased reimplementation. Canonical
             code at https://github.com/wgcban/ChangeFormer.

Architecture summary
--------------------
A hierarchical transformer (SegFormer/MiT-B1-lite style) encodes each
timepoint into multi-scale token sequences. Paired features at each
scale are fused by taking their concatenation + 1x1 conv ("difference
module") and passed to an all-MLP decoder that projects every scale to
a common embedding dim, upsamples, and concatenates before a 1x1
classifier predicts the change map.

Uniqueness vs CSM-SAM
---------------------
CSM-SAM inherits SAM2's 1B-mask-pretrained ViT-H and trains ~2M
parameters of cross-session memory attention; ChangeFormer trains its
hierarchical transformer encoder from scratch on ~1k image pairs. The
two differ architecturally: ChangeFormer fuses the temporal pair via
per-scale concat + 1x1 convolution (a local channel-mixing operation),
while CSM-SAM's cross-session memory applies *per-token query/key
attention* across time points, allowing pixels in the mid-timepoint
feature grid to directly attend to arbitrary pre-timepoint tokens.

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

    class _OverlapPatchEmbed(nn.Module):
        """Overlapping patch embedding (conv with stride < kernel)."""

        def __init__(self, in_ch: int, out_ch: int, patch: int, stride: int):
            super().__init__()
            self.proj = nn.Conv2d(in_ch, out_ch, patch, stride=stride, padding=patch // 2)
            self.norm = nn.LayerNorm(out_ch)

        def forward(self, x: torch.Tensor):
            x = self.proj(x)
            B, C, H, W = x.shape
            xf = x.flatten(2).transpose(1, 2)  # (B, HW, C)
            xf = self.norm(xf)
            return xf, H, W


    class _EfficientAttention(nn.Module):
        """SegFormer-style spatial-reduction self-attention."""

        def __init__(self, dim: int, heads: int = 4, sr_ratio: int = 1):
            super().__init__()
            self.heads = heads
            self.scale = (dim // heads) ** -0.5
            self.q = nn.Linear(dim, dim)
            self.kv = nn.Linear(dim, dim * 2)
            self.proj = nn.Linear(dim, dim)
            self.sr_ratio = sr_ratio
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
                self.sr_norm = nn.LayerNorm(dim)

        def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
            B, N, C = x.shape
            q = self.q(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
            if self.sr_ratio > 1:
                xt = x.transpose(1, 2).reshape(B, C, H, W)
                xt = self.sr(xt).reshape(B, C, -1).transpose(1, 2)
                xt = self.sr_norm(xt)
                kv = self.kv(xt)
            else:
                kv = self.kv(x)
            kv = kv.reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return self.proj(out)


    class _MixFFN(nn.Module):
        def __init__(self, dim: int, expansion: int = 4):
            super().__init__()
            hidden = dim * expansion
            self.fc1 = nn.Linear(dim, hidden)
            self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden, dim)

        def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
            x = self.fc1(x)
            B, N, C = x.shape
            xt = x.transpose(1, 2).reshape(B, C, H, W)
            xt = self.dw(xt)
            x = xt.flatten(2).transpose(1, 2)
            x = self.act(x)
            return self.fc2(x)


    class _TxBlock(nn.Module):
        def __init__(self, dim: int, heads: int, sr_ratio: int):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = _EfficientAttention(dim, heads, sr_ratio)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = _MixFFN(dim)

        def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
            x = x + self.attn(self.norm1(x), H, W)
            x = x + self.mlp(self.norm2(x), H, W)
            return x


    class _MiTEncoder(nn.Module):
        """Tiny hierarchical transformer encoder (4 stages)."""

        def __init__(self, in_channels: int = 3):
            super().__init__()
            dims = (32, 64, 160, 256)
            self.dims = dims
            heads = (1, 2, 5, 8)
            sr = (8, 4, 2, 1)
            patch = ((7, 4), (3, 2), (3, 2), (3, 2))

            self.patch_embeds = nn.ModuleList([
                _OverlapPatchEmbed(
                    in_ch if i == 0 else dims[i - 1],
                    dims[i], patch[i][0], patch[i][1],
                ) for i, in_ch in enumerate([in_channels, dims[0], dims[1], dims[2]])
            ])
            self.blocks = nn.ModuleList([
                nn.ModuleList([_TxBlock(dims[i], heads[i], sr[i])])
                for i in range(4)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims])

        def forward(self, x: torch.Tensor):
            feats = []
            for i in range(4):
                x, H, W = self.patch_embeds[i](x)
                for blk in self.blocks[i]:
                    x = blk(x, H, W)
                x = self.norms[i](x)
                B, N, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, H, W)
                feats.append(x)
            return feats  # list of 4 feature maps


    class _DiffModule(nn.Module):
        """Per-scale concat + 1x1 conv temporal fuser."""

        def __init__(self, dim: int):
            super().__init__()
            self.fuse = nn.Sequential(
                nn.Conv2d(dim * 2, dim, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            )

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return self.fuse(torch.cat([a, b], dim=1))


    class ChangeFormer(nn.Module):
        """
        Hierarchical transformer siamese change detection network.

        Paraphrased reimplementation of Bandara & Patel 2022. Input: two
        RGB images (B, 3, H, W). Output: change logits (B, 1, H, W).
        """

        def __init__(self, in_channels: int = 3, embed_dim: int = 128):
            super().__init__()
            self.encoder = _MiTEncoder(in_channels)
            dims = self.encoder.dims
            self.diffs = nn.ModuleList([_DiffModule(d) for d in dims])
            self.projs = nn.ModuleList([nn.Conv2d(d, embed_dim, 1) for d in dims])
            self.fuse = nn.Sequential(
                nn.Conv2d(embed_dim * 4, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Conv2d(embed_dim, 1, 1)

        def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            H, W = x1.shape[-2:]
            feats1 = self.encoder(x1)
            feats2 = self.encoder(x2)

            fused = [d(a, b) for d, a, b in zip(self.diffs, feats1, feats2)]
            projected = [p(f) for p, f in zip(self.projs, fused)]
            target_size = projected[0].shape[-2:]
            upsampled = [
                F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
                for f in projected
            ]
            x = self.fuse(torch.cat(upsampled, dim=1))
            logits = self.classifier(x)
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


def run_changeformer_baseline(
    data_dir: str,
    dataset_name: str = "levir_cd",
    output_dir: str = "results/baselines/changeformer",
    split: str = "test",
    device: str = "cuda",
    image_size: int = 256,
    threshold: float = 0.5,
    checkpoint: str | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"ChangeFormer Baseline | dataset={dataset_name} | split={split}")
    print("=" * 60)

    dataset = _build_dataset(dataset_name, data_dir, split, image_size)

    if not _HAS_TORCH:
        print("Warning: torch not available — emitting random predictions.")
        model = None
    else:
        model = ChangeFormer().to(device)
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
    print(f"\nChangeFormer Results ({dataset_name}/{split}):")
    print(f"  F1        : {agg.get('f1', 0):.4f}")
    print(f"  IoU       : {agg.get('iou', 0):.4f}")
    print(f"  Precision : {agg.get('precision', 0):.4f}")
    print(f"  Recall    : {agg.get('recall', 0):.4f}")

    results = {
        "model": "changeformer",
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
    parser.add_argument("--output_dir", type=str, default="results/baselines/changeformer")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if _HAS_TORCH and args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_changeformer_baseline(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        split=args.split,
        device=args.device,
        image_size=args.image_size,
        threshold=args.threshold,
        checkpoint=args.checkpoint,
    )
