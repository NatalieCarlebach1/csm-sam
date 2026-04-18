"""
SwinUNETR Baseline — 2D Swin-Transformer U-Net segmentation via MONAI.

A Swin-Transformer-based U-Net that processes mid-RT slices as 2D images
using MONAI's SwinUNETR with spatial_dims=2.

Paper: Hatamizadeh et al. "Swin UNETR: Swin Transformers for Semantic
Segmentation of Brain Tumors in MRI Images." BrainLes @ MICCAI 2021.

Install:
    pip install "monai[all]"

Uniqueness vs CSM-SAM:
    CSM-SAM introduces a cross-session memory attention module that carries
    pre-RT tumor embeddings forward to guide the mid-RT prediction; SwinUNETR
    here is trained only on mid-RT volumes and sees no pre-RT information.
    CSM-SAM also reuses SAM2's frozen ViT-H pretrained on 1B natural-image
    masks, whereas SwinUNETR trains its ~62M-parameter Swin encoder from
    scratch on only ~150 HNTS-MRG patients.

Usage:
    python baselines/swinunetr_baseline.py \
        --data_dir data/processed \
        --output_dir results/experiments/trained_baselines/swinunetr \
        --epochs 50 --lr 1e-3
"""

from __future__ import annotations

import argparse
import sys

import torch

try:
    from monai.networks.nets import SwinUNETR
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

from baselines.baseline_trainer import train_and_evaluate


def build_model(image_size: int = 256, device: str = "cuda") -> torch.nn.Module:
    if not HAS_MONAI:
        print("ERROR: MONAI not installed.")
        print('Install with: pip install "monai[all]"')
        sys.exit(1)
    model = SwinUNETR(
        img_size=(image_size, image_size),
        in_channels=3,
        out_channels=1,
        spatial_dims=2,
    ).to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwinUNETR baseline for HNTS-MRG 2024")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/experiments/trained_baselines/swinunetr")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    model = build_model(image_size=args.image_size, device=args.device)

    train_and_evaluate(
        model=model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        model_name="SwinUNETR",
    )
