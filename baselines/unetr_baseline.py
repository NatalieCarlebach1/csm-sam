"""
UNETR Baseline — 2D ViT-encoder / CNN-decoder segmentation via MONAI.

UNETR couples a plain Vision-Transformer encoder with a convolutional decoder
and skip connections, applied to 2D mid-RT slices via MONAI's UNETR with
spatial_dims=2.

Paper: Hatamizadeh et al. "UNETR: Transformers for 3D Medical Image
Segmentation." WACV 2022.

Install:
    pip install "monai[all]"

Uniqueness vs CSM-SAM:
    CSM-SAM models the mid-RT scan jointly with the pre-RT scan through a
    cross-session memory attention module, giving the decoder access to the
    earlier tumor state and its change map; UNETR is a single-volume segmenter
    with no pre-RT signal. CSM-SAM relies on SAM2's frozen ViT-H pretrained on
    1B masks, whereas UNETR trains a ~93M-parameter ViT from scratch on only
    ~150 HNTS-MRG patients.

Usage:
    python baselines/unetr_baseline.py \
        --data_dir data/processed \
        --output_dir results/experiments/trained_baselines/unetr \
        --epochs 50 --lr 1e-3
"""

from __future__ import annotations

import argparse
import sys

import torch

try:
    from monai.networks.nets import UNETR
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

from baselines.baseline_trainer import train_and_evaluate


def build_model(image_size: int = 256, device: str = "cuda") -> torch.nn.Module:
    if not HAS_MONAI:
        print("ERROR: MONAI not installed.")
        print('Install with: pip install "monai[all]"')
        sys.exit(1)
    model = UNETR(
        in_channels=3,
        out_channels=1,
        img_size=(image_size, image_size),
        spatial_dims=2,
    ).to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNETR baseline for HNTS-MRG 2024")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/experiments/trained_baselines/unetr")
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
        model_name="UNETR",
    )
