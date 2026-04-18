"""
Concat-Channels Baseline — naive 2-channel (pre-RT, mid-RT) 2D U-Net.

The simplest way to "use" the pre-RT information: stack the pre-RT and mid-RT
slices together as a 2-channel grayscale input to a 2D encoder-decoder.

Backbone: segmentation_models_pytorch U-Net with an ImageNet-pretrained
encoder. The first conv is re-initialized for 2 input channels (pre, mid).

Uniqueness vs CSM-SAM:
    CSM-SAM propagates the PRE-RT MEMORY BANK (per-slice key/value tokens
    from the frozen SAM2 encoder) to the mid-RT stream via learned
    cross-session attention. This baseline merely stacks raw pre/mid pixels
    into the same 2D tensor.

Usage:
    python baselines/concat_channels_baseline.py \
        --data_dir data/processed \
        --output_dir results/experiments/trained_baselines/concat_channels \
        --epochs 50 --lr 1e-3
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

from baselines.baseline_trainer import train_and_evaluate


def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) SAM2-normalized RGB --> (B, 1, H, W) grayscale."""
    return x.mean(dim=1, keepdim=True)


def concat_adapter(pre_img, mid_img, patient_data, pre_idx):
    """Convert (B, 3, H, W) RGB pair to (B, 2, H, W) grayscale pair."""
    pre_gray = _rgb_to_gray(pre_img)
    mid_gray = _rgb_to_gray(mid_img)
    return torch.cat([pre_gray, mid_gray], dim=1)


def build_model(encoder_name: str = "resnet34", device: str = "cuda") -> nn.Module:
    if not HAS_SMP:
        print("ERROR: segmentation_models_pytorch not installed.")
        sys.exit(1)
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=2,
        classes=1,
    ).to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concat-Channels (pre, mid) 2D U-Net baseline.")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/experiments/trained_baselines/concat_channels")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--encoder_name", type=str, default="resnet34")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    model = build_model(encoder_name=args.encoder_name, device=args.device)

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
        model_name="ConcatChannels",
        uses_pre=True,
        input_adapter=concat_adapter,
    )
