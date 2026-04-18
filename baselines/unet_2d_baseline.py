"""
2D U-Net Baseline — slice-wise mid-RT segmentation via segmentation_models_pytorch.

A classical 2D encoder-decoder U-Net (Ronneberger et al., MICCAI 2015) applied
slice-by-slice to mid-RT volumes. Uses an ImageNet-pretrained ResNet-34 encoder.

Paper: Ronneberger, Fischer, Brox. "U-Net: Convolutional Networks for Biomedical
Image Segmentation." MICCAI 2015.

Install:
    pip install segmentation-models-pytorch

Uniqueness vs CSM-SAM:
    CSM-SAM uses cross-session memory from the pre-RT scan to propagate tumor
    context into the mid-RT segmentation; this 2D U-Net baseline processes each
    mid-RT slice in isolation with no pre-RT information and no memory. CSM-SAM
    also leverages SAM2's frozen ViT-H pretrained on 1B masks for strong priors,
    whereas this baseline is a ~24M-parameter U-Net trained from an ImageNet
    initialization on only ~150 HNTS-MRG patients.

Usage:
    python baselines/unet_2d_baseline.py \
        --data_dir data/processed \
        --output_dir results/experiments/trained_baselines/unet2d \
        --epochs 50 --lr 1e-3
"""

from __future__ import annotations

import argparse
import sys

import torch

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

from baselines.baseline_trainer import train_and_evaluate


def build_model(encoder_name: str = "resnet34", device: str = "cuda") -> torch.nn.Module:
    if not HAS_SMP:
        print("ERROR: segmentation_models_pytorch not installed.")
        print("Install with: pip install segmentation-models-pytorch")
        sys.exit(1)
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D U-Net baseline for HNTS-MRG 2024")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/experiments/trained_baselines/unet2d")
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
        model_name="UNet2D",
    )
