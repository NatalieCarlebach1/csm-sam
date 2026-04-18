"""
DeepLabV3+ Baseline — atrous-conv slice-wise segmentation via segmentation_models_pytorch.

DeepLabV3+ with a ResNet-50 encoder applied slice-by-slice to mid-RT volumes.
Combines atrous spatial pyramid pooling with an encoder-decoder refinement stage.

Paper: Chen, Zhu, Papandreou, Schroff, Adam. "Encoder-Decoder with Atrous
Separable Convolution for Semantic Image Segmentation." ECCV 2018.

Install:
    pip install segmentation-models-pytorch

Uniqueness vs CSM-SAM:
    CSM-SAM uses a cross-session memory attention module that conditions mid-RT
    segmentation on features and masks from the pre-RT scan; this DeepLabV3+
    baseline has no temporal/cross-session mechanism and predicts each mid-RT
    slice independently. CSM-SAM's backbone is SAM2's frozen ViT-H pretrained on
    1B masks, whereas DeepLabV3+ uses a ~40M-parameter ResNet-50 backbone with
    ImageNet weights that must be fine-tuned on only ~150 patients.

Usage:
    python baselines/deeplabv3plus_baseline.py \
        --data_dir data/processed \
        --output_dir results/experiments/trained_baselines/deeplabv3plus \
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


def build_model(encoder_name: str = "resnet50", device: str = "cuda") -> torch.nn.Module:
    if not HAS_SMP:
        print("ERROR: segmentation_models_pytorch not installed.")
        print("Install with: pip install segmentation-models-pytorch")
        sys.exit(1)
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepLabV3+ baseline for HNTS-MRG 2024")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/experiments/trained_baselines/deeplabv3plus")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--encoder_name", type=str, default="resnet50")
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
        model_name="DeepLabV3Plus",
    )
