"""
MedNeXt Baseline — ConvNeXt-style 3D U-Net for medical segmentation.

MedNeXt is a scalable, ConvNeXt-inspired 3D U-Net designed specifically for
medical imaging; it is a strong convolutional counterpart to SwinUNETR/UNETR
and a standard modern baseline on many MICCAI benchmarks.

Since the official MedNeXt package provides only 3D models, this baseline
implements a lightweight 2D "MedNeXt-style" encoder-decoder with ConvNeXt
blocks, suitable for training via the shared baseline_trainer.

Paper: Roy et al. "MedNeXt: Transformer-driven Scaling of ConvNets for Medical
Image Segmentation." MICCAI 2023.

Uniqueness vs CSM-SAM:
    CSM-SAM couples the mid-RT encoder with a cross-session memory attention
    module that injects pre-RT features and mask priors into the decoder;
    MedNeXt has no cross-session or temporal pathway and predicts from the
    mid-RT volume alone.

Usage:
    python baselines/mednext_baseline.py \
        --data_dir data/processed \
        --output_dir results/experiments/trained_baselines/mednext \
        --epochs 50 --lr 1e-3
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.baseline_trainer import train_and_evaluate


class _ConvNeXtBlock2D(nn.Module):
    """A single ConvNeXt-style block: depthwise conv + LN + 1x1 expand + GELU + 1x1 project."""

    def __init__(self, dim: int, expand_ratio: int = 4):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)  # equivalent to LayerNorm over spatial
        self.pw1 = nn.Conv2d(dim, dim * expand_ratio, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * expand_ratio, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x + residual


class MedNeXt2D(nn.Module):
    """Lightweight 2D MedNeXt-style encoder-decoder for segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        n_blocks: tuple[int, ...] = (2, 2, 2, 2),
    ):
        super().__init__()
        channels = [base_channels * (2 ** i) for i in range(len(n_blocks))]

        # Encoder
        self.stem = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, (ch, nb) in enumerate(zip(channels, n_blocks)):
            self.encoders.append(nn.Sequential(*[_ConvNeXtBlock2D(ch) for _ in range(nb)]))
            if i < len(n_blocks) - 1:
                self.downsamples.append(nn.Conv2d(ch, channels[i + 1], 2, stride=2))

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(len(n_blocks) - 2, -1, -1):
            self.upsamples.append(nn.ConvTranspose2d(channels[i + 1], channels[i], 2, stride=2))
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(channels[i] * 2, channels[i], 1),
                    *[_ConvNeXtBlock2D(channels[i]) for _ in range(n_blocks[i])],
                )
            )

        self.head = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skip_connections = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < len(self.downsamples):
                skip_connections.append(x)
                x = self.downsamples[i](x)

        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skip_connections)):
            x = up(x)
            # Handle size mismatch from odd spatial dims
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))

        return self.head(x)


def build_model(device: str = "cuda") -> nn.Module:
    model = MedNeXt2D(in_channels=3, out_channels=1).to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedNeXt baseline for HNTS-MRG 2024")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/experiments/trained_baselines/mednext")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    model = build_model(device=args.device)

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
        model_name="MedNeXt2D",
    )
