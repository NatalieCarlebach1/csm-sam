"""
Siamese U-Net Baseline — shared encoder for pre-RT and mid-RT, late feature fusion.

A single smp U-Net encoder is applied TWICE (weight-shared) to the pre-RT and
mid-RT slices, the two feature pyramids are concatenated at every stage, and a
widened decoder produces the mid-RT segmentation.

Uniqueness vs CSM-SAM:
    CSM-SAM propagates the PRE-RT MEMORY BANK via learned cross-session
    ATTENTION between SAM2 key/value tokens -- a per-region correspondence
    mechanism that cannot be expressed by feature concatenation.

Usage:
    python baselines/siamese_unet_baseline.py \
        --data_dir data/processed \
        --output_dir results/experiments/trained_baselines/siamese_unet \
        --epochs 50 --lr 1e-3
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
    from segmentation_models_pytorch.base import SegmentationHead
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

from baselines.baseline_trainer import train_and_evaluate


class SiameseUNet(nn.Module):
    """Weight-shared encoder applied to pre and mid; fused pyramid --> decoder."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
    ):
        super().__init__()
        assert HAS_SMP, "segmentation_models_pytorch required for SiameseUNet"

        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=3, depth=5, weights=encoder_weights
        )

        enc_channels = list(self.encoder.out_channels)
        fused_channels = [c * 2 for c in enc_channels]
        fused_channels[0] = enc_channels[0]

        # smp decoder API differs across versions. Build with the minimal
        # kwargs that work everywhere; add optional kwargs only if accepted.
        import inspect
        decoder_kwargs = dict(
            encoder_channels=fused_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
            attention_type=None,
        )
        sig = inspect.signature(UnetDecoder.__init__)
        if "use_norm" in sig.parameters:
            decoder_kwargs["use_norm"] = "batchnorm"
        elif "use_batchnorm" in sig.parameters:
            decoder_kwargs["use_batchnorm"] = True
        if "center" in sig.parameters:
            decoder_kwargs["center"] = False
        self.decoder = UnetDecoder(**decoder_kwargs)

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accept (B, 6, H, W) = [pre_rgb, mid_rgb] or two separate tensors."""
        if x.shape[1] == 6:
            pre = x[:, :3]
            mid = x[:, 3:]
        else:
            raise ValueError(f"Expected 6-channel input, got {x.shape[1]}")
        feats_pre = self.encoder(pre)
        feats_mid = self.encoder(mid)
        fused = [feats_mid[0]] + [
            torch.cat([p, m], dim=1) for p, m in zip(feats_pre[1:], feats_mid[1:])
        ]
        # smp 0.5 changed forward signature: new takes list, old takes *args
        import inspect
        sig = inspect.signature(self.decoder.forward)
        params = list(sig.parameters)
        if len(params) == 1 and params[0] not in ("args", "kwargs"):
            dec_out = self.decoder(fused)  # new API: single list arg
        else:
            dec_out = self.decoder(*fused)  # old API: positional
        return self.segmentation_head(dec_out)


def build_model(encoder_name: str = "resnet34", device: str = "cuda") -> nn.Module:
    if not HAS_SMP:
        print("ERROR: segmentation_models_pytorch not installed.")
        sys.exit(1)
    model = SiameseUNet(encoder_name=encoder_name).to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siamese U-Net baseline for HNTS-MRG 2024")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/experiments/trained_baselines/siamese_unet")
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
        model_name="SiameseUNet",
        uses_pre=True,
    )
