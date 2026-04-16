"""
Siamese U-Net Baseline — shared encoder for pre-RT and mid-RT, late feature fusion.

A smarter "use both scans" baseline than channel concat: a single smp U-Net
encoder is applied TWICE (weight-shared) to the pre-RT and mid-RT slices, the
two feature pyramids are concatenated at every stage, and a widened decoder
produces the mid-RT segmentation. Matches the classical Siamese
change-detection architecture (Daudt et al., 2018).

Uniqueness vs CSM-SAM:
    CSM-SAM propagates the PRE-RT MEMORY BANK via learned cross-session
    ATTENTION between SAM2 key/value tokens — a per-region correspondence
    mechanism that cannot be expressed by feature concatenation. CSM-SAM's
    change-head also supervises localization of tumor CHANGE via pre/mid
    MASK XOR, and CSM-SAM embeds weeks_elapsed to calibrate response
    magnitude; this siamese U-Net has neither.

Usage:
    python baselines/siamese_unet_baseline.py \
        --data_dir data/processed --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from csmsam.datasets import HNTSMRGDataset
from csmsam.utils.metrics import aggregate_metrics, evaluate_patient

try:
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
    from segmentation_models_pytorch.base import SegmentationHead
    HAS_SMP = True
except ImportError:
    HAS_SMP = False


class SiameseUNet(nn.Module):
    """Weight-shared encoder applied to pre and mid; fused pyramid --> decoder."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        num_classes: int = 2,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
    ):
        super().__init__()
        assert HAS_SMP, "segmentation_models_pytorch required for SiameseUNet"

        # One encoder, applied twice (weight-shared).
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=3, depth=5, weights=encoder_weights
        )

        # Decoder expects a pyramid whose stage widths match the encoder.
        # We concatenate pre+mid features, so decoder in_channels are doubled.
        enc_channels = list(self.encoder.out_channels)
        fused_channels = [c * 2 for c in enc_channels]
        # Keep the stem untouched (first element) because SMP's UnetDecoder
        # uses encoder.out_channels[0] as the "input_image" stub.
        fused_channels[0] = enc_channels[0]

        self.decoder = UnetDecoder(
            encoder_channels=fused_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, pre: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
        feats_pre = self.encoder(pre)
        feats_mid = self.encoder(mid)
        # Concatenate at every stage except the "input image" stub.
        fused = [feats_mid[0]] + [torch.cat([p, m], dim=1) for p, m in zip(feats_pre[1:], feats_mid[1:])]
        return self.segmentation_head(self.decoder(*fused))


class SiameseUNetBaseline:
    """Inference wrapper: slice-wise Siamese U-Net over paired (pre, mid) slices."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        num_classes: int = 2,
        checkpoint: str | None = None,
        device: str = "cuda",
    ):
        self.device = device
        self.num_classes = num_classes
        self.model: SiameseUNet | None = None

        if not HAS_SMP:
            print("Warning: segmentation_models_pytorch not installed.")
            print("Install with: pip install segmentation-models-pytorch")
            print("Falling back to random predictions.")
            return

        self.model = SiameseUNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            num_classes=num_classes,
        ).to(device)

        if checkpoint is not None and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location=device)
            state = state.get("model", state)
            self.model.load_state_dict(state, strict=False)
            print(f"Loaded checkpoint: {checkpoint}")
        else:
            print("No checkpoint provided; using ImageNet-initialized encoder only.")

        self.model.eval()

    @torch.no_grad()
    def predict_volume(
        self,
        pre_images: torch.Tensor,
        mid_images: torch.Tensor,
        threshold: float = 0.5,
    ) -> np.ndarray:
        N_mid, _, H, W = mid_images.shape
        if self.model is None:
            return (np.random.rand(N_mid, H, W) > 0.9).astype(np.float32)

        N_pre = pre_images.shape[0]
        pred_slices = []
        for i in range(N_mid):
            j = min(i, N_pre - 1)
            pre = pre_images[j:j + 1].to(self.device)
            mid = mid_images[i:i + 1].to(self.device)
            logits = self.model(pre, mid)
            if self.num_classes == 1:
                mask = (torch.sigmoid(logits) > threshold).squeeze().cpu().numpy()
            else:
                probs = F.softmax(logits, dim=1)
                mask = (probs[:, 1:].sum(dim=1) > threshold).squeeze().cpu().numpy()
            pred_slices.append(mask.astype(np.float32))

        return np.stack(pred_slices)


def run_siamese_unet_baseline(
    data_dir: str,
    output_dir: str,
    split: str = "test",
    checkpoint: str | None = None,
    device: str = "cuda",
    threshold: float = 0.5,
    image_size: int = 512,
    encoder_name: str = "resnet34",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Siamese U-Net Baseline (shared encoder, pre/mid feature fusion)")
    print(f"Encoder: {encoder_name} | Split: {split}")
    print("=" * 60)

    model = SiameseUNetBaseline(
        encoder_name=encoder_name,
        checkpoint=checkpoint,
        device=device,
    )

    dataset = HNTSMRGDataset(data_dir=data_dir, split=split, image_size=image_size)
    per_patient_metrics = []

    for idx in tqdm(range(len(dataset)), desc="Patients"):
        patient_data = dataset[idx]
        pid = patient_data["patient_id"]

        try:
            pre_images = patient_data["pre_images"]
            mid_images = patient_data["mid_images"]
            pred_binary = model.predict_volume(pre_images, mid_images, threshold)

            gt_gtvp = (patient_data["mid_masks_gtvp"] > 0.5).squeeze(1).numpy()
            gt_gtvn = (patient_data["mid_masks_gtvn"] > 0.5).squeeze(1).numpy()

            metrics = evaluate_patient(
                pred_masks=pred_binary,
                pred_gtvp=pred_binary,
                pred_gtvn=pred_binary,
                target_gtvp=gt_gtvp,
                target_gtvn=gt_gtvn,
            )
            metrics["patient_id"] = pid
            per_patient_metrics.append(metrics)
        except Exception as e:
            print(f"  Error on {pid}: {e}")

    agg = aggregate_metrics(per_patient_metrics)

    print(f"\nSiamese U-Net Baseline Results ({split}):")
    print(f"  aggDSC  : {agg.get('agg_dsc_mean', 0):.4f} ± {agg.get('agg_dsc_std', 0):.4f}")
    print(f"  GTVp DSC: {agg.get('dsc_gtvp_mean', 0):.4f}")
    print(f"  GTVn DSC: {agg.get('dsc_gtvn_mean', 0):.4f}")

    results = {
        "aggregate": agg,
        "per_patient": per_patient_metrics,
        "config": {
            "baseline": "siamese_unet",
            "encoder_name": encoder_name,
            "image_size": image_size,
            "threshold": threshold,
            "split": split,
        },
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {output_dir}/metrics.json")
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siamese U-Net baseline for HNTS-MRG 2024")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/baselines/siamese_unet")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--encoder_name", type=str, default="resnet34")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_siamese_unet_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        checkpoint=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        image_size=args.image_size,
        encoder_name=args.encoder_name,
    )
