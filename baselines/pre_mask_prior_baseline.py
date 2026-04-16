"""
Pre-Mask-Prior Baseline — single-image U-Net with pre-RT MASK as an extra channel.

Instead of the pre-RT IMAGE, this baseline feeds the pre-RT MASK (the
GTVp+GTVn union, resampled onto the mid-RT grid by same-index copy) as an
additional input channel to a 2D U-Net operating on the mid-RT slice. The
model therefore sees:

    channel 0: mid-RT grayscale
    channel 1: pre-RT mask prior (0/1)

This is very close to the HNTS-MRG 2024 winner's approach (nnUNet with a
pre-RT mask channel) and is therefore the single strongest "longitudinal"
baseline we compare against.

Reference (winner):
    BAMF (Best et al.), HNTS-MRG 2024 challenge report.

Uniqueness vs CSM-SAM:
    CSM-SAM propagates the PRE-RT MEMORY BANK — per-slice key/value tokens
    from the frozen SAM2 encoder — not the pre-RT raw MASK. The mask-prior
    baseline collapses rich multi-scale feature information into a single
    binary channel, losing appearance, texture, and SAM2's 1B-mask prior.
    CSM-SAM's change head additionally supervises localization of tumor
    CHANGE via the pre/mid MASK XOR, whereas a pre-mask-channel U-Net has
    no explicit change signal. And CSM-SAM embeds weeks_elapsed to
    calibrate response magnitude, which this baseline ignores.

Usage:
    python baselines/pre_mask_prior_baseline.py \
        --data_dir data/processed \
        --output_dir results/baselines/pre_mask_prior \
        --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from csmsam.datasets import HNTSMRGDataset
from csmsam.utils.metrics import aggregate_metrics, evaluate_patient

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False


def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """(N, 3, H, W) SAM2 RGB --> (N, 1, H, W) grayscale (channel mean)."""
    return x.mean(dim=1, keepdim=True)


class PreMaskPriorBaseline:
    """
    2D U-Net on mid-RT with the pre-RT MASK as an extra input channel.

    At slice i we form input
        [mid_gray_i, pre_mask_min(i, N_pre-1)]     (1, 2, H, W)
    and decode a binary mid-RT segmentation.
    """

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
        self.model = None

        if not HAS_SMP:
            print("Warning: segmentation_models_pytorch not installed.")
            print("Install with: pip install segmentation-models-pytorch")
            print("Falling back to random predictions.")
            return

        # 2 input channels: [mid_gray, pre_mask]
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=2,
            classes=num_classes,
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
        mid_images: torch.Tensor,
        pre_masks: torch.Tensor,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Args:
            mid_images : (N_mid, 3, H, W)   — mid-RT SAM2 RGB slices
            pre_masks  : (N_pre, 1, H, W)   — pre-RT combined GTVp+GTVn mask
        Returns:
            (N_mid, H, W) foreground prediction
        """
        N_mid, _, H, W = mid_images.shape
        if self.model is None:
            return (np.random.rand(N_mid, H, W) > 0.9).astype(np.float32)

        N_pre = pre_masks.shape[0]
        mid_gray = _rgb_to_gray(mid_images)   # (N_mid, 1, H, W)

        pred_slices = []
        for i in range(N_mid):
            j = min(i, N_pre - 1)
            prior = (pre_masks[j:j + 1] > 0.5).float()  # (1, 1, H, W)
            x = torch.cat([mid_gray[i:i + 1], prior], dim=1).to(self.device)
            logits = self.model(x)
            if self.num_classes == 1:
                mask = (torch.sigmoid(logits) > threshold).squeeze().cpu().numpy()
            else:
                probs = F.softmax(logits, dim=1)
                mask = (probs[:, 1:].sum(dim=1) > threshold).squeeze().cpu().numpy()
            pred_slices.append(mask.astype(np.float32))

        return np.stack(pred_slices)


def run_pre_mask_prior_baseline(
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
    print("Pre-Mask-Prior Baseline (mid-RT image + pre-RT mask channel)")
    print(f"Encoder: {encoder_name} | Split: {split}")
    print("=" * 60)

    model = PreMaskPriorBaseline(
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
            mid_images = patient_data["mid_images"]
            pre_masks = patient_data["pre_masks"]  # (N_pre, 1, H, W) combined
            pred_binary = model.predict_volume(mid_images, pre_masks, threshold)

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

    print(f"\nPre-Mask-Prior Baseline Results ({split}):")
    print(f"  aggDSC  : {agg.get('agg_dsc_mean', 0):.4f} ± {agg.get('agg_dsc_std', 0):.4f}")
    print(f"  GTVp DSC: {agg.get('dsc_gtvp_mean', 0):.4f}")
    print(f"  GTVn DSC: {agg.get('dsc_gtvn_mean', 0):.4f}")

    results = {
        "aggregate": agg,
        "per_patient": per_patient_metrics,
        "config": {
            "baseline": "pre_mask_prior",
            "encoder_name": encoder_name,
            "in_channels": 2,
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
    parser = argparse.ArgumentParser(description="Pre-mask-prior 2D U-Net baseline.")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/baselines/pre_mask_prior")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--encoder_name", type=str, default="resnet34")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_pre_mask_prior_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        checkpoint=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        image_size=args.image_size,
        encoder_name=args.encoder_name,
    )
