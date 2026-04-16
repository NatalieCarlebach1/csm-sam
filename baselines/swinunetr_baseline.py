"""
SwinUNETR Baseline — 3D volumetric transformer segmentation via MONAI.

A Swin-Transformer-based 3D U-Net that processes the entire mid-RT volume at
once, operating directly on (D, H, W) patches via sliding-window inference.

Paper: Hatamizadeh et al. "Swin UNETR: Swin Transformers for Semantic
Segmentation of Brain Tumors in MRI Images." BrainLes @ MICCAI 2021.

Install:
    pip install "monai[all]"

Uniqueness vs CSM-SAM:
    CSM-SAM introduces a cross-session memory attention module that carries
    pre-RT tumor embeddings forward to guide the mid-RT prediction; SwinUNETR
    here is trained only on mid-RT volumes and sees no pre-RT information.
    CSM-SAM also reuses SAM2's frozen ViT-H pretrained on 1B natural-image
    masks, whereas SwinUNETR trains its ~62M-parameter 3D Swin encoder from
    scratch (or from a generic medical SSL initialization) on only ~150
    HNTS-MRG patients.

Usage:
    python baselines/swinunetr_baseline.py \
        --data_dir data/processed \
        --output_dir results/baselines/swinunetr \
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
from csmsam.utils.metrics import evaluate_patient, aggregate_metrics

try:
    from monai.networks.nets import SwinUNETR
    from monai.inferers import sliding_window_inference
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


class SwinUNETRBaseline:
    """
    SwinUNETR 3D baseline for mid-RT volume segmentation.

    Falls back to random predictions if MONAI is not available.
    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 1,
        out_channels: int = 3,
        feature_size: int = 48,
        checkpoint: str | None = None,
        device: str = "cuda",
    ):
        self.device = device
        self.img_size = img_size
        self.out_channels = out_channels
        self.model = None

        if not HAS_MONAI:
            print("Warning: MONAI not installed.")
            print('Install with: pip install "monai[all]"')
            print("Falling back to random predictions.")
            return

        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=True,
        ).to(device)

        if checkpoint is not None and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location=device)
            state = state.get("model", state.get("state_dict", state))
            self.model.load_state_dict(state, strict=False)
            print(f"Loaded checkpoint: {checkpoint}")
        else:
            print("No checkpoint provided; using randomly initialized SwinUNETR.")

        self.model.eval()

    @torch.no_grad()
    def predict_volume(
        self,
        mid_images: torch.Tensor,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Run SwinUNETR on the full mid-RT volume with sliding-window inference.

        Args:
            mid_images : (N, 3, H, W) mid-RT slices. Channel 0 is used as the
                         grayscale MRI (MONAI expects single-channel 3D input).

        Returns:
            pred_binary : (N, H, W) binary foreground prediction.
        """
        N, _, H, W = mid_images.shape

        if self.model is None:
            return (np.random.rand(N, H, W) > 0.9).astype(np.float32)

        # Reshape (N, 3, H, W) slices -> (1, 1, D, H, W) volume.
        vol = mid_images[:, 0].unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, N, H, W)

        logits = sliding_window_inference(
            inputs=vol,
            roi_size=self.img_size,
            sw_batch_size=2,
            predictor=self.model,
            overlap=0.5,
            mode="gaussian",
        )  # (1, C, N, H, W)

        if self.out_channels == 1:
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0)  # (N, H, W)
            pred = (probs > threshold).cpu().numpy().astype(np.float32)
        else:
            probs = F.softmax(logits, dim=1).squeeze(0)  # (C, N, H, W)
            pred = (probs[1:].sum(dim=0) > threshold).cpu().numpy().astype(np.float32)

        return pred


def run_swinunetr_baseline(
    data_dir: str,
    output_dir: str,
    split: str = "test",
    checkpoint: str | None = None,
    device: str = "cuda",
    threshold: float = 0.5,
    image_size: int = 256,
    roi_size: int = 96,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SwinUNETR Baseline (3D mid-RT only, no cross-session)")
    print(f"ROI: {roi_size}^3 | Split: {split}")
    print("=" * 60)

    model = SwinUNETRBaseline(
        img_size=(roi_size, roi_size, roi_size),
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
            pred_binary = model.predict_volume(mid_images, threshold)

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

    print(f"\nSwinUNETR Baseline Results ({split}):")
    print(f"  aggDSC  : {agg.get('agg_dsc_mean', 0):.4f} ± {agg.get('agg_dsc_std', 0):.4f}")
    print(f"  GTVp DSC: {agg.get('dsc_gtvp_mean', 0):.4f}")
    print(f"  GTVn DSC: {agg.get('dsc_gtvn_mean', 0):.4f}")

    results = {"aggregate": agg, "per_patient": per_patient_metrics}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {output_dir}/metrics.json")
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwinUNETR baseline for HNTS-MRG 2024")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/baselines/swinunetr")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--roi_size", type=int, default=96)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_swinunetr_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        checkpoint=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        image_size=args.image_size,
        roi_size=args.roi_size,
    )
