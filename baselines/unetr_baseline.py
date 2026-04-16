"""
UNETR Baseline — 3D ViT-encoder / CNN-decoder segmentation via MONAI.

UNETR couples a plain Vision-Transformer encoder with a convolutional decoder
and skip connections, applied directly to 3D mid-RT volumes via sliding-window
inference.

Paper: Hatamizadeh et al. "UNETR: Transformers for 3D Medical Image
Segmentation." WACV 2022.

Install:
    pip install "monai[all]"

Uniqueness vs CSM-SAM:
    CSM-SAM models the mid-RT scan jointly with the pre-RT scan through a
    cross-session memory attention module, giving the decoder access to the
    earlier tumor state and its change map; UNETR is a single-volume segmenter
    with no pre-RT signal. CSM-SAM relies on SAM2's frozen ViT-H pretrained on
    1B masks, whereas UNETR trains a ~93M-parameter 3D ViT from scratch on only
    ~150 HNTS-MRG patients.

Usage:
    python baselines/unetr_baseline.py \
        --data_dir data/processed \
        --output_dir results/baselines/unetr \
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
    from monai.networks.nets import UNETR
    from monai.inferers import sliding_window_inference
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


class UNETRBaseline:
    """
    UNETR 3D baseline for mid-RT volume segmentation.

    Falls back to random predictions if MONAI is not installed.
    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 1,
        out_channels: int = 3,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
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

        self.model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            proj_type="conv",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(device)

        if checkpoint is not None and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location=device)
            state = state.get("model", state.get("state_dict", state))
            self.model.load_state_dict(state, strict=False)
            print(f"Loaded checkpoint: {checkpoint}")
        else:
            print("No checkpoint provided; using randomly initialized UNETR.")

        self.model.eval()

    @torch.no_grad()
    def predict_volume(
        self,
        mid_images: torch.Tensor,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Run UNETR on the full mid-RT volume with sliding-window inference.

        Args:
            mid_images : (N, 3, H, W) mid-RT slices; channel 0 is used as the
                         grayscale MRI input.

        Returns:
            pred_binary : (N, H, W) binary foreground prediction.
        """
        N, _, H, W = mid_images.shape

        if self.model is None:
            return (np.random.rand(N, H, W) > 0.9).astype(np.float32)

        vol = mid_images[:, 0].unsqueeze(0).unsqueeze(0).to(self.device)

        logits = sliding_window_inference(
            inputs=vol,
            roi_size=self.img_size,
            sw_batch_size=2,
            predictor=self.model,
            overlap=0.5,
            mode="gaussian",
        )

        if self.out_channels == 1:
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0)
            pred = (probs > threshold).cpu().numpy().astype(np.float32)
        else:
            probs = F.softmax(logits, dim=1).squeeze(0)
            pred = (probs[1:].sum(dim=0) > threshold).cpu().numpy().astype(np.float32)

        return pred


def run_unetr_baseline(
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
    print("UNETR Baseline (3D mid-RT only, no cross-session)")
    print(f"ROI: {roi_size}^3 | Split: {split}")
    print("=" * 60)

    model = UNETRBaseline(
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

    print(f"\nUNETR Baseline Results ({split}):")
    print(f"  aggDSC  : {agg.get('agg_dsc_mean', 0):.4f} ± {agg.get('agg_dsc_std', 0):.4f}")
    print(f"  GTVp DSC: {agg.get('dsc_gtvp_mean', 0):.4f}")
    print(f"  GTVn DSC: {agg.get('dsc_gtvn_mean', 0):.4f}")

    results = {"aggregate": agg, "per_patient": per_patient_metrics}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {output_dir}/metrics.json")
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNETR baseline for HNTS-MRG 2024")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/baselines/unetr")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--roi_size", type=int, default=96)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_unetr_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        checkpoint=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        image_size=args.image_size,
        roi_size=args.roi_size,
    )
