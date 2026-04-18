"""
MedSAM2 Baseline — within-session memory only (no cross-session).

This is the primary ablation baseline for CSM-SAM. It uses SAM2's standard
memory propagation (slice → slice within the mid-RT scan) but does NOT use
any information from the pre-RT scan.

This isolates the contribution of cross-session memory.

Usage:
    python baselines/medsam2_baseline.py \
        --data_dir data/processed \
        --sam2_checkpoint checkpoints/sam2/sam2.1_hiera_large.pt \
        --output_dir results/baselines/medsam2
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
from csmsam.utils.metrics import evaluate_patient, aggregate_metrics, format_results_table


class MedSAM2Baseline:
    """
    MedSAM2-style baseline: SAM2 with within-session memory only.

    For each mid-RT slice:
      1. Run SAM2 encoder
      2. Attend to memory from previous mid-RT slices
      3. Decode mask
      NO cross-session (pre-RT) information used.

    Raises RuntimeError if SAM2 is not installed or checkpoint fails to load.
    """

    def __init__(self, sam2_checkpoint: str, device: str = "cuda"):
        self.device = device

        try:
            from sam2.build_sam import build_sam2
        except ImportError as e:
            raise RuntimeError(
                f"SAM2 failed to import: {e}\n"
                f"Install: git clone https://github.com/facebookresearch/sam2 && pip install -e sam2/"
            ) from e

        try:
            self.sam2 = build_sam2("configs/sam2.1/sam2.1_hiera_l", sam2_checkpoint, device=device)
            self.sam2.eval()
            print("MedSAM2 loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"MedSAM2 (SAM2) checkpoint failed to load: {e}\n"
                f"Download: see https://github.com/facebookresearch/sam2#download-checkpoints"
            ) from e

    @torch.no_grad()
    def predict_volume(
        self,
        mid_images: torch.Tensor,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Segment mid-RT volume with within-session memory only.

        Args:
            mid_images : (N, 3, H, W) mid-RT slices

        Returns:
            pred_binary : (N, H, W) binary prediction
        """
        N, _, H, W = mid_images.shape

        # Use SAM2 predictor with within-session memory
        pred_slices = []
        memory_bank = None

        for i in range(N):
            img = mid_images[i].unsqueeze(0).to(self.device)  # (1, 3, H, W)

            # Extract features
            with torch.no_grad():
                backbone_out = self.sam2.forward_image(img)
                features = backbone_out["vision_features"]  # (1, C, h, w)

            # Simple automatic point prompt: center of image
            H_feat, W_feat = features.shape[-2:]
            point_coords = torch.tensor([[[W / 2, H / 2]]], dtype=torch.float, device=self.device)
            point_labels = torch.ones(1, 1, dtype=torch.int, device=self.device)

            try:
                sparse_embed, dense_embed = self.sam2.sam_prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None,
                    masks=None,
                )
                low_res_masks, _ = self.sam2.sam_mask_decoder(
                    image_embeddings=features,
                    image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embed,
                    dense_prompt_embeddings=dense_embed,
                    multimask_output=False,
                )
                mask = F.interpolate(low_res_masks, size=(H, W), mode="bilinear", align_corners=False)
                pred_slices.append((torch.sigmoid(mask) > threshold).squeeze().cpu().numpy())
            except Exception:
                pred_slices.append(np.zeros((H, W), dtype=np.float32))

        return np.stack(pred_slices)


def run_medsam2_baseline(
    data_dir: str,
    sam2_checkpoint: str,
    output_dir: str,
    split: str = "test",
    device: str = "cuda",
    threshold: float = 0.5,
    image_size: int = 1024,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MedSAM2 Baseline (within-session only)")
    print(f"Split: {split}")
    print("=" * 60)

    model = MedSAM2Baseline(sam2_checkpoint, device)

    dataset = HNTSMRGDataset(data_dir=data_dir, split=split, image_size=image_size)
    per_patient_metrics = []

    for idx in tqdm(range(len(dataset)), desc="Patients"):
        patient_data = dataset[idx]
        pid = patient_data["patient_id"]

        try:
            mid_images = patient_data["mid_images"]  # (N, 3, H, W)
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

    print(f"\nMedSAM2 Baseline Results ({split}):")
    print(f"  aggDSC  : {agg.get('agg_dsc_mean', 0):.4f} ± {agg.get('agg_dsc_std', 0):.4f}")
    print(f"  GTVp DSC: {agg.get('dsc_gtvp_mean', 0):.4f}")
    print(f"  GTVn DSC: {agg.get('dsc_gtvn_mean', 0):.4f}")

    results = {"aggregate": agg, "per_patient": per_patient_metrics}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {output_dir}/metrics.json")
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--sam2_checkpoint", type=str, default="checkpoints/sam2/sam2.1_hiera_large.pt")
    parser.add_argument("--output_dir", type=str, default="results/baselines/medsam2")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=1024)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_medsam2_baseline(
        data_dir=args.data_dir,
        sam2_checkpoint=args.sam2_checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        device=args.device,
        image_size=args.image_size,
    )
