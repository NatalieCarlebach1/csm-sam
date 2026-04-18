"""
SAM2 Point-Prompt Baseline — zero-shot per-slice, NO memory, NO finetuning.

Model       : SAM2 (Segment Anything 2), Ravi et al., Meta FAIR.
Paper       : "SAM 2: Segment Anything in Images and Videos" (2024).
Year        : 2024.
Backbone    : Hiera-Large image encoder; pretrained on SA-1B + SA-V.
Install     : git clone https://github.com/facebookresearch/sam2 && pip install -e sam2/
              # checkpoint: sam2.1_hiera_large.pt from the sam2 release

Uniqueness note vs CSM-SAM:
    CSM-SAM uses the same SAM2 image encoder but adds a CrossSessionMemoryAttention
    module that attends from mid-RT queries to a memory bank derived from the pre-RT
    visit, plus a change-map head. This baseline ablates both: each slice is decoded
    independently from a center point, with no within-session or cross-session
    memory — it isolates SAM2's raw zero-shot prior, which is blind to HNC tumor
    appearance and to the pre-RT lesion location.

Usage:
    python baselines/sam2_point_prompt_baseline.py \
        --data_dir data/processed \
        --sam2_checkpoint checkpoints/sam2/sam2.1_hiera_large.pt \
        --output_dir results/baselines/sam2_point_prompt
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


class SAM2PointPromptBaseline:
    """
    SAM2 zero-shot with an automatic center-point prompt per slice.

    Explicitly disables SAM2's video/memory propagation — each slice is decoded
    independently. This is the cleanest isolation of SAM2's frozen pretraining
    prior on 2D HNC MRI slices.

    Raises RuntimeError if sam2 is missing or checkpoint fails to load.
    """

    def __init__(self, sam2_checkpoint: str, model_cfg: str = "configs/sam2.1/sam2.1_hiera_l", device: str = "cuda"):
        self.device = device

        try:
            from sam2.build_sam import build_sam2
        except ImportError as e:
            raise RuntimeError(
                f"SAM2 failed to import: {e}\n"
                f"Install: git clone https://github.com/facebookresearch/sam2 && pip install -e sam2/"
            ) from e

        try:
            self.sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device)
            self.sam2.eval()
            print("SAM2 (point-prompt, no memory) loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"SAM2 checkpoint failed to load: {e}\n"
                f"Download: see https://github.com/facebookresearch/sam2#download-checkpoints"
            ) from e

    @torch.no_grad()
    def predict_volume(self, mid_images: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        N, _, H, W = mid_images.shape

        pred_slices = []
        for i in range(N):
            img = mid_images[i].unsqueeze(0).to(self.device)

            try:
                backbone_out = self.sam2.forward_image(img)
                features = backbone_out["vision_features"]

                pt = torch.tensor([[[W / 2, H / 2]]], dtype=torch.float, device=self.device)
                lbl = torch.ones(1, 1, dtype=torch.int, device=self.device)

                sparse, dense = self.sam2.sam_prompt_encoder(
                    points=(pt, lbl), boxes=None, masks=None,
                )
                low_res, _ = self.sam2.sam_mask_decoder(
                    image_embeddings=features,
                    image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                )
                mask = F.interpolate(low_res, size=(H, W), mode="bilinear", align_corners=False)
                pred_slices.append((torch.sigmoid(mask) > threshold).squeeze().cpu().numpy().astype(np.float32))
            except Exception:
                pred_slices.append(np.zeros((H, W), dtype=np.float32))

        return np.stack(pred_slices)


def run_sam2_point_prompt_baseline(
    data_dir: str,
    sam2_checkpoint: str,
    output_dir: str,
    split: str = "test",
    device: str = "cuda",
    threshold: float = 0.5,
    image_size: int = 1024,
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SAM2 Point-Prompt Baseline (no memory, no finetuning)")
    print(f"Split: {split}")
    print("=" * 60)

    model = SAM2PointPromptBaseline(sam2_checkpoint, model_cfg, device)

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

    print(f"\nSAM2 Point-Prompt Results ({split}):")
    print(f"  aggDSC  : {agg.get('agg_dsc_mean', 0):.4f} ± {agg.get('agg_dsc_std', 0):.4f}")
    print(f"  GTVp DSC: {agg.get('dsc_gtvp_mean', 0):.4f}")
    print(f"  GTVn DSC: {agg.get('dsc_gtvn_mean', 0):.4f}")

    results = {
        "baseline": "sam2_point_prompt",
        "model_cfg": model_cfg,
        "aggregate": agg,
        "per_patient": per_patient_metrics,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {output_dir}/metrics.json")
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--sam2_checkpoint", type=str, default="checkpoints/sam2/sam2.1_hiera_large.pt")
    parser.add_argument("--output_dir", type=str, default="results/baselines/sam2_point_prompt")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_sam2_point_prompt_baseline(
        data_dir=args.data_dir,
        sam2_checkpoint=args.sam2_checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        device=args.device,
        image_size=args.image_size,
        model_cfg=args.model_cfg,
    )
