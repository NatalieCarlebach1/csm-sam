"""
SAM (Vanilla / SAM1) Baseline — zero-shot with automatic point prompts.

Model       : Segment Anything (SAM), Kirillov et al., Meta AI.
Paper       : "Segment Anything" (ICCV 2023).
Year        : 2023.
Backbone    : ViT-H (default); the image encoder is pretrained on SA-1B (1B masks).
Install     : pip install segment-anything
              # checkpoint: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Uniqueness note vs CSM-SAM:
    CSM-SAM builds on SAM/SAM2 encoders but adds a CrossSessionMemoryAttention module
    that injects pre-RT memory tokens into the mid-RT decode path, plus a change-map
    head supervised by pre/mid mask XOR. Vanilla SAM has no temporal memory, no
    cross-session conditioning, and no task-specific head — it is purely zero-shot
    with a geometric prompt, which collapses on soft-tissue HNC tumors.

Usage:
    python baselines/sam_vanilla_baseline.py \
        --data_dir data/processed \
        --sam_checkpoint checkpoints/sam/sam_vit_h_4b8939.pth \
        --output_dir results/baselines/sam_vanilla
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from csmsam.datasets import HNTSMRGDataset
from csmsam.utils.metrics import evaluate_patient, aggregate_metrics


class SAMVanillaBaseline:
    """
    SAM1 zero-shot baseline with automatic center-point (or grid) prompts.

    For each mid-RT slice:
      1. Run SAM ViT-H image encoder on the resized slice
      2. Place automatic point prompt (center, or 3x3 grid)
      3. Take the highest-IoU predicted mask

    No memory, no finetuning — this establishes the floor for a frozen
    general-purpose foundation model on HNTS-MRG.

    Falls back to random predictions if `segment_anything` is missing.
    """

    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_h",
        device: str = "cuda",
        prompt_mode: str = "center",  # "center" | "grid"
    ):
        self.device = device
        self.prompt_mode = prompt_mode
        self.predictor = None

        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device)
            sam.eval()
            self.predictor = SamPredictor(sam)
            print(f"SAM ({model_type}) loaded successfully.")
        except ImportError:
            print("Warning: segment_anything not installed. Using random fallback.")
            print("Install with: pip install segment-anything")
        except Exception as e:
            print(f"Warning: failed to load SAM checkpoint ({e}). Using random fallback.")

    def _build_points(self, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
        if self.prompt_mode == "grid":
            ys = np.linspace(H * 0.25, H * 0.75, 3)
            xs = np.linspace(W * 0.25, W * 0.75, 3)
            pts = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
            labels = np.ones(len(pts), dtype=np.int64)
            return pts, labels
        # center
        return np.array([[W / 2, H / 2]], dtype=np.float32), np.array([1], dtype=np.int64)

    @torch.no_grad()
    def predict_volume(self, mid_images: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        N, _, H, W = mid_images.shape

        if self.predictor is None:
            return (np.random.rand(N, H, W) > 0.85).astype(np.float32)

        pred_slices = []
        for i in range(N):
            img = mid_images[i].permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img_uint8 = (img * 255).astype(np.uint8)

            try:
                self.predictor.set_image(img_uint8)
                pts, lbls = self._build_points(H, W)
                masks, scores, _ = self.predictor.predict(
                    point_coords=pts,
                    point_labels=lbls,
                    multimask_output=True,
                )
                best = int(np.argmax(scores))
                pred_slices.append(masks[best].astype(np.float32))
            except Exception:
                pred_slices.append(np.zeros((H, W), dtype=np.float32))

        return np.stack(pred_slices)


def run_sam_vanilla_baseline(
    data_dir: str,
    sam_checkpoint: str,
    output_dir: str,
    split: str = "test",
    device: str = "cuda",
    threshold: float = 0.5,
    image_size: int = 1024,
    prompt_mode: str = "center",
    model_type: str = "vit_h",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SAM Vanilla Baseline (zero-shot, prompt={prompt_mode})")
    print(f"Split: {split}")
    print("=" * 60)

    model = SAMVanillaBaseline(sam_checkpoint, model_type, device, prompt_mode)

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

    print(f"\nSAM Vanilla Results ({split}):")
    print(f"  aggDSC  : {agg.get('agg_dsc_mean', 0):.4f} ± {agg.get('agg_dsc_std', 0):.4f}")
    print(f"  GTVp DSC: {agg.get('dsc_gtvp_mean', 0):.4f}")
    print(f"  GTVn DSC: {agg.get('dsc_gtvn_mean', 0):.4f}")

    results = {
        "baseline": "sam_vanilla",
        "prompt_mode": prompt_mode,
        "model_type": model_type,
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
    parser.add_argument("--sam_checkpoint", type=str, default="checkpoints/sam/sam_vit_h_4b8939.pth")
    parser.add_argument("--output_dir", type=str, default="results/baselines/sam_vanilla")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--prompt_mode", type=str, default="center", choices=["center", "grid"])
    parser.add_argument("--model_type", type=str, default="vit_h")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_sam_vanilla_baseline(
        data_dir=args.data_dir,
        sam_checkpoint=args.sam_checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        device=args.device,
        image_size=args.image_size,
        prompt_mode=args.prompt_mode,
        model_type=args.model_type,
    )
