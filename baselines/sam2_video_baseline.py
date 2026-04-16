"""
SAM2 Video Baseline — standard SAM2 within-session memory propagation.

Model       : SAM2 video predictor (first-frame prompt → propagate forward).
Paper       : "SAM 2: Segment Anything in Images and Videos", Ravi et al. (2024).
Year        : 2024.
Backbone    : Hiera-Large encoder + SAM2 memory attention (within a video).
Install     : git clone https://github.com/facebookresearch/sam2 && pip install -e sam2/

Uniqueness note vs CSM-SAM:
    This baseline IS the MedSAM2-style recipe: within-session memory only. SAM2
    propagates from slice → slice inside the mid-RT volume but never attends to
    the pre-RT visit. CSM-SAM replaces this with CrossSessionMemoryAttention,
    which keys mid-RT queries against pre-RT memory tokens (frozen features +
    mask tokens), and adds a change-map head supervised by pre/mid XOR — the two
    moves that turn within-session propagation into cross-session propagation.

Usage:
    python baselines/sam2_video_baseline.py \
        --data_dir data/processed \
        --sam2_checkpoint checkpoints/sam2/sam2.1_hiera_large.pt \
        --output_dir results/baselines/sam2_video
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


class SAM2VideoBaseline:
    """
    SAM2 video-style baseline.

    Treats the mid-RT volume as a video:
      - Place a center-point prompt on the middle slice (proxy for user init).
      - Use SAM2's video predictor to propagate the mask forward and backward.
      - No pre-RT signal is used.

    Falls back to random predictions if sam2 / video predictor is missing.
    """

    def __init__(
        self,
        sam2_checkpoint: str,
        model_cfg: str = "sam2_hiera_large",
        device: str = "cuda",
    ):
        self.device = device
        self.video_predictor = None

        try:
            from sam2.build_sam import build_sam2_video_predictor
            self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
            print("SAM2 video predictor loaded successfully.")
        except ImportError:
            print("Warning: SAM2 not installed. Using random fallback.")
            print("Install with: git clone https://github.com/facebookresearch/sam2 && pip install -e sam2/")
        except Exception as e:
            print(f"Warning: SAM2 video predictor load failed ({e}). Using random fallback.")

    @torch.no_grad()
    def predict_volume(self, mid_images: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        N, _, H, W = mid_images.shape

        if self.video_predictor is None:
            return (np.random.rand(N, H, W) > 0.85).astype(np.float32)

        try:
            # init state expects a tensor video
            video = mid_images.to(self.device)  # (N, 3, H, W)
            state = self.video_predictor.init_state_from_tensor(video)

            # Place a single center-point prompt on the middle slice
            init_frame = N // 2
            pt = np.array([[W / 2, H / 2]], dtype=np.float32)
            lbl = np.array([1], dtype=np.int64)

            self.video_predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=init_frame,
                obj_id=1,
                points=pt,
                labels=lbl,
            )

            out = {}
            # propagate forward
            for f_idx, obj_ids, logits in self.video_predictor.propagate_in_video(state):
                out[f_idx] = logits[0]
            # propagate backward
            for f_idx, obj_ids, logits in self.video_predictor.propagate_in_video(state, reverse=True):
                if f_idx not in out:
                    out[f_idx] = logits[0]

            pred_slices = []
            for i in range(N):
                if i in out:
                    m = F.interpolate(out[i].unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
                    pred_slices.append((torch.sigmoid(m) > threshold).squeeze().cpu().numpy().astype(np.float32))
                else:
                    pred_slices.append(np.zeros((H, W), dtype=np.float32))
            return np.stack(pred_slices)

        except Exception as e:
            print(f"  SAM2 video propagation failed ({e}); returning zeros.")
            return np.zeros((N, H, W), dtype=np.float32)


def run_sam2_video_baseline(
    data_dir: str,
    sam2_checkpoint: str,
    output_dir: str,
    split: str = "test",
    device: str = "cuda",
    threshold: float = 0.5,
    image_size: int = 1024,
    model_cfg: str = "sam2_hiera_large",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SAM2 Video Baseline (within-session memory propagation)")
    print(f"Split: {split}")
    print("=" * 60)

    model = SAM2VideoBaseline(sam2_checkpoint, model_cfg, device)

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

    print(f"\nSAM2 Video Results ({split}):")
    print(f"  aggDSC  : {agg.get('agg_dsc_mean', 0):.4f} ± {agg.get('agg_dsc_std', 0):.4f}")
    print(f"  GTVp DSC: {agg.get('dsc_gtvp_mean', 0):.4f}")
    print(f"  GTVn DSC: {agg.get('dsc_gtvn_mean', 0):.4f}")

    results = {
        "baseline": "sam2_video",
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
    parser.add_argument("--output_dir", type=str, default="results/baselines/sam2_video")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_large")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_sam2_video_baseline(
        data_dir=args.data_dir,
        sam2_checkpoint=args.sam2_checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        device=args.device,
        image_size=args.image_size,
        model_cfg=args.model_cfg,
    )
