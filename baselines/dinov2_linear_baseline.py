"""
DINOv2 + Linear Probe Baseline — frozen self-supervised ViT features + tiny decoder.

Model       : DINOv2 ViT-L/14, Oquab et al., Meta AI.
Paper       : "DINOv2: Learning Robust Visual Features without Supervision" (2023).
Year        : 2023 (Apr 2023 arXiv; TMLR 2024).
Backbone    : ViT-L/14 (frozen). Features are self-supervised on LVD-142M.
Install     : pip install torch torchvision
              # model loaded via torch.hub: torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

Uniqueness note vs CSM-SAM:
    DINOv2 provides a strong frozen visual backbone but lacks any temporal or
    cross-visit mechanism — a single-timepoint linear probe must answer "is this
    pixel tumor?" from appearance alone. CSM-SAM instead conditions mid-RT decoding
    on pre-RT memory via CrossSessionMemoryAttention and supervises a change-map
    head with pre/mid XOR; this baseline ablates both cross-session context and
    the SAM2 mask-decoder prior.

Usage:
    python baselines/dinov2_linear_baseline.py \
        --data_dir data/processed \
        --output_dir results/baselines/dinov2_linear
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
from csmsam.utils.metrics import evaluate_patient, aggregate_metrics


class DINOv2LinearBaseline:
    """
    DINOv2 ViT-L/14 frozen encoder + a small linear/conv decoder.

    Produces per-pixel segmentation logits by projecting patch tokens back to
    spatial resolution and passing through a 1x1 + upsample decoder head.

    In this baseline we use the decoder UNTRAINED (zero-init) as a pure
    "foundation-model prior" readout; swapping in a trained linear probe is a
    one-line extension. Raises RuntimeError if torch.hub model is unavailable.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "dinov2_vitl14",
        patch_size: int = 14,
        feat_dim: int = 1024,  # vit-l
    ):
        self.device = device
        self.patch_size = patch_size
        self.feat_dim = feat_dim

        try:
            self.encoder = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
            self.encoder.to(device).eval()
            self.decoder = nn.Sequential(
                nn.Conv2d(feat_dim, 256, 1),
                nn.GELU(),
                nn.Conv2d(256, 1, 1),
            ).to(device)
            self.decoder.eval()
            print(f"DINOv2 ({model_name}) loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"DINOv2 failed to load: {e}\n"
                f"Install: pip install torch torchvision\n"
                f"Model loaded via: torch.hub.load('facebookresearch/dinov2', '{model_name}')"
            ) from e

    @torch.no_grad()
    def predict_volume(self, mid_images: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        N, _, H, W = mid_images.shape

        # DINOv2 requires input sizes divisible by patch_size
        Hn = (H // self.patch_size) * self.patch_size
        Wn = (W // self.patch_size) * self.patch_size

        pred_slices = []
        for i in range(N):
            img = mid_images[i].unsqueeze(0).to(self.device)
            img = F.interpolate(img, size=(Hn, Wn), mode="bilinear", align_corners=False)

            try:
                out = self.encoder.forward_features(img)
                tokens = out["x_norm_patchtokens"]  # (1, N_patches, C)
                h_p = Hn // self.patch_size
                w_p = Wn // self.patch_size
                feat = tokens.transpose(1, 2).reshape(1, self.feat_dim, h_p, w_p)
                logits = self.decoder(feat)
                logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
                pred_slices.append((torch.sigmoid(logits) > threshold).squeeze().cpu().numpy().astype(np.float32))
            except Exception:
                pred_slices.append(np.zeros((H, W), dtype=np.float32))

        return np.stack(pred_slices)


def run_dinov2_linear_baseline(
    data_dir: str,
    output_dir: str,
    split: str = "test",
    device: str = "cuda",
    threshold: float = 0.5,
    image_size: int = 1024,
    model_name: str = "dinov2_vitl14",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DINOv2 Linear Probe Baseline (frozen SSL features)")
    print(f"Split: {split}")
    print("=" * 60)

    model = DINOv2LinearBaseline(device=device, model_name=model_name)

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

    print(f"\nDINOv2 Linear Results ({split}):")
    print(f"  aggDSC  : {agg.get('agg_dsc_mean', 0):.4f} ± {agg.get('agg_dsc_std', 0):.4f}")
    print(f"  GTVp DSC: {agg.get('dsc_gtvp_mean', 0):.4f}")
    print(f"  GTVn DSC: {agg.get('dsc_gtvn_mean', 0):.4f}")

    results = {
        "baseline": "dinov2_linear",
        "model_name": model_name,
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
    parser.add_argument("--output_dir", type=str, default="results/baselines/dinov2_linear")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--model_name", type=str, default="dinov2_vitl14")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_dinov2_linear_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        device=args.device,
        image_size=args.image_size,
        model_name=args.model_name,
    )
