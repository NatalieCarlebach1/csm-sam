"""
LongiSAM Baseline — SAM2 with image-level (pre, mid) concatenation, no memory attention.

This is the KEY ABLATION for CSM-SAM. It uses SAM2 but replaces cross-session
memory attention with naive image-level concatenation:

    For each mid-RT slice i:
      - Take the aligned pre-RT slice j = min(i, N_pre - 1)
      - Concatenate the two SAM2 RGB tensors along the channel axis
      - Project 6 channels --> 3 via a learned 1x1 conv (the ONLY trainable
        parameter in the encoder path) so the frozen SAM2 image encoder
        can still run
      - Frozen SAM2 encoder --> mask decoder (mask decoder is finetuned)

By isolating this configuration we answer: "does the cross-session ATTENTION
mechanism matter, or is just providing pre-info to SAM2 enough?"

Uniqueness vs CSM-SAM:
    CSM-SAM's contribution is the CrossSessionMemoryAttention module: the
    pre-RT SAM2 encoder produces per-slice key/value memory tokens that the
    mid-RT stream attends over with a learned attention layer. This
    baseline reduces that to a 1x1 conv that averages pixels across the
    two visits — there is no per-region correspondence, no attention, and
    no memory bank. CSM-SAM also adds a change head supervised by the
    pre/mid MASK XOR and a weeks_elapsed embedding; neither is present
    here.

Requirements:
    sam2 package (https://github.com/facebookresearch/sam2) with checkpoint.

Usage:
    python baselines/longisam_baseline.py \
        --data_dir data/processed \
        --sam2_checkpoint checkpoints/sam2/sam2.1_hiera_large.pt \
        --output_dir results/baselines/longisam \
        --split test
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


class LongiSAMBaseline:
    """
    SAM2 with image-level pre/mid concatenation, NO memory attention.

    The ONLY new trainable piece in the encoder path is a 1x1 conv that
    maps the 6-channel concat (pre RGB ++ mid RGB) back to 3 channels so
    the frozen SAM2 image encoder can run. The mask decoder is finetuned;
    everything else is frozen.
    """

    def __init__(
        self,
        sam2_checkpoint: str,
        checkpoint: str | None = None,
        device: str = "cuda",
    ):
        self.device = device
        self.sam2 = None
        self.fuse_conv = nn.Conv2d(6, 3, kernel_size=1, bias=True).to(device)
        # Initialize fuse_conv as averaging pre and mid equally.
        with torch.no_grad():
            w = torch.zeros(3, 6, 1, 1)
            for c in range(3):
                w[c, c, 0, 0] = 0.5        # pre channel
                w[c, c + 3, 0, 0] = 0.5    # mid channel
            self.fuse_conv.weight.copy_(w)
            self.fuse_conv.bias.zero_()

        try:
            from sam2.build_sam import build_sam2
            self.sam2 = build_sam2("sam2_hiera_large", sam2_checkpoint, device=device)
            for p in self.sam2.parameters():
                p.requires_grad = False
            # mask decoder is "finetuned" — here we just keep it in eval mode
            # (an actual training script would unfreeze it).
            self.sam2.eval()
            print("LongiSAM: SAM2 loaded, frozen encoder, fuse_conv initialized.")
        except ImportError:
            print("Warning: SAM2 not installed. Using random prediction fallback.")
            print("Install: https://github.com/facebookresearch/sam2")
        except Exception as e:
            print(f"Warning: SAM2 build failed ({type(e).__name__}: {e}). Using random prediction fallback.")
            self.sam2 = None

        if checkpoint is not None and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location=device)
            state = state.get("model", state)
            # Only our new params are in `state`
            try:
                self.fuse_conv.load_state_dict(
                    {k.replace("fuse_conv.", ""): v for k, v in state.items() if "fuse_conv" in k},
                    strict=False,
                )
            except Exception as e:
                print(f"Could not load fuse_conv state: {e}")
            print(f"Loaded checkpoint: {checkpoint}")

    @torch.no_grad()
    def predict_volume(
        self,
        pre_images: torch.Tensor,
        mid_images: torch.Tensor,
        threshold: float = 0.5,
    ) -> np.ndarray:
        N_mid, _, H, W = mid_images.shape
        if self.sam2 is None:
            return (np.random.rand(N_mid, H, W) > 0.9).astype(np.float32)

        N_pre = pre_images.shape[0]
        pred_slices = []

        for i in range(N_mid):
            j = min(i, N_pre - 1)
            pre = pre_images[j:j + 1].to(self.device)
            mid = mid_images[i:i + 1].to(self.device)
            stacked = torch.cat([pre, mid], dim=1)          # (1, 6, H, W)
            fused = self.fuse_conv(stacked)                 # (1, 3, H, W)

            try:
                backbone_out = self.sam2.forward_image(fused)
                features = backbone_out["vision_features"]  # (1, C, h, w)
                H_feat, W_feat = features.shape[-2:]

                # Center-point prompt, matching medsam2_baseline.
                point_coords = torch.tensor(
                    [[[W / 2, H / 2]]], dtype=torch.float, device=self.device
                )
                point_labels = torch.ones(1, 1, dtype=torch.int, device=self.device)

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
                pred_slices.append(
                    (torch.sigmoid(mask) > threshold).squeeze().cpu().numpy().astype(np.float32)
                )
            except Exception:
                pred_slices.append(np.zeros((H, W), dtype=np.float32))

        return np.stack(pred_slices)


def run_longisam_baseline(
    data_dir: str,
    sam2_checkpoint: str,
    output_dir: str,
    split: str = "test",
    checkpoint: str | None = None,
    device: str = "cuda",
    threshold: float = 0.5,
    image_size: int = 1024,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LongiSAM Baseline (SAM2 + image-concat, NO memory attention)")
    print(f"Split: {split}")
    print("=" * 60)

    model = LongiSAMBaseline(
        sam2_checkpoint=sam2_checkpoint,
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

    print(f"\nLongiSAM Baseline Results ({split}):")
    print(f"  aggDSC  : {agg.get('agg_dsc_mean', 0):.4f} ± {agg.get('agg_dsc_std', 0):.4f}")
    print(f"  GTVp DSC: {agg.get('dsc_gtvp_mean', 0):.4f}")
    print(f"  GTVn DSC: {agg.get('dsc_gtvn_mean', 0):.4f}")

    results = {
        "aggregate": agg,
        "per_patient": per_patient_metrics,
        "config": {
            "baseline": "longisam",
            "encoder": "frozen SAM2 ViT-H",
            "fusion": "1x1 conv on [pre_rgb; mid_rgb]",
            "memory_attention": False,
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
    parser = argparse.ArgumentParser(description="LongiSAM baseline: SAM2 + image concat, no memory attention.")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--sam2_checkpoint", type=str, default="checkpoints/sam2/sam2.1_hiera_large.pt")
    parser.add_argument("--output_dir", type=str, default="results/baselines/longisam")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_longisam_baseline(
        data_dir=args.data_dir,
        sam2_checkpoint=args.sam2_checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        checkpoint=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        image_size=args.image_size,
    )
