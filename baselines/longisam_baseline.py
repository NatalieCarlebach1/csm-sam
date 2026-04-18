"""
LongiSAM Baseline — SAM2 with image-level (pre, mid) concatenation, no memory attention.

This is the KEY ABLATION for CSM-SAM. It uses SAM2 but replaces cross-session
memory attention with naive image-level concatenation:

    For each mid-RT slice i:
      - Take the aligned pre-RT slice j = min(i, N_pre - 1)
      - Concatenate the two SAM2 RGB tensors along the channel axis
      - Project 6 channels --> 3 via a learned 1x1 conv
      - Frozen SAM2 encoder --> mask decoder (mask decoder is finetuned)

Uniqueness vs CSM-SAM:
    CSM-SAM's contribution is the CrossSessionMemoryAttention module. This
    baseline reduces that to a 1x1 conv that averages pixels across the
    two visits.

Requirements:
    sam2 package (https://github.com/facebookresearch/sam2) with checkpoint.

Usage:
    python baselines/longisam_baseline.py \
        --data_dir data/processed \
        --sam2_checkpoint checkpoints/sam2/sam2.1_hiera_large.pt \
        --output_dir results/experiments/trained_baselines/longisam \
        --epochs 50 --lr 1e-4
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.baseline_trainer import train_and_evaluate


class LongiSAMModel(nn.Module):
    """SAM2 with a learned 6->3 channel fusion conv. Only fuse_conv + mask decoder trained."""

    def __init__(self, sam2_checkpoint: str, device: str = "cuda"):
        super().__init__()
        self.device_str = device

        # 6 -> 3 channel fusion
        self.fuse_conv = nn.Conv2d(6, 3, kernel_size=1, bias=True)
        with torch.no_grad():
            w = torch.zeros(3, 6, 1, 1)
            for c in range(3):
                w[c, c, 0, 0] = 0.5
                w[c, c + 3, 0, 0] = 0.5
            self.fuse_conv.weight.copy_(w)
            self.fuse_conv.bias.zero_()

        # Load SAM2
        from sam2.build_sam import build_sam2
        self.sam2 = build_sam2(
            "configs/sam2.1/sam2.1_hiera_l", sam2_checkpoint, device=device
        )

        # Freeze everything in SAM2
        for p in self.sam2.parameters():
            p.requires_grad = False

        # Unfreeze mask decoder
        for p in self.sam2.sam_mask_decoder.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accept (B, 6, H, W) = [pre_rgb, mid_rgb]. Returns (B, 1, H, W) logits."""
        B, _, H, W = x.shape
        fused = self.fuse_conv(x)  # (B, 3, H, W)

        outputs = []
        for b in range(B):
            img = fused[b:b + 1]
            backbone_out = self.sam2.forward_image(img)
            features = backbone_out["vision_features"]

            point_coords = torch.tensor(
                [[[W / 2, H / 2]]], dtype=torch.float, device=x.device
            )
            point_labels = torch.ones(1, 1, dtype=torch.int, device=x.device)

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
            mask = F.interpolate(
                low_res_masks, size=(H, W), mode="bilinear", align_corners=False
            )
            outputs.append(mask)

        return torch.cat(outputs, dim=0)  # (B, 1, H, W)


def build_model(sam2_checkpoint: str, device: str = "cuda") -> nn.Module:
    model = LongiSAMModel(sam2_checkpoint=sam2_checkpoint, device=device).to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LongiSAM baseline: SAM2 + image concat, no memory attention."
    )
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--sam2_checkpoint", type=str, default="checkpoints/sam2/sam2.1_hiera_large.pt")
    parser.add_argument("--output_dir", type=str, default="results/experiments/trained_baselines/longisam")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    model = build_model(sam2_checkpoint=args.sam2_checkpoint, device=args.device)

    # Only train fuse_conv + mask decoder
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    train_and_evaluate(
        model=model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        model_name="LongiSAM",
        uses_pre=True,
        trainable_params=trainable_params,
    )
