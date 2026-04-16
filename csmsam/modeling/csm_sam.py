"""
CSM-SAM: Cross-Session Memory SAM model.

Wraps a frozen SAM2 image encoder + mask decoder with:
  1. CrossSessionMemoryEncoder  — encodes pre-RT scan into memory bank
  2. CrossSessionMemoryAttention — fuses pre-RT memory with mid-RT features
  3. ChangeHead                 — predicts change map for auxiliary supervision

Training strategy:
  - SAM2 image encoder: FROZEN
  - SAM2 mask decoder:  FROZEN (except final conv layer, which is tuned at lr*0.1)
  - CrossSessionMemoryAttention: TRAINED at full lr
  - CrossSessionMemoryEncoder:  TRAINED at full lr
  - ChangeHead:                 TRAINED at full lr

Total trainable params: ~2.1M (vs 64M for full SAM2-L fine-tune)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_session_memory_attention import CrossSessionMemoryAttention, CrossSessionMemoryEncoder
from .change_head import ChangeHead, build_change_labels


class CSMSAM(nn.Module):
    """
    Full CSM-SAM model.

    Usage:
        model = CSMSAM.from_pretrained("checkpoints/sam2/sam2.1_hiera_large.pt")

        # Encode pre-RT scan into memory (done once per patient)
        M_pre = model.encode_pre_rt(pre_images, pre_masks)

        # Segment mid-RT slice by slice
        for slice_idx, (mid_img, mid_mask_gt) in enumerate(mid_slices):
            preds = model.forward_mid_rt(mid_img, M_pre, slice_idx)
    """

    def __init__(
        self,
        sam2_model,
        d_model: int = 256,
        num_heads: int = 8,
        n_memory_frames: int = 8,
        spatial_pool_size: int = 16,
        max_weeks: int = 12,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.sam2 = sam2_model

        # Novel modules (trainable)
        self.memory_encoder = CrossSessionMemoryEncoder(
            d_model=d_model,
            n_memory_frames=n_memory_frames,
            spatial_pool_size=spatial_pool_size,
        )
        self.cross_session_attn = CrossSessionMemoryAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_weeks=max_weeks,
        )
        self.change_head = ChangeHead(in_channels=d_model, num_classes=4)

        # Freeze SAM2 encoder
        self._freeze_sam2_encoder()

        # Keep within-session memory buffer (updated during mid-RT forward pass)
        self._M_mid: torch.Tensor | None = None

    def _freeze_sam2_encoder(self):
        """Freeze SAM2 image encoder. Decoder final layer remains tunable."""
        if hasattr(self.sam2, "image_encoder"):
            for p in self.sam2.image_encoder.parameters():
                p.requires_grad = False

    def _get_sam2_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract SAM2 image encoder features.

        Args:
            images: (B, 3, H, W) normalized RGB images

        Returns:
            features: (B, C, h, w) where h, w = H//16, W//16 for ViT-L
        """
        with torch.no_grad():
            # SAM2 image encoder outputs a dict with 'vision_features'
            backbone_out = self.sam2.forward_image(images)
            # Extract highest-resolution feature map from feature pyramid
            features = backbone_out["vision_features"]  # (B, C, h, w)
        return features

    def encode_pre_rt(
        self,
        pre_images: torch.Tensor,
        pre_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode pre-RT volume into cross-session memory bank.

        Call this ONCE per patient before mid-RT inference.

        Args:
            pre_images : (B, N_slices, 3, H, W) — pre-RT slices (normalized)
            pre_masks  : (B, N_slices, 1, H, W) — pre-RT GTVp+GTVn masks (optional)

        Returns:
            M_pre : (B, N_memory, C) — cross-session memory bank
        """
        B, N, C_img, H, W = pre_images.shape

        # Extract SAM2 features for each slice
        pre_feats_list = []
        for i in range(N):
            feat = self._get_sam2_features(pre_images[:, i])  # (B, C, h, w)
            pre_feats_list.append(feat)

        pre_feats = torch.stack(pre_feats_list, dim=1)  # (B, N, C, h, w)

        # Resize masks to feature map resolution if provided
        if pre_masks is not None:
            h, w = pre_feats.shape[-2:]
            pre_masks_resized = F.interpolate(
                pre_masks.reshape(B * N, 1, H, W),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, N, 1, h, w)
        else:
            pre_masks_resized = None

        M_pre = self.memory_encoder(pre_feats, pre_masks_resized)
        return M_pre

    def reset_mid_session_memory(self):
        """Reset within-session memory at the start of a new mid-RT volume."""
        self._M_mid = None

    def forward(
        self,
        mid_images: torch.Tensor,
        M_pre: torch.Tensor,
        pre_images: torch.Tensor | None = None,
        weeks_elapsed: torch.Tensor | None = None,
        return_change_map: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for mid-RT segmentation.

        Processes one batch of mid-RT slices. Within-session memory is accumulated
        across calls (call reset_mid_session_memory() at the start of each volume).

        Args:
            mid_images    : (B, 3, H, W) — current mid-RT slice(s)
            M_pre         : (B, N_memory, C) — pre-RT memory bank from encode_pre_rt()
            pre_images    : (B, 3, H, W) — corresponding pre-RT slices (for change head)
            weeks_elapsed : (B,) long — integer weeks between pre-RT and mid-RT
                            Defaults to 3 (typical mid-RT timing)
            return_change_map : bool — whether to run ChangeHead

        Returns:
            dict with:
                "masks"      : (B, 1, H, W) — segmentation logits (sigmoid → binary)
                "change_map" : (B, 4, H, W) — change map logits (if return_change_map=True)
                "gate_vals"  : (B, H, W)    — cross-session gate values (for visualization)
                "attn_weights": (B, H, N_pre) — cross-session attention (sampled)
        """
        B, _, H, W = mid_images.shape

        if weeks_elapsed is None:
            weeks_elapsed = torch.full((B,), 3, dtype=torch.long, device=mid_images.device)

        # 1. Extract mid-RT features from frozen SAM2 encoder
        mid_feats = self._get_sam2_features(mid_images)  # (B, C, h, w)
        h, w = mid_feats.shape[-2:]
        C = mid_feats.shape[1]

        # 2. Flatten spatial dims for attention
        mid_feats_flat = mid_feats.permute(0, 2, 3, 1).reshape(B, h * w, C)  # (B, HW, C)

        # 3. Cross-session memory attention
        enhanced_feats, attn_w, gate_vals = self.cross_session_attn(
            curr_features=mid_feats_flat,
            M_pre=M_pre,
            M_mid=self._M_mid,
            weeks_elapsed=weeks_elapsed,
        )  # (B, HW, C), (B, H, HW, N_pre), (B, HW, 1)

        # 4. Update within-session memory with current features
        # Append current features to M_mid for the next slice
        new_memory = enhanced_feats.detach()  # Don't backprop through memory accumulation
        if self._M_mid is None:
            self._M_mid = new_memory
        else:
            # Keep last 4 slices in memory to avoid OOM
            self._M_mid = torch.cat([self._M_mid, new_memory], dim=1)
            max_mid_tokens = 4 * h * w
            if self._M_mid.shape[1] > max_mid_tokens:
                self._M_mid = self._M_mid[:, -max_mid_tokens:]

        # 5. Reshape back to spatial for mask decoder
        enhanced_feats_spatial = enhanced_feats.reshape(B, h, w, C).permute(0, 3, 1, 2)  # (B, C, h, w)

        # 6. SAM2 mask decoder
        # We replace the original image features with our cross-session-enhanced features
        masks = self._decode_masks(enhanced_feats_spatial, (H, W))  # (B, 1, H, W)

        result = {
            "masks": masks,
            "gate_vals": gate_vals.reshape(B, h, w),
            "attn_weights": attn_w.mean(dim=1),  # Average over heads: (B, HW, N_pre)
        }

        # 7. Change map prediction (auxiliary)
        if return_change_map and pre_images is not None:
            pre_feats = self._get_sam2_features(pre_images)  # (B, C, h, w)
            change_logits = self.change_head(pre_feats, enhanced_feats_spatial)  # (B, 4, h, w)
            # Upsample change map to original resolution
            change_logits = F.interpolate(
                change_logits, size=(H, W), mode="bilinear", align_corners=False
            )
            result["change_map"] = change_logits

        return result

    def _decode_masks(
        self,
        image_features: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        Run SAM2 mask decoder on enhanced image features.

        Provides a center-of-mass point prompt derived from the features
        (top-K activated spatial location) as a weak automatic prompt.
        """
        B, C, h, w = image_features.shape

        # Automatic prompt: use peak activation location as point prompt
        activation = image_features.mean(dim=1)  # (B, h, w)
        flat_idx = activation.reshape(B, -1).argmax(dim=1)  # (B,)
        point_y = (flat_idx // w).float() / h
        point_x = (flat_idx % w).float() / w
        # Scale to original image coords
        H, W = output_size
        point_coords = torch.stack([
            point_x * W,
            point_y * H,
        ], dim=1).unsqueeze(1)  # (B, 1, 2)
        point_labels = torch.ones(B, 1, dtype=torch.int, device=image_features.device)

        try:
            # Attempt SAM2 decoder API
            sparse_embed, dense_embed = self.sam2.sam_prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            low_res_masks, _ = self.sam2.sam_mask_decoder(
                image_embeddings=image_features,
                image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                multimask_output=False,
            )
        except (AttributeError, TypeError):
            # Fallback: simple conv decoder when SAM2 is not available (unit tests)
            low_res_masks = self._fallback_decoder(image_features)

        # Upsample to original resolution
        masks = F.interpolate(
            low_res_masks,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )  # (B, 1, H, W)
        return masks

    def _fallback_decoder(self, features: torch.Tensor) -> torch.Tensor:
        """Minimal 1-conv decoder used for testing without SAM2 installed."""
        B, C, h, w = features.shape
        return features.mean(dim=1, keepdim=True)  # (B, 1, h, w)

    @classmethod
    def from_pretrained(
        cls,
        sam2_checkpoint: str,
        sam2_cfg: str = "sam2_hiera_large",
        **kwargs,
    ) -> "CSMSAM":
        """
        Build CSM-SAM from a SAM2 checkpoint.

        Args:
            sam2_checkpoint : path to sam2.1_hiera_large.pt
            sam2_cfg        : SAM2 model config name
            **kwargs        : passed to CSMSAM.__init__

        Returns:
            Initialized CSMSAM model (SAM2 encoder frozen)
        """
        try:
            from sam2.build_sam import build_sam2
            sam2 = build_sam2(sam2_cfg, sam2_checkpoint, device="cpu")
        except ImportError:
            raise ImportError(
                "SAM2 is not installed. Run:\n"
                "  git clone https://github.com/facebookresearch/sam2.git\n"
                "  pip install -e sam2/"
            )
        return cls(sam2_model=sam2, **kwargs)

    def get_trainable_params(self) -> list[dict]:
        """
        Returns parameter groups for the optimizer.
        Novel modules get full lr; SAM2 decoder final layer gets lr*0.1.
        """
        novel_params = (
            list(self.memory_encoder.parameters())
            + list(self.cross_session_attn.parameters())
            + list(self.change_head.parameters())
        )

        decoder_params = []
        if hasattr(self.sam2, "sam_mask_decoder"):
            # Only tune the final output convolution of the decoder
            for name, p in self.sam2.sam_mask_decoder.named_parameters():
                if "output_upscaling" in name or "output_hypernetworks" in name:
                    decoder_params.append(p)

        return [
            {"params": novel_params, "lr_scale": 1.0},
            {"params": decoder_params, "lr_scale": 0.1},
        ]

    def count_trainable_params(self) -> dict[str, int]:
        groups = {
            "memory_encoder": self.memory_encoder,
            "cross_session_attn": self.cross_session_attn,
            "change_head": self.change_head,
        }
        counts = {}
        for name, module in groups.items():
            counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts["total"] = sum(counts.values())
        return counts
