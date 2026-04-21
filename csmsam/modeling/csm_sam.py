"""
CSM-SAM: Cross-Session Memory SAM model.

Wraps a frozen SAM2 image encoder + mask decoder with:
  1. CrossSessionMemoryEncoder   — encodes pre-RT scan into memory bank
  2. CrossSessionMemoryAttention — fuses pre-RT memory with mid-RT features
  3. ChangeHead                  — predicts change map for auxiliary supervision

Key design points:
  - Output is DUAL-CHANNEL: (B, 2, H, W) with channel 0 = GTVp, channel 1 = GTVn.
    The decoder is invoked twice, once per structure, conditioned on the
    corresponding pre-RT mask as a mask prompt. This is what makes the
    primary metric (aggDSC = (DSC_GTVp + DSC_GTVn) / 2) meaningful.
  - Pre-RT mask is used as the dense prompt to the SAM2 prompt encoder.
    The argmax-of-activations heuristic is only used when the pre-RT mask
    for a given structure is empty on that slice.
  - Within-session memory M_mid accumulates WITHOUT .detach(), so gradients
    propagate through the gate and within-session attention during sequence
    training.

Training strategy:
  - SAM2 image encoder: FROZEN
  - SAM2 mask decoder:  FROZEN (except final upscaling layers, lr*0.1)
  - CrossSessionMemoryAttention: TRAINED at full lr
  - CrossSessionMemoryEncoder:   TRAINED at full lr
  - ChangeHead:                  TRAINED at full lr
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_session_memory_attention import CrossSessionMemoryAttention, CrossSessionMemoryEncoder
from .change_head import ChangeHead, build_change_labels
from .retrieval import (
    CrossPatientBank,
    CrossPatientRetrieval,
    compute_pre_summary,
    compute_change_template,
)


class CSMSAM(nn.Module):
    """Full CSM-SAM model producing per-structure (GTVp, GTVn) predictions."""

    def __init__(
        self,
        sam2_model,
        d_model: int = 256,
        num_heads: int = 8,
        n_memory_frames: int = 8,
        spatial_pool_size: int = 16,
        max_weeks: int = 12,
        dropout: float = 0.0,
        memory_bank_max_slices: int = 4,
        temporal_encoder_type: str = "continuous",
        temporal_hidden_dim: int = 128,
        temporal_n_frequencies: int = 6,
        in_chans: int = 3,
        use_cross_patient_retrieval: bool = True,
        retrieval_k: int = 5,
        retrieval_n_tokens: int = 16,
        retrieval_gate_init: float = 0.0,
    ):
        super().__init__()
        self.sam2 = sam2_model
        self.memory_bank_max_slices = memory_bank_max_slices
        self.use_cross_patient_retrieval = use_cross_patient_retrieval
        self.retrieval_n_tokens = retrieval_n_tokens

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
            temporal_encoder_type=temporal_encoder_type,
            temporal_hidden_dim=temporal_hidden_dim,
            temporal_n_frequencies=temporal_n_frequencies,
        )
        self.change_head = ChangeHead(in_channels=d_model, num_classes=4)

        # Cross-patient retrieval (optional).
        if use_cross_patient_retrieval:
            self.retrieval = CrossPatientRetrieval(
                d_model=d_model,
                k=retrieval_k,
                gate_init=retrieval_gate_init,
                dropout=dropout,
            )
        else:
            self.retrieval = None
        self.patient_bank: CrossPatientBank | None = None

        self._freeze_sam2_encoder()

        # Within-session memory buffer (updated during mid-RT forward pass)
        self._M_mid: torch.Tensor | None = None

    def _freeze_sam2_encoder(self):
        if hasattr(self.sam2, "image_encoder"):
            for p in self.sam2.image_encoder.parameters():
                p.requires_grad = False

    def _get_sam2_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract SAM2 image encoder features. (B, 3, H, W) -> (B, C, h, w)."""
        with torch.no_grad():
            if hasattr(self.sam2, "forward_image"):
                backbone_out = self.sam2.forward_image(images)
                features = backbone_out["vision_features"]
            else:
                # Fallback: a random-init projection so downstream code/tests run.
                features = self._fallback_encoder(images)
        return features

    def _fallback_encoder(self, images: torch.Tensor) -> torch.Tensor:
        """Stand-in encoder used only when SAM2 isn't installed (unit tests)."""
        B, _, H, W = images.shape
        h, w = H // 16, W // 16
        # Simple average pool + channel expansion
        pooled = F.adaptive_avg_pool2d(images, (h, w))  # (B, 3, h, w)
        if not hasattr(self, "_fallback_proj"):
            self._fallback_proj = nn.Conv2d(3, 256, kernel_size=1).to(images.device)
        return self._fallback_proj(pooled)

    def encode_pre_rt(
        self,
        pre_images: torch.Tensor,
        pre_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode pre-RT volume into cross-session memory bank.

        Args:
            pre_images : (B, N_slices, 3, H, W)
            pre_masks  : (B, N_slices, 1, H, W) — combined GTVp+GTVn

        Returns:
            M_pre : (B, N_memory, C)
        """
        B, N, C_img, H, W = pre_images.shape

        # Subsample slices BEFORE encoding: CrossSessionMemoryEncoder only keeps
        # n_memory_frames slices anyway, so encoding the rest wastes GPU memory
        # proportional to N/n_memory_frames (~12x for a typical 100-slice volume).
        n_frames = self.memory_encoder.n_memory_frames
        if N > n_frames:
            indices = torch.linspace(0, N - 1, n_frames, dtype=torch.long, device=pre_images.device)
            pre_images = pre_images[:, indices]
            if pre_masks is not None:
                pre_masks = pre_masks[:, indices]
            N = n_frames

        # Batched feature extraction: flatten (B, N) -> (B*N), then reshape back.
        flat_imgs = pre_images.reshape(B * N, C_img, H, W)
        flat_feats = self._get_sam2_features(flat_imgs)       # (B*N, C, h, w)
        C, h, w = flat_feats.shape[1:]
        pre_feats = flat_feats.reshape(B, N, C, h, w)

        if pre_masks is not None:
            pre_masks_resized = F.interpolate(
                pre_masks.reshape(B * N, 1, H, W),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, N, 1, h, w)
        else:
            pre_masks_resized = None

        return self.memory_encoder(pre_feats, pre_masks_resized)

    def reset_mid_session_memory(self):
        """Call at the start of each new mid-RT volume."""
        self._M_mid = None

    # ------------------------------------------------------------------
    # Cross-patient retrieval
    # ------------------------------------------------------------------
    def set_patient_bank(self, bank: "CrossPatientBank | None") -> None:
        """Attach a pre-built retrieval bank (typically at inference time)."""
        self.patient_bank = bank

    def retrieve(self, query_pre_summary: torch.Tensor) -> torch.Tensor | None:
        """
        Retrieve cross-patient memory tokens for a batch query.

        Returns None if retrieval is disabled or no bank has been attached.
        Otherwise returns (B, K*N_tokens, C) gated tokens ready to concat
        onto M_pre.
        """
        if self.retrieval is None or self.patient_bank is None or len(self.patient_bank) == 0:
            return None
        tokens, _gate = self.retrieval(query_pre_summary, self.patient_bank)
        return tokens

    @torch.no_grad()
    def build_patient_bank(
        self,
        dataloader,
        device: torch.device | str = "cpu",
        n_tokens: int | None = None,
    ) -> "CrossPatientBank":
        """
        Build a CrossPatientBank by iterating over volume-level samples.

        Each batch is expected to yield the dict produced by HNTSMRGDataset
        (volume-level): keys `pre_images`, `mid_images`, `pre_masks`,
        `mid_masks`, `weeks_elapsed`, `patient_id`.

        Stores `pre_summary` (C,) and `change_template` (N_tokens, C) per
        patient. Returns the bank.
        """
        if n_tokens is None:
            n_tokens = self.retrieval_n_tokens

        bank = CrossPatientBank()
        self.eval()

        for batch in dataloader:
            pre_images = batch["pre_images"].to(device)       # (B, N, 3, H, W) or (N, 3, H, W)
            mid_images = batch["mid_images"].to(device)
            pre_masks = batch["pre_masks"].to(device)
            mid_masks = batch["mid_masks"].to(device)
            weeks_elapsed = batch.get("weeks_elapsed", 3)
            patient_ids = batch.get("patient_id")

            # Normalize shapes to (B, N, 3, H, W) / (B, N, 1, H, W).
            if pre_images.dim() == 4:
                pre_images = pre_images.unsqueeze(0)
                mid_images = mid_images.unsqueeze(0)
                pre_masks = pre_masks.unsqueeze(0)
                mid_masks = mid_masks.unsqueeze(0)
                if isinstance(patient_ids, str):
                    patient_ids = [patient_ids]
                if isinstance(weeks_elapsed, int):
                    weeks_elapsed = [weeks_elapsed]

            B, N, C_img, H, W = pre_images.shape

            flat_pre = pre_images.reshape(B * N, C_img, H, W)
            flat_mid = mid_images.reshape(B * N, C_img, H, W)
            pre_feats_flat = self._get_sam2_features(flat_pre)   # (B*N, C, h, w)
            mid_feats_flat = self._get_sam2_features(flat_mid)
            C, h, w = pre_feats_flat.shape[1:]
            pre_feats = pre_feats_flat.reshape(B, N, C, h, w)
            mid_feats = mid_feats_flat.reshape(B, N, C, h, w)

            # Down-sample masks to feature resolution.
            pre_mask_feat = F.interpolate(
                pre_masks.reshape(B * N, 1, H, W), size=(h, w), mode="bilinear", align_corners=False
            ).reshape(B, N, 1, h, w)
            mid_mask_feat = F.interpolate(
                mid_masks.reshape(B * N, 1, H, W), size=(h, w), mode="bilinear", align_corners=False
            ).reshape(B, N, 1, h, w)

            summary = compute_pre_summary(pre_feats, pre_mask_feat)           # (B, C)
            template = compute_change_template(
                pre_feats, mid_feats, pre_mask_feat, mid_mask_feat, n_tokens=n_tokens
            )  # (B, n_tokens, C)

            # Handle various weeks_elapsed shapes.
            if torch.is_tensor(weeks_elapsed):
                weeks_list = weeks_elapsed.tolist()
            elif isinstance(weeks_elapsed, (list, tuple)):
                weeks_list = list(weeks_elapsed)
            else:
                weeks_list = [int(weeks_elapsed)] * B

            for b in range(B):
                pid = patient_ids[b] if patient_ids is not None else f"patient_{len(bank)}"
                bank.add(
                    patient_id=str(pid),
                    pre_summary=summary[b],
                    change_template=template[b],
                    weeks_elapsed=int(weeks_list[b]),
                )

        self.patient_bank = bank
        return bank

    def forward(
        self,
        mid_images: torch.Tensor,
        M_pre: torch.Tensor,
        pre_images: torch.Tensor | None = None,
        weeks_elapsed: torch.Tensor | None = None,
        pre_gtvp_mask: torch.Tensor | None = None,
        pre_gtvn_mask: torch.Tensor | None = None,
        return_change_map: bool = True,
        detach_memory: bool = False,
        retrieved_memory: torch.Tensor | None = None,
        training_mode: bool = False,
        pre_class_masks: "torch.Tensor | None" = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for mid-RT segmentation.

        Args:
            mid_images     : (B, 3, H, W)
            M_pre          : (B, N_memory, C)
            pre_images     : (B, 3, H, W) — optional, for change head
            weeks_elapsed  : (B,) long — integer weeks between scans
            pre_gtvp_mask  : (B, 1, H, W) — pre-RT GTVp mask, used as prompt
            pre_gtvn_mask  : (B, 1, H, W) — pre-RT GTVn mask, used as prompt
            return_change_map: bool
            detach_memory  : if True, detach features when appending to M_mid
                             (use during evaluation to save memory; during
                             sequence training keep False so gate gets gradient)

        Returns:
            dict:
                "masks"       : (B, 2, H, W) — [GTVp, GTVn] logits
                "change_map"  : (B, 4, H, W)
                "gate_vals"   : (B, h, w)
                "attn_weights": (B, HW, N_pre)
        """
        B, _, H, W = mid_images.shape

        if weeks_elapsed is None:
            weeks_elapsed = torch.full((B,), 3, dtype=torch.long, device=mid_images.device)

        # 1. Extract mid-RT features
        mid_feats = self._get_sam2_features(mid_images)       # (B, C, h, w)
        _, C, h, w = mid_feats.shape

        # 2. Flatten spatial dims
        mid_feats_flat = mid_feats.permute(0, 2, 3, 1).reshape(B, h * w, C)

        # 2b. Augment M_pre with retrieved cross-patient memory tokens.
        M_pre_aug = M_pre
        if retrieved_memory is not None and retrieved_memory.numel() > 0:
            M_pre_aug = torch.cat([M_pre, retrieved_memory], dim=1)

        # 3. Cross-session memory attention
        enhanced_feats, attn_w, gate_vals = self.cross_session_attn(
            curr_features=mid_feats_flat,
            M_pre=M_pre_aug,
            M_mid=self._M_mid,
            weeks_elapsed=weeks_elapsed,
        )

        # 4. Update within-session memory.
        # KEY: do NOT detach during training so the gate learns.
        # The trailing-window keeps OOM in check.
        new_memory = enhanced_feats.detach() if detach_memory else enhanced_feats
        if self._M_mid is None:
            self._M_mid = new_memory
        else:
            self._M_mid = torch.cat([self._M_mid, new_memory], dim=1)
            max_mid_tokens = self.memory_bank_max_slices * h * w
            if self._M_mid.shape[1] > max_mid_tokens:
                self._M_mid = self._M_mid[:, -max_mid_tokens:].clone()

        # 5. Back to spatial
        enhanced_feats_spatial = enhanced_feats.reshape(B, h, w, C).permute(0, 3, 1, 2)

        # 6. SAM2 mask decoder — run ONCE per structure with its pre-RT mask prompt.
        mask_p = self._decode_with_mask_prompt(enhanced_feats_spatial, pre_gtvp_mask, (H, W))
        mask_n = self._decode_with_mask_prompt(enhanced_feats_spatial, pre_gtvn_mask, (H, W))
        masks = torch.cat([mask_p, mask_n], dim=1)            # (B, 2, H, W)

        result = {
            "masks": masks,
            "gate_vals": gate_vals.reshape(B, h, w),
            "attn_weights": attn_w.mean(dim=1),               # (B, HW, N_pre)
        }

        # 7. Change map
        if return_change_map and pre_images is not None:
            pre_feats = self._get_sam2_features(pre_images)
            change_logits = self.change_head(pre_feats, enhanced_feats_spatial)
            change_logits = F.interpolate(change_logits, size=(H, W), mode="bilinear", align_corners=False)
            result["change_map"] = change_logits

        return result

    # ------------------------------------------------------------------
    # Mask decoding
    # ------------------------------------------------------------------
    def _decode_with_mask_prompt(
        self,
        image_features: torch.Tensor,
        prior_mask: torch.Tensor | None,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        Decode one structure using a pre-RT mask as the SAM2 mask prompt.

        Falls back to a centroid-point prompt derived from the prior mask,
        and, if the prior mask is empty, to the peak-activation point.
        """
        B, C, h, w = image_features.shape
        H, W = output_size
        device = image_features.device

        # Prepare low-res mask prompt if available (SAM2 expects 1/4 res, ie 256 for 1024).
        low_res_mask_prompt = None
        point_coords = None
        point_labels = None

        if prior_mask is not None and prior_mask.sum() > 0:
            # Down-sample mask to SAM2 low-res prompt resolution.
            low_res_mask_prompt = F.interpolate(
                prior_mask.float(), size=(256, 256), mode="bilinear", align_corners=False
            )
            # Also compute a centroid point per batch element as a backup prompt.
            point_coords, point_labels = self._mask_to_centroid_points(prior_mask, (H, W))
        else:
            # Empty prior mask: fall back to peak-activation point.
            point_coords, point_labels = self._activation_peak_points(image_features, (H, W))

        low_res_masks = None
        try:
            sparse_embed, dense_embed = self.sam2.sam_prompt_encoder(
                points=(point_coords, point_labels) if point_coords is not None else None,
                boxes=None,
                masks=low_res_mask_prompt,
            )
            low_res_masks, _ = self.sam2.sam_mask_decoder(
                image_embeddings=image_features,
                image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                multimask_output=False,
            )
        except (AttributeError, TypeError):
            low_res_masks = self._fallback_decoder(image_features, prior_mask)

        masks = F.interpolate(low_res_masks, size=(H, W), mode="bilinear", align_corners=False)
        return masks

    @staticmethod
    def _mask_to_centroid_points(
        mask: torch.Tensor, output_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-batch centroid of positive mask pixels, in image coords."""
        B, _, H_m, W_m = mask.shape
        H, W = output_size
        device = mask.device

        m = (mask > 0.5).float().squeeze(1)                   # (B, H_m, W_m)
        ys = torch.arange(H_m, device=device).view(1, H_m, 1).expand_as(m)
        xs = torch.arange(W_m, device=device).view(1, 1, W_m).expand_as(m)

        total = m.sum(dim=(1, 2)).clamp_min(1.0)
        cy = (m * ys).sum(dim=(1, 2)) / total
        cx = (m * xs).sum(dim=(1, 2)) / total
        # Rescale to output_size
        cy = cy * (H / H_m)
        cx = cx * (W / W_m)
        point_coords = torch.stack([cx, cy], dim=1).unsqueeze(1)   # (B, 1, 2)
        point_labels = torch.ones(B, 1, dtype=torch.int, device=device)
        return point_coords, point_labels

    @staticmethod
    def _activation_peak_points(
        features: torch.Tensor, output_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Argmax of mean-activation as a last-resort prompt."""
        B, C, h, w = features.shape
        H, W = output_size
        activation = features.mean(dim=1)                      # (B, h, w)
        flat_idx = activation.reshape(B, -1).argmax(dim=1)
        point_y = (flat_idx // w).float() / max(h - 1, 1) * H
        point_x = (flat_idx % w).float() / max(w - 1, 1) * W
        point_coords = torch.stack([point_x, point_y], dim=1).unsqueeze(1)
        point_labels = torch.ones(B, 1, dtype=torch.int, device=features.device)
        return point_coords, point_labels

    def _fallback_decoder(
        self,
        features: torch.Tensor,
        prior_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Used only when SAM2's decoder is unavailable (unit tests).
        Produces a plausible logits map biased toward the prior mask location.
        """
        B, C, h, w = features.shape
        base = features.mean(dim=1, keepdim=True)              # (B, 1, h, w)
        if prior_mask is not None:
            prior_low = F.interpolate(prior_mask.float(), size=(h, w), mode="bilinear", align_corners=False)
            base = base + prior_low
        return base

    @classmethod
    def from_pretrained(
        cls,
        sam2_checkpoint: str,
        sam2_cfg: str = "sam2_hiera_large",
        **kwargs,
    ) -> "CSMSAM":
        try:
            from sam2.build_sam import build_sam2
            sam2 = build_sam2(sam2_cfg, sam2_checkpoint, device="cpu")
        except ImportError:
            raise ImportError(
                "SAM2 is not installed. Run:\n"
                "  git clone https://github.com/facebookresearch/sam2.git\n"
                "  pip install -e sam2/"
            )
        in_chans = kwargs.pop("in_chans", 3)
        if in_chans != 3:
            CSMSAM._expand_patch_embed(sam2, in_chans)
        return cls(sam2_model=sam2, **kwargs)

    @staticmethod
    def _expand_patch_embed(sam2_model, in_chans: int) -> None:
        """Expand SAM2 patch-embed proj from 3 to in_chans; new channels init as mean of existing 3."""
        trunk = getattr(getattr(sam2_model, "image_encoder", None), "trunk", None)
        patch_embed = getattr(trunk, "patch_embed", None)
        if patch_embed is None or not hasattr(patch_embed, "proj"):
            return
        old_proj = patch_embed.proj
        if old_proj.in_channels == in_chans:
            return
        import torch.nn as nn
        new_proj = nn.Conv2d(
            in_chans, old_proj.out_channels,
            kernel_size=old_proj.kernel_size, stride=old_proj.stride,
            padding=old_proj.padding, bias=old_proj.bias is not None,
        )
        with torch.no_grad():
            new_proj.weight[:, :3] = old_proj.weight
            for i in range(3, in_chans):
                new_proj.weight[:, i] = old_proj.weight.mean(dim=1)
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        patch_embed.proj = new_proj

    def get_trainable_params(self) -> list[dict]:
        novel_params = (
            list(self.memory_encoder.parameters())
            + list(self.cross_session_attn.parameters())
            + list(self.change_head.parameters())
        )
        if self.retrieval is not None:
            novel_params += list(self.retrieval.parameters())

        decoder_params = []
        if hasattr(self.sam2, "sam_mask_decoder"):
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
        if self.retrieval is not None:
            groups["retrieval"] = self.retrieval
        counts = {name: sum(p.numel() for p in m.parameters() if p.requires_grad) for name, m in groups.items()}
        counts["total"] = sum(counts.values())
        return counts
