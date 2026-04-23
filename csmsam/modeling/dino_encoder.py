"""
DINOv2 + Bridge image encoder — drop-in replacement for SAM2's Hiera encoder.

Claim (the "galash" idea): a frozen DINOv2 backbone produces better medical-image
features than SAM2's prompt-tuned ViT-H / Hiera, while SAM2's prompt encoder and
mask decoder remain strong. So we keep SAM2 as the decoder and swap the encoder.

The forward contract matches csm_sam.CSMSAM._get_sam2_features — i.e.::

    images [B, 3, H, W]  →  (image_emb [B, 256, 64, 64],
                             high_res_features [[B, 32, 256, 256],
                                                [B, 64, 128, 128]])

The trainable Bridge fuses 4 equally-spaced DINO layers and produces
decoder-compatible tensors. Bridge is adapted from the galash
(~/Documents/galash) repo — trimmed for CSM-SAM:

  * No `change_proj` / `dense_prompt` pathway: CSM-SAM processes a single image,
    not a symmetric pair, so there are no change tokens. Dense-prompt handling
    stays with SAM2's mask-prompt encoder downstream.
  * No CrossChangeAttention: same reason.

LR groups: the DINO backbone is FROZEN. The Bridge is TRAINABLE — expose its
parameters via `trainable_parameters()` so csm_sam can add them to the "novel"
lr group (lr_scale=1.0).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Small registry — add more variants as you need them. Hidden sizes / layer
# counts come from the HF configs; patch 14 is DINOv2 standard (DINOv3 uses 16).
DINO_VARIANTS = {
    "dinov2_small": {"hf": "facebook/dinov2-small", "dim": 384,  "layers": 12, "patch": 14},
    "dinov2_base":  {"hf": "facebook/dinov2-base",  "dim": 768,  "layers": 12, "patch": 14},
    "dinov2_large": {"hf": "facebook/dinov2-large", "dim": 1024, "layers": 24, "patch": 14},
}


# ---------------------------------------------------------------------------
# Bridge building blocks (ported from galash/model.py, unchanged)
# ---------------------------------------------------------------------------
class _ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, dim),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class _TransformerRefinement(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).view(B, C, H, W)


class _MaskEncoder(nn.Module):
    """Tiny conv stack: [B, 1, H, W] → [B, sam_dim, target_grid, target_grid]."""

    def __init__(self, out_dim: int = 256, target_grid: int = 64):
        super().__init__()
        self.target_grid = target_grid
        # Work at target_grid resolution end-to-end — the input mask is first
        # bilinearly downsampled to that grid so receptive field is bounded
        # regardless of input size.
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 128, 3, padding=1, bias=False),
            nn.GroupNorm(32, 128), nn.GELU(),
            nn.Conv2d(128, out_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_dim),
        )

    def forward(self, mask: Tensor) -> Tensor:
        if mask.shape[-1] != self.target_grid or mask.shape[-2] != self.target_grid:
            mask = F.interpolate(
                mask.float(), size=(self.target_grid, self.target_grid),
                mode="bilinear", align_corners=False,
            )
        return self.stem(mask)                                  # [B, 256, g, g]


# ---------------------------------------------------------------------------
# Bridge (trimmed: no dense_prompt / change_proj path)
# ---------------------------------------------------------------------------
class _Bridge(nn.Module):
    """Multi-scale DINO → SAM2-compatible feature adapter.

    Input:
        multi_feats: list of [B, S, dino_dim] tensors, one per selected layer
                     (CLS already stripped). S = h * w patches.
        h, w: DINO patch-grid spatial dims.

    Output of `forward`:
        image_emb        : [B, sam_dim, target_h, target_w]
        high_res_features: [feat_s0 [B, 32, 4·target_h, 4·target_w],
                            feat_s1 [B, 64, 2·target_h, 2·target_w]]

    Output of `forward_mask(image_emb, mask)` — galash-style dense_prompt path:
        dense_prompt     : [B, sam_dim, target_h, target_w]
        A tiny MaskEncoder embeds the pre-RT structure mask to the decoder
        grid; mask tokens are then the query in a cross-attention block whose
        key/value is the (memory-aware) image_emb. The resulting tokens become
        dense_prompt for the SAM2 mask decoder, replacing SAM2's frozen
        prompt_encoder(masks=...) path. Running this per structure (GTVp,
        GTVn) shares the single DINO+Bridge image forward.
    """

    def __init__(self, dino_dim: int, sam_dim: int = 256, num_scales: int = 4,
                 target_grid: int = 64):
        super().__init__()
        self.sam_dim = sam_dim
        self.target_grid = target_grid

        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dino_dim, sam_dim),
                nn.LayerNorm(sam_dim),
                nn.GELU(),
                nn.Linear(sam_dim, sam_dim),
            )
            for _ in range(num_scales)
        ])
        self.scale_convs = nn.ModuleList([
            nn.Sequential(_ResidualBlock(sam_dim), _ResidualBlock(sam_dim))
            for _ in range(num_scales)
        ])
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(sam_dim, sam_dim, 1), nn.GroupNorm(32, sam_dim))
            for _ in range(num_scales - 1)
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(sam_dim * num_scales, sam_dim, 1),
            _ResidualBlock(sam_dim),
        )
        self.refine = _TransformerRefinement(dim=sam_dim, num_heads=4, num_layers=2)

        # SAM2-only: high-res FPN-style outputs at strides 4 and 8 from decoder
        # target. For target 64×64 this is 256×256 and 128×128.
        self.hr_proj_s1 = nn.Sequential(nn.Linear(dino_dim, 64), nn.GELU())
        self.hr_conv_s1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.GroupNorm(16, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.GroupNorm(16, 64), nn.GELU(),
        )
        self.hr_proj_s0 = nn.Sequential(nn.Linear(dino_dim, 32), nn.GELU())
        self.hr_conv_s0 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.GroupNorm(8, 32), nn.GELU(),
        )

        # Mask → dense_prompt path (per-structure, replaces SAM2's mask prompt).
        self.mask_encoder = _MaskEncoder(out_dim=sam_dim, target_grid=target_grid)
        self.mask_pos_emb = nn.Parameter(torch.zeros(1, sam_dim, target_grid, target_grid))
        nn.init.trunc_normal_(self.mask_pos_emb, std=0.02)
        self.mask_cross_attn = nn.TransformerDecoderLayer(
            d_model=sam_dim, nhead=4, dim_feedforward=sam_dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.dense_out_norm = nn.LayerNorm(sam_dim)

    def forward(
        self,
        multi_feats: List[Tensor],
        h: int,
        w: int,
        target_h: int = 64,
        target_w: int = 64,
    ) -> Tuple[Tensor, List[Tensor]]:
        B = multi_feats[0].size(0)

        scale_maps: List[Tensor] = []
        for feat, proj, conv in zip(multi_feats, self.scale_projs, self.scale_convs):
            x = proj(feat)                                              # [B, S, sam_dim]
            x = x.view(B, h, w, self.sam_dim).permute(0, 3, 1, 2)       # [B, C, h, w]
            x = F.interpolate(x, (target_h, target_w), mode="bilinear", align_corners=False)
            x = conv(x)
            scale_maps.append(x)

        # Top-down FPN fusion (coarse → fine)
        for i in range(len(scale_maps) - 1, 0, -1):
            coarse = scale_maps[i]
            lateral = self.lateral_convs[i - 1](scale_maps[i - 1])
            scale_maps[i - 1] = lateral + coarse

        image_emb = self.fusion(torch.cat(scale_maps, dim=1))
        image_emb = self.refine(image_emb)

        # High-res from the finest (earliest-layer) DINO features.
        feat_s1 = self.hr_proj_s1(multi_feats[0])
        feat_s1 = feat_s1.view(B, h, w, 64).permute(0, 3, 1, 2)
        feat_s1 = F.interpolate(feat_s1, (target_h * 2, target_w * 2), mode="bilinear", align_corners=False)
        feat_s1 = self.hr_conv_s1(feat_s1)

        feat_s0 = self.hr_proj_s0(multi_feats[0])
        feat_s0 = feat_s0.view(B, h, w, 32).permute(0, 3, 1, 2)
        feat_s0 = F.interpolate(feat_s0, (target_h * 4, target_w * 4), mode="bilinear", align_corners=False)
        feat_s0 = self.hr_conv_s0(feat_s0)

        return image_emb, [feat_s0, feat_s1]

    def forward_mask(self, image_emb: Tensor, mask: Tensor) -> Tensor:
        """Build a dense_prompt by cross-attending a mask embedding to image_emb.

        Args:
            image_emb: [B, sam_dim, g, g] — Bridge output (optionally memory-
                       enriched upstream; we don't care here, just use as K/V).
            mask:      [B, 1, H, W]        — pre-RT structure mask (0/1 or
                       soft). An all-zero mask becomes a learned "no prior"
                       prompt via the MaskEncoder biases.
        Returns:
            dense_prompt: [B, sam_dim, g, g]
        """
        B, C, g, gw = image_emb.shape
        if g != self.target_grid or gw != self.target_grid:
            raise ValueError(
                f"image_emb grid {g}×{gw} != Bridge target_grid {self.target_grid}"
            )
        mask_emb = self.mask_encoder(mask)                      # [B, C, g, g]
        mask_emb = mask_emb + self.mask_pos_emb                 # + learned 2D pos

        mask_tokens = mask_emb.flatten(2).transpose(1, 2)       # [B, g·g, C]
        image_tokens = image_emb.flatten(2).transpose(1, 2)     # [B, g·g, C]

        # TransformerDecoderLayer: self-attn(mask) → cross-attn(mask → image) → FFN.
        attended = self.mask_cross_attn(tgt=mask_tokens, memory=image_tokens)
        attended = self.dense_out_norm(attended)

        return attended.transpose(1, 2).view(B, C, g, g).contiguous()


# ---------------------------------------------------------------------------
# Full encoder: DINOv2 (frozen) + Bridge (trainable)
# ---------------------------------------------------------------------------
class DinoEncoder(nn.Module):
    """Drop-in image encoder: (images) → (image_emb, high_res_features).

    Matches the contract of `CSMSAM._get_sam2_features`, so csm_sam only needs
    to dispatch on encoder_type and the rest of the pipeline is unchanged.
    """

    def __init__(
        self,
        variant: str = "dinov2_base",
        dino_img_size: int = 518,
        target_grid: int = 64,
        feature_layers: Optional[List[int]] = None,
        sam_dim: int = 256,
        hf_id: Optional[str] = None,
        lora_rank: int = 0,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_targets: Optional[List[str]] = None,
    ):
        super().__init__()

        info = DINO_VARIANTS.get(variant)
        if info is None and hf_id is None:
            raise ValueError(
                f"Unknown DINO variant '{variant}'. Known: {list(DINO_VARIANTS)}. "
                "Pass hf_id=... to load an arbitrary HF model."
            )
        self.variant = variant
        self.hf_id = hf_id or info["hf"]
        self.dino_img_size = int(dino_img_size)
        self.target_grid = int(target_grid)

        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "transformers is required for DinoEncoder. Run: pip install 'transformers>=4.40'"
            ) from e

        self.dino = AutoModel.from_pretrained(self.hf_id, output_hidden_states=True)
        cfg = self.dino.config
        self.dino_dim = int(cfg.hidden_size)
        self.patch_size = int(cfg.patch_size)
        n_layers = int(cfg.num_hidden_layers)

        if feature_layers is None:
            # Equally-spaced: quarter, half, three-quarter, last (galash default).
            feature_layers = [n_layers // 4 - 1, n_layers // 2 - 1, 3 * n_layers // 4 - 1, n_layers - 1]
        # HF hidden_states is length n_layers+1 (embeddings at index 0); we index
        # with (i+1) in forward so these indices are in [0, n_layers-1].
        if any(i < 0 or i >= n_layers for i in feature_layers):
            raise ValueError(
                f"feature_layers {feature_layers} out of range for encoder with {n_layers} blocks"
            )
        self.feature_layers = list(feature_layers)

        # Number of register tokens (DINOv2-with-registers has 4; vanilla DINOv2 has 0).
        self.num_registers = int(getattr(cfg, "num_register_tokens", 0))

        # Freeze the backbone (LoRA will selectively unfreeze below if requested).
        for p in self.dino.parameters():
            p.requires_grad = False

        # Optional LoRA adapters on DINO attention projections. r=0 → fully
        # frozen (original behaviour). Otherwise we inject low-rank adapters
        # on the requested modules ("query","value" by default — adapts
        # attention without touching FFN → ~0.5M trainable at r=8 for
        # DINOv2-base) so DINO features can adapt to tumor MRI without
        # overfitting ~125 patients.
        self.lora_rank = int(lora_rank) if lora_rank is not None else 0
        if self.lora_rank > 0:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as e:
                raise ImportError(
                    "peft is required when lora_rank > 0. Run: pip install 'peft>=0.11'"
                ) from e
            targets = lora_targets or ["query", "value"]
            lora_cfg = LoraConfig(
                r=self.lora_rank,
                lora_alpha=lora_alpha,
                target_modules=targets,
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.dino = get_peft_model(self.dino, lora_cfg)
            # peft will have set requires_grad=True on lora_A / lora_B weights.
            # Keep dino in train mode so dropout etc. in adapters fire.
        else:
            self.dino.eval()

        self.bridge = _Bridge(
            dino_dim=self.dino_dim,
            sam_dim=sam_dim,
            num_scales=len(self.feature_layers),
            target_grid=self.target_grid,
        )

    # When DINO is fully frozen we keep it in eval() (no dropout, no BN update).
    # When LoRA adapters are active we must respect the top-level train/eval
    # mode so LoRA dropout fires during training but not during validation.
    def train(self, mode: bool = True):
        super().train(mode)
        if self.lora_rank == 0:
            self.dino.eval()
        return self

    def trainable_parameters(self):
        """Bridge params + LoRA adapter params (if lora_rank > 0)."""
        yield from self.bridge.parameters()
        if self.lora_rank > 0:
            for p in self.dino.parameters():
                if p.requires_grad:
                    yield p

    def count_trainable_params(self) -> int:
        n = sum(p.numel() for p in self.bridge.parameters() if p.requires_grad)
        if self.lora_rank > 0:
            n += sum(p.numel() for p in self.dino.parameters() if p.requires_grad)
        return n

    def _forward_dino(self, images: Tensor) -> Tuple[List[Tensor], int, int]:
        """Run DINO, return selected hidden states (CLS/registers stripped) and patch grid.

        Wrapped in torch.no_grad() when DINO is fully frozen; under LoRA we
        need gradients to flow through the adapters so we just run it normally.
        """
        if self.lora_rank > 0:
            return self._forward_dino_impl(images)
        with torch.no_grad():
            return self._forward_dino_impl(images)

    def _forward_dino_impl(self, images: Tensor) -> Tuple[List[Tensor], int, int]:
        # HF DINOv2 supports arbitrary input sizes via interpolate_pos_encoding, but
        # patch-14 + odd input can leave 1 extra row/col that gets sliced. Resize
        # to a DINO-friendly square so the patch grid is deterministic.
        if images.shape[-1] != self.dino_img_size or images.shape[-2] != self.dino_img_size:
            images = F.interpolate(
                images, size=(self.dino_img_size, self.dino_img_size),
                mode="bilinear", align_corners=False,
            )
        h = w = self.dino_img_size // self.patch_size
        out = self.dino(pixel_values=images, output_hidden_states=True, interpolate_pos_encoding=True)
        hs = out.hidden_states                                      # len n_layers+1

        n_skip = 1 + self.num_registers                             # CLS + registers
        multi: List[Tensor] = []
        for i in self.feature_layers:
            multi.append(hs[i + 1][:, n_skip:, :])                  # [B, S, D]
        return multi, h, w

    def forward(self, images: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Args:
            images: [B, 3, H, W] (already ImageNet-normalized by the dataset).
        Returns:
            image_emb          : [B, 256, target_grid, target_grid]
            high_res_features  : [[B, 32, 4·target_grid, 4·target_grid],
                                  [B, 64, 2·target_grid, 2·target_grid]]
        """
        multi, h, w = self._forward_dino(images)
        image_emb, high_res = self.bridge(
            multi, h, w, target_h=self.target_grid, target_w=self.target_grid
        )
        return image_emb, high_res

    def encode_mask_prompt(self, image_emb: Tensor, mask: Tensor) -> Tensor:
        """Run Bridge's mask cross-attention to produce the decoder dense_prompt.

        Galash-style path — replaces SAM2's frozen prompt_encoder(masks=...)
        entirely. Safe to call with an all-zero `mask` (prompt-dropout / BG
        slices); MaskEncoder biases act as a learned "no prior" signature.

        Shapes:
            image_emb [B, sam_dim, target_grid, target_grid]
            mask      [B, 1, H, W]
            returns   [B, sam_dim, target_grid, target_grid]
        """
        return self.bridge.forward_mask(image_emb, mask)
