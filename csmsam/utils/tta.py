"""Test-time augmentation (TTA) helpers for CSM-SAM.

Currently exposes a simple horizontal-flip TTA wrapper. The helper is
model-agnostic: it works with any forward callable that accepts a batch
dict and returns a dict containing a ``"masks"`` logits tensor.
"""
from __future__ import annotations

from typing import Callable

import torch


def _flip(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.flip(x, dims=(dim,))


def hflip_tta(
    model,
    forward_fn: Callable[[dict], dict],
    batch: dict,
    mid_image_key: str = "mid_images",
    pre_image_key: str = "pre_images",
    mask_keys: tuple = ("pre_gtvp_mask", "pre_gtvn_mask"),
    flip_dim: int = -1,
) -> torch.Tensor:
    """Average logits from the original batch and its width-flipped copy.

    The flipped forward pass is flipped back before averaging, so the
    returned logits live in the original image frame.

    Args:
        model        : unused, kept for API symmetry (forward_fn closes over it).
        forward_fn   : callable(batch_dict) -> dict with key ``"masks"`` (B, 2, H, W).
        batch        : batch dict with image + (optional) prior-mask tensors.
        mid_image_key: key of the mid-RT images to flip.
        pre_image_key: key of the pre-RT images to flip (if present).
        mask_keys    : keys of prior masks (pre-RT GTVp/GTVn) to flip if present.
        flip_dim     : spatial dim to flip on. Default -1 (width).

    Returns:
        Averaged (B, 2, H, W) logits.
    """
    del model  # forward_fn carries the model reference.

    out_a = forward_fn(batch)
    logits_a = out_a["masks"]

    flipped = dict(batch)
    if mid_image_key in flipped and torch.is_tensor(flipped[mid_image_key]):
        flipped[mid_image_key] = _flip(flipped[mid_image_key], flip_dim)
    if pre_image_key in flipped and torch.is_tensor(flipped[pre_image_key]):
        flipped[pre_image_key] = _flip(flipped[pre_image_key], flip_dim)
    for k in mask_keys:
        if k in flipped and torch.is_tensor(flipped[k]):
            flipped[k] = _flip(flipped[k], flip_dim)

    out_b = forward_fn(flipped)
    logits_b = _flip(out_b["masks"], flip_dim)

    return (logits_a + logits_b) / 2.0
