"""
Shared training + evaluation utility for 2D segmentation baselines.

Any baseline that produces logits from mid-RT images (or pre+mid pairs) can
call ``train_and_evaluate`` to get a full training loop, validation with
aggDSC, checkpoint selection, and test-time evaluation — all using the
existing HNTS-MRG dataset classes and metric utilities.

Usage from a baseline file::

    from baselines.baseline_trainer import train_and_evaluate

    model = build_my_model(...)
    results = train_and_evaluate(
        model=model,
        data_dir="data/processed",
        output_dir="results/experiments/trained_baselines/my_model",
        epochs=50,
        model_name="my_model",
    )
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.utils.data import Dataset

from csmsam.datasets.hnts_mrg import (
    HNTSMRGSliceDataset, HNTSMRGDataset, SAM2_MEAN, SAM2_STD, to_rgb_tensor, to_mask_tensor,
)
from csmsam.utils.metrics import evaluate_patient, aggregate_metrics


# ---------------------------------------------------------------------------
# Fast cached dataset — reads from .npz instead of NIfTI
# ---------------------------------------------------------------------------

class _CachedSliceDataset(Dataset):
    """Reads pre-extracted memmap cache for near-instant epoch starts. No RAM load."""

    def __init__(self, cache_dir: str, image_size: int = 256, augment: bool = False):
        cache_dir = Path(cache_dir)
        with open(cache_dir / "meta.json") as f:
            meta = json.load(f)
        N, H, W = meta["n_slices"], meta["H"], meta["W"]
        self.pre = np.memmap(cache_dir / "pre_images.npy", dtype=np.float16, mode="r", shape=(N, H, W))
        self.mid = np.memmap(cache_dir / "mid_images.npy", dtype=np.float16, mode="r", shape=(N, H, W))
        self.pre_m = np.memmap(cache_dir / "pre_masks.npy", dtype=np.uint8, mode="r", shape=(N, H, W))
        self.mid_m = np.memmap(cache_dir / "mid_masks.npy", dtype=np.uint8, mode="r", shape=(N, H, W))
        self.pids = meta["patient_ids"]
        self.image_size = image_size
        self.augment = augment
        print(f"  [CachedSliceDataset] memmap: {N} slices @ {H}x{W} from {cache_dir}")

    def __len__(self):
        return len(self.pids)

    def _fast_rgb(self, arr: np.ndarray) -> torch.Tensor:
        """Minimal CPU path: skip F.interpolate when already at target size."""
        H, W = arr.shape
        size = self.image_size
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        if H != size or W != size:
            t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        # (1,1,H,W) -> (3,H,W) — expand is zero-copy
        t = t.squeeze(0).expand(3, -1, -1).contiguous()
        # Pre-shaped constants for broadcast
        if not hasattr(self, "_mean"):
            self._mean = SAM2_MEAN.view(3, 1, 1)
            self._std = SAM2_STD.view(3, 1, 1)
        return (t - self._mean) / self._std

    def _fast_mask(self, arr: np.ndarray) -> torch.Tensor:
        H, W = arr.shape
        size = self.image_size
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        if H != size or W != size:
            t = F.interpolate(t, size=(size, size), mode="nearest")
        return t.squeeze(0)  # (1,H,W)

    def __getitem__(self, idx):
        # Cast to float32 only for the ONE slice we return (not the whole array)
        pre_sl = np.asarray(self.pre[idx], dtype=np.float32)
        mid_sl = np.asarray(self.mid[idx], dtype=np.float32)
        pre_m = np.asarray(self.pre_m[idx], dtype=np.float32)
        mid_m = np.asarray(self.mid_m[idx], dtype=np.float32)

        if self.augment:
            import random
            if random.random() < 0.5:
                pre_sl = np.ascontiguousarray(np.fliplr(pre_sl))
                mid_sl = np.ascontiguousarray(np.fliplr(mid_sl))
                pre_m = np.ascontiguousarray(np.fliplr(pre_m))
                mid_m = np.ascontiguousarray(np.fliplr(mid_m))
            if random.random() < 0.3:
                pre_sl = np.ascontiguousarray(np.flipud(pre_sl))
                mid_sl = np.ascontiguousarray(np.flipud(mid_sl))
                pre_m = np.ascontiguousarray(np.flipud(pre_m))
                mid_m = np.ascontiguousarray(np.flipud(mid_m))

        return {
            "pre_image": self._fast_rgb(pre_sl),
            "mid_image": self._fast_rgb(mid_sl),
            "pre_mask": self._fast_mask(pre_m),
            "mid_mask": self._fast_mask(mid_m),
            "patient_id": str(self.pids[idx]),
        }


# ---------------------------------------------------------------------------
# Loss: Dice + BCE
# ---------------------------------------------------------------------------

def _dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss on sigmoid-activated predictions.

    Args:
        pred:   (B, 1, H, W) raw logits
        target: (B, 1, H, W) binary targets
    """
    pred_sig = torch.sigmoid(pred)
    pred_flat = pred_sig.reshape(pred_sig.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def _combined_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice + weighted BCE. Up-weights tumor pixels (~1% of voxels) so the
    model doesn't collapse to predicting background everywhere."""
    # Positive class is ~1% of pixels. pos_weight=50 roughly balances the signal.
    pos_weight = torch.tensor([50.0], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    dice = _dice_loss(logits, target)
    # Weight Dice heavier to force actual overlap, not just right-classifying background
    return bce + 2.0 * dice


# ---------------------------------------------------------------------------
# Volume-level evaluation (per-patient)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_cached(
    model: nn.Module,
    cache_dir: Path,
    image_size: int,
    device: str,
    uses_pre: bool,
    input_adapter,
    threshold: float,
) -> tuple[dict, list[dict]]:
    """Fast eval from memmap cache, grouped by patient_id."""
    with open(cache_dir / "meta.json") as f:
        meta = json.load(f)
    N, H, W = meta["n_slices"], meta["H"], meta["W"]
    pre = np.memmap(cache_dir / "pre_images.npy", dtype=np.float16, mode="r", shape=(N, H, W))
    mid = np.memmap(cache_dir / "mid_images.npy", dtype=np.float16, mode="r", shape=(N, H, W))
    pre_m = np.memmap(cache_dir / "pre_masks.npy", dtype=np.uint8, mode="r", shape=(N, H, W))
    mid_m = np.memmap(cache_dir / "mid_masks.npy", dtype=np.uint8, mode="r", shape=(N, H, W))
    pids = meta["patient_ids"]

    # Group slice indices by patient
    groups: dict[str, list[int]] = {}
    for i, pid in enumerate(pids):
        groups.setdefault(pid, []).append(i)

    mean = SAM2_MEAN.view(3, 1, 1).to(device)
    std = SAM2_STD.view(3, 1, 1).to(device)

    per_patient: list[dict] = []
    model.eval()

    for pid, idxs in groups.items():
        pred_slices = []
        gt_slices = []
        for i in idxs:
            mid_np = np.asarray(mid[i], dtype=np.float32)
            t_mid = torch.from_numpy(mid_np).unsqueeze(0).unsqueeze(0).to(device)
            if t_mid.shape[-1] != image_size:
                t_mid = F.interpolate(t_mid, (image_size, image_size), mode="bilinear", align_corners=False)
            mid_rgb = (t_mid.expand(1, 3, image_size, image_size) - mean) / std

            if uses_pre:
                pre_np = np.asarray(pre[i], dtype=np.float32)
                t_pre = torch.from_numpy(pre_np).unsqueeze(0).unsqueeze(0).to(device)
                if t_pre.shape[-1] != image_size:
                    t_pre = F.interpolate(t_pre, (image_size, image_size), mode="bilinear", align_corners=False)
                pre_rgb = (t_pre.expand(1, 3, image_size, image_size) - mean) / std
                inp = input_adapter(pre_rgb, mid_rgb, None, None) if input_adapter else torch.cat([pre_rgb, mid_rgb], dim=1)
            else:
                inp = mid_rgb

            logits = model(inp)
            if logits.shape[1] > 1:
                logits = logits[:, 1:2]
            mask = (torch.sigmoid(logits) > threshold).squeeze().cpu().numpy()
            pred_slices.append(mask.astype(np.float32))
            gt_slices.append(np.asarray(mid_m[i], dtype=np.float32))

        pred_binary = np.stack(pred_slices)
        gt = np.stack(gt_slices)
        if gt.shape[-1] != pred_binary.shape[-1]:
            # Resize GT to match prediction size
            gt_t = torch.from_numpy(gt).unsqueeze(1)
            gt_t = F.interpolate(gt_t, size=pred_binary.shape[-2:], mode="nearest")
            gt = gt_t.squeeze(1).numpy()
        metrics = evaluate_patient(
            pred_masks=pred_binary, pred_gtvp=pred_binary, pred_gtvn=pred_binary,
            target_gtvp=gt, target_gtvn=gt,
        )
        metrics["patient_id"] = pid
        per_patient.append(metrics)

    return aggregate_metrics(per_patient), per_patient


@torch.no_grad()
def _evaluate_split(
    model: nn.Module,
    data_dir: str,
    split: str,
    image_size: int,
    device: str,
    uses_pre: bool = False,
    input_adapter=None,
    threshold: float = 0.5,
) -> tuple[dict, list[dict]]:
    """Run volume-level evaluation on *split*, return (aggregate, per_patient).

    Fast path: if a memmap cache exists under data_dir/cache/<split>/, group its
    slices by patient and evaluate without touching NIfTI.
    """
    cache_dir = Path(data_dir) / "cache" / split
    if (cache_dir / "meta.json").exists():
        return _evaluate_cached(
            model, cache_dir, image_size, device, uses_pre, input_adapter, threshold
        )

    dataset = HNTSMRGDataset(data_dir=data_dir, split=split, image_size=image_size)
    per_patient: list[dict] = []

    model.eval()
    for idx in range(len(dataset)):
        patient_data = dataset[idx]
        pid = patient_data["patient_id"]

        mid_images = patient_data["mid_images"]  # (N, 3, H, W)
        N, _, H, W = mid_images.shape

        pred_slices = []
        for i in range(N):
            img = mid_images[i : i + 1].to(device)  # (1, 3, H, W)

            if uses_pre:
                pre_images = patient_data["pre_images"]
                N_pre = pre_images.shape[0]
                j = min(i, N_pre - 1)
                pre_img = pre_images[j : j + 1].to(device)  # (1, 3, H, W)
                if input_adapter is not None:
                    inp = input_adapter(pre_img, img, patient_data, j)
                else:
                    inp = torch.cat([pre_img, img], dim=1)  # (1, 6, H, W)
            else:
                inp = img

            logits = model(inp)  # (1, C, H, W)
            if logits.shape[1] == 1:
                mask = (torch.sigmoid(logits) > threshold).squeeze().cpu().numpy()
            else:
                probs = F.softmax(logits, dim=1)
                mask = (probs[:, 1:].sum(dim=1) > threshold).squeeze().cpu().numpy()
            pred_slices.append(mask.astype(np.float32))

        pred_binary = np.stack(pred_slices)
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
        per_patient.append(metrics)

    agg = aggregate_metrics(per_patient)
    return agg, per_patient


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_and_evaluate(
    model: nn.Module,
    data_dir: str,
    output_dir: str,
    split: str = "test",
    image_size: int = 256,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cuda",
    num_workers: int = 8,
    model_name: str = "baseline",
    uses_pre: bool = False,
    input_adapter=None,
    trainable_params: Optional[list] = None,
) -> dict:
    """Train a 2D segmentation model on HNTS-MRG slices and evaluate.

    Args:
        model: ``nn.Module`` whose ``forward(x)`` returns logits
            ``(B, 1, H, W)`` or ``(B, C, H, W)``.
        data_dir: root processed data directory (must contain train/val/test).
        output_dir: where to write metrics.json, best.pth, training_log.json.
        split: evaluation split (usually ``"test"``).
        image_size: spatial resolution for dataset loading.
        epochs: number of training epochs.
        batch_size: training batch size.
        lr: AdamW learning rate.
        device: ``"cuda"`` or ``"cpu"``.
        num_workers: DataLoader workers.
        model_name: for logging.
        uses_pre: when ``True``, training stacks ``pre_image`` and
            ``mid_image`` as ``(B, 6, H, W)`` and passes to the model.
            For test-time eval, each slice is similarly paired.
        input_adapter: optional callable
            ``(pre_img, mid_img, patient_data, pre_idx) -> tensor`` used
            during **evaluation** to produce the model input from a
            ``(1, 3, H, W)`` pre tensor and ``(1, 3, H, W)`` mid tensor.
            During **training** the same adapter is called with
            ``(pre_image_batch, mid_image_batch, None, None)`` and should
            return ``(B, C, H, W)``.  Only used when ``uses_pre=True``.
        trainable_params: if provided, only these parameters are optimised
            (useful when the model contains frozen sub-networks).

    Returns:
        Aggregated test metrics dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Training {model_name}")
    print(f"  epochs={epochs}  lr={lr}  batch_size={batch_size}  image_size={image_size}")
    print(f"  uses_pre={uses_pre}  device={device}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    cache_dir = Path(data_dir) / "cache" / "train"
    if (cache_dir / "meta.json").exists():
        print(f"  Using memmap cache: {cache_dir}")
        train_ds = _CachedSliceDataset(str(cache_dir), image_size=image_size, augment=True)
    else:
        print(f"  No cache found at {cache_path} — loading from NIfTI (slow).")
        print(f"  TIP: run `python scripts/cache_slices.py` first for 10x speedup.")
        train_ds = HNTSMRGSliceDataset(
            data_dir=data_dir, split="train", image_size=image_size, augment=True,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    if trainable_params is not None:
        params = trainable_params
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    n_params = sum(p.numel() for p in params)
    print(f"  Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_agg_dsc = -1.0
    training_log: list[dict] = []

    from tqdm import tqdm

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            mid_img = batch["mid_image"].to(device)    # (B, 3, H, W)
            mid_mask = batch["mid_mask"].to(device)     # (B, 1, H, W)

            if uses_pre:
                pre_img = batch["pre_image"].to(device)  # (B, 3, H, W)
                if input_adapter is not None:
                    inp = input_adapter(pre_img, mid_img, None, None)
                else:
                    inp = torch.cat([pre_img, mid_img], dim=1)  # (B, 6, H, W)
            else:
                inp = mid_img

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(inp)
                # Ensure single-channel output for loss
                if logits.shape[1] > 1:
                    # Multi-class: take channel 1 as foreground logit
                    logits = logits[:, 1:2]
                loss = _combined_loss(logits, mid_mask)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{running_loss/n_batches:.4f}")

        epoch_loss = running_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        log_entry: dict = {"epoch": epoch, "train_loss": epoch_loss, "time_s": elapsed}

        # --------------------------------------------------------------
        # Validation every 5 epochs
        # --------------------------------------------------------------
        val_agg_dsc = None
        if True:  # validate every epoch
            val_agg, _ = _evaluate_split(
                model, data_dir, "val", image_size, device,
                uses_pre=uses_pre, input_adapter=input_adapter,
            )
            val_agg_dsc = val_agg.get("agg_dsc_mean", 0.0)
            log_entry["val_agg_dsc"] = val_agg_dsc
            print(
                f"  Epoch {epoch:3d}/{epochs} | loss={epoch_loss:.4f} | "
                f"val aggDSC={val_agg_dsc:.4f} | {elapsed:.1f}s"
            )

            if val_agg_dsc > best_val_agg_dsc:
                best_val_agg_dsc = val_agg_dsc
                torch.save({"model": model.state_dict(), "epoch": epoch}, out / "best.pth")
        else:
            print(f"  Epoch {epoch:3d}/{epochs} | loss={epoch_loss:.4f} | {elapsed:.1f}s")

        training_log.append(log_entry)

    # ------------------------------------------------------------------
    # Load best checkpoint and evaluate on test split
    # ------------------------------------------------------------------
    best_ckpt = out / "best.pth"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model"], strict=False)
        print(f"\nLoaded best checkpoint (epoch {state.get('epoch', '?')})")
    else:
        print("\nNo best checkpoint saved (val never ran?); evaluating last epoch.")

    test_agg, test_per_patient = _evaluate_split(
        model, data_dir, split, image_size, device,
        uses_pre=uses_pre, input_adapter=input_adapter,
    )

    print(f"\n{model_name} Test Results ({split}):")
    print(f"  aggDSC  : {test_agg.get('agg_dsc_mean', 0):.4f} +/- {test_agg.get('agg_dsc_std', 0):.4f}")
    print(f"  GTVp DSC: {test_agg.get('dsc_gtvp_mean', 0):.4f}")
    print(f"  GTVn DSC: {test_agg.get('dsc_gtvn_mean', 0):.4f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    results = {
        "aggregate": test_agg,
        "per_patient": test_per_patient,
        "config": {
            "model_name": model_name,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "image_size": image_size,
            "uses_pre": uses_pre,
            "trainable_params": n_params,
            "best_val_agg_dsc": best_val_agg_dsc,
        },
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(out / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"Results saved: {out / 'metrics.json'}")
    return test_agg
