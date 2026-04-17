"""
CSM-SAM Training Script.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume checkpoints/csmsam/latest.pth
    python train.py --config configs/default.yaml --fold 0 --n_folds 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from csmsam.datasets import HNTSMRGDataset, HNTSMRGSliceDataset, build_dataloaders
from csmsam.losses import CombinedLoss
from csmsam.modeling import CSMSAM
from csmsam.utils.metrics import aggregate_metrics, compute_agg_dsc


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: CSMSAM, cfg) -> torch.optim.Optimizer:
    """Build AdamW with separate learning rates for novel vs. decoder params."""
    param_groups = model.get_trainable_params()

    base_lr = cfg.training.lr
    optimizer_groups = []
    for group in param_groups:
        lr = base_lr * group.get("lr_scale", 1.0)
        optimizer_groups.append({
            "params": [p for p in group["params"] if p.requires_grad],
            "lr": lr,
        })

    # Filter empty groups
    optimizer_groups = [g for g in optimizer_groups if len(g["params"]) > 0]

    return AdamW(
        optimizer_groups,
        weight_decay=cfg.training.weight_decay,
    )


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    """Build cosine LR scheduler with linear warmup."""
    warmup_steps = cfg.training.warmup_epochs * steps_per_epoch
    total_steps = cfg.training.epochs * steps_per_epoch

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)

    return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])


def _weighted_2ch_target(
    gtvp: torch.Tensor,
    gtvn: torch.Tensor,
) -> torch.Tensor:
    """Stack GTVp (ch0) and GTVn (ch1) targets into a (B, 2, H, W) tensor."""
    return torch.cat([gtvp, gtvn], dim=1)


def _apply_channel_loss(
    loss_fn: CombinedLoss,
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
    change_logits: torch.Tensor | None,
    pre_masks: torch.Tensor | None,
    mid_masks: torch.Tensor | None,
    gtvn_weight: float,
) -> dict[str, torch.Tensor]:
    """Thin pass-through to the 2-channel CombinedLoss.

    CombinedLoss natively consumes (B, 2, H, W) pred+target and applies
    its own ``gtvn_weight`` internally, so this helper just forwards the
    call with consistent kwargs. The ``gtvn_weight`` argument is unused
    here (kept in the signature for call-site compatibility and because
    the effective weighting is handled by the loss module).
    """
    del gtvn_weight  # handled inside CombinedLoss
    return loss_fn(
        pred_masks=pred_masks,
        target_masks=target_masks,
        change_logits=change_logits,
        pre_masks=pre_masks,
        mid_masks=mid_masks,
    )


def _get_kl_beta(epoch: int, cfg) -> float:
    """Compute the current KL annealing coefficient β(epoch)."""
    cl_cfg = getattr(cfg, "change_latent", None)
    if cl_cfg is None or not getattr(cl_cfg, "enabled", False):
        return 0.0
    from csmsam.modeling.change_latent import kl_beta as _kl_beta_fn
    return _kl_beta_fn(
        epoch=epoch,
        beta_max=getattr(cl_cfg, "lambda_kl", 0.1),
        warmup_epochs=getattr(cl_cfg, "kl_warmup_epochs", 10),
    )


def train_one_epoch(
    model: CSMSAM,
    loader,
    optimizer,
    scheduler,
    loss_fn: CombinedLoss,
    scaler: GradScaler,
    device: str,
    cfg,
    epoch: int,
) -> dict:
    """Single-slice training (fallback when sequence_mode=false)."""
    model.train()
    total_losses = {"total": 0.0, "dice": 0.0, "bce": 0.0, "change": 0.0, "kl": 0.0}
    n_steps = 0
    accumulate = cfg.training.accumulate_grad_batches
    gtvn_weight = getattr(cfg.loss, "gtvn_weight", 1.0)
    kl_beta_val = _get_kl_beta(epoch, cfg)

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for step, batch in enumerate(pbar):
        mid_images = batch["mid_image"].to(device)                 # (B, 3, H, W)
        pre_images = batch["pre_image"].to(device)                 # (B, 3, H, W)
        mid_masks = batch["mid_mask"].to(device)                   # (B, 1, H, W) combined
        pre_masks = batch["pre_mask"].to(device)                   # (B, 1, H, W) combined
        mid_masks_gtvp = batch["mid_mask_gtvp"].to(device)         # (B, 1, H, W)
        mid_masks_gtvn = batch["mid_mask_gtvn"].to(device)         # (B, 1, H, W)
        pre_masks_gtvp = batch["pre_mask_gtvp"].to(device)         # (B, 1, H, W)
        pre_masks_gtvn = batch["pre_mask_gtvn"].to(device)         # (B, 1, H, W)
        weeks = batch["weeks_elapsed"]

        if isinstance(weeks, list):
            weeks = torch.tensor(weeks, dtype=torch.long)
        weeks = weeks.to(device)

        # 2-channel target: [GTVp, GTVn]
        target_masks = _weighted_2ch_target(mid_masks_gtvp, mid_masks_gtvn)

        with autocast(enabled=cfg.training.amp):
            # Encode pre-RT memory (batch of single slices)
            pre_imgs_5d = pre_images.unsqueeze(1)        # (B, 1, 3, H, W)
            pre_masks_5d = pre_masks.unsqueeze(1)        # (B, 1, 1, H, W)
            M_pre = model.encode_pre_rt(pre_imgs_5d, pre_masks_5d)

            # Reset within-session memory for training (each slice is independent)
            model.reset_mid_session_memory()

            out = model(
                mid_images=mid_images,
                M_pre=M_pre,
                pre_images=pre_images,
                weeks_elapsed=weeks,
                pre_gtvp_mask=pre_masks_gtvp,
                pre_gtvn_mask=pre_masks_gtvn,
                return_change_map=True,
                detach_memory=False,
            )

            losses = _apply_channel_loss(
                loss_fn=loss_fn,
                pred_masks=out["masks"],               # (B, 2, H, W)
                target_masks=target_masks,             # (B, 2, H, W)
                change_logits=out.get("change_map"),
                pre_masks=pre_masks,                   # combined (B, 1, H, W)
                mid_masks=mid_masks,                   # combined (B, 1, H, W)
                gtvn_weight=gtvn_weight,
            )
            # KL loss from variational change latent (annealed)
            kl = out.get("kl_loss")
            if kl is not None and kl_beta_val > 0:
                kl_weight = getattr(getattr(cfg, "change_latent", None), "lambda_kl", 0.1)
                total_with_kl = losses["total"] + kl_weight * kl_beta_val * kl
                losses = dict(losses)
                losses["total"] = total_with_kl
                losses["kl"] = kl.detach()
            loss = losses["total"] / accumulate

        scaler.scale(loss).backward()

        if (step + 1) % accumulate == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                cfg.training.gradient_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        for k in total_losses:
            v = losses.get(k, torch.tensor(0.0))
            total_losses[k] += v.item() if torch.is_tensor(v) else float(v)
        n_steps += 1

        pbar.set_postfix({
            "loss": f"{losses['total'].item():.4f}",
            "dice": f"{float(losses.get('dice', 0)):.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    return {k: v / max(n_steps, 1) for k, v in total_losses.items()}


def train_sequence_epoch(
    model: CSMSAM,
    volume_loader,
    optimizer,
    scheduler,
    loss_fn: CombinedLoss,
    scaler: GradScaler,
    device: str,
    cfg,
    epoch: int,
) -> dict:
    """
    Sequence-mode training.

    For each patient volume:
      - Encode pre-RT memory once from ALL pre-RT slices.
      - Sample S consecutive mid-RT slices.
      - Reset M_mid, then forward slices one-by-one WITHOUT resetting memory,
        accumulating loss.
      - Backward once after the full sequence (per effective grad-accum step).
    """
    model.train()
    total_losses = {"total": 0.0, "dice": 0.0, "bce": 0.0, "change": 0.0, "kl": 0.0}
    n_steps = 0
    accumulate = cfg.training.accumulate_grad_batches
    seq_len = max(1, int(cfg.training.sequence_length))
    gtvn_weight = getattr(cfg.loss, "gtvn_weight", 1.0)
    kl_beta_val = _get_kl_beta(epoch, cfg)
    kl_weight = getattr(getattr(cfg, "change_latent", None), "lambda_kl", 0.1)

    optimizer.zero_grad()
    pbar = tqdm(volume_loader, desc=f"Epoch {epoch} [seq]", leave=False)

    for step, batch in enumerate(pbar):
        # Volume loader uses batch_size=1. Squeeze the batch dim.
        pre_images = batch["pre_images"].squeeze(0).to(device)          # (N, 3, H, W)
        mid_images = batch["mid_images"].squeeze(0).to(device)          # (N, 3, H, W)
        pre_masks_combined = batch["pre_masks"].squeeze(0).to(device)   # (N, 1, H, W)
        mid_masks_combined = batch["mid_masks"].squeeze(0).to(device)   # (N, 1, H, W)
        pre_gtvp = batch["pre_masks_gtvp"].squeeze(0).to(device)        # (N, 1, H, W)
        pre_gtvn = batch["pre_masks_gtvn"].squeeze(0).to(device)        # (N, 1, H, W)
        mid_gtvp = batch["mid_masks_gtvp"].squeeze(0).to(device)        # (N, 1, H, W)
        mid_gtvn = batch["mid_masks_gtvn"].squeeze(0).to(device)        # (N, 1, H, W)

        weeks = batch["weeks_elapsed"]
        if isinstance(weeks, (list, tuple)):
            weeks = weeks[0]
        weeks_scalar = int(weeks.item()) if torch.is_tensor(weeks) else int(weeks)

        N = mid_images.shape[0]
        if N == 0:
            continue

        # Pick a random consecutive window of S slices, biased toward tumor
        # slices when available.
        S = min(seq_len, N)
        tumor_slice_mask = (mid_masks_combined.flatten(1).sum(dim=1) > 0).cpu().numpy()
        tumor_indices = np.where(tumor_slice_mask)[0]
        if len(tumor_indices) > 0:
            center = int(np.random.choice(tumor_indices))
            start = max(0, min(center - S // 2, N - S))
        else:
            start = random.randint(0, max(0, N - S))
        window = slice(start, start + S)

        with autocast(enabled=cfg.training.amp):
            # Encode pre-RT memory ONCE from all pre-RT slices of this patient.
            M_pre = model.encode_pre_rt(
                pre_images.unsqueeze(0),                  # (1, N, 3, H, W)
                pre_masks_combined.unsqueeze(0),          # (1, N, 1, H, W)
            )  # (1, N_mem, C)

            # Reset within-session memory ONCE for the whole sequence.
            model.reset_mid_session_memory()

            seq_loss = None
            seq_loss_components = {"dice": 0.0, "bce": 0.0, "change": 0.0, "kl": 0.0}
            n_in_seq = 0

            for i in range(window.start, window.stop):
                mid_slice = mid_images[i].unsqueeze(0)           # (1, 3, H, W)
                pre_slice = pre_images[i].unsqueeze(0)           # (1, 3, H, W)
                pgp = pre_gtvp[i].unsqueeze(0)
                pgn = pre_gtvn[i].unsqueeze(0)
                mgp = mid_gtvp[i].unsqueeze(0)
                mgn = mid_gtvn[i].unsqueeze(0)
                pm = pre_masks_combined[i].unsqueeze(0)
                mm = mid_masks_combined[i].unsqueeze(0)
                weeks_t = torch.tensor([weeks_scalar], dtype=torch.long, device=device)

                out = model(
                    mid_images=mid_slice,
                    M_pre=M_pre,
                    pre_images=pre_slice,
                    weeks_elapsed=weeks_t,
                    pre_gtvp_mask=pgp,
                    pre_gtvn_mask=pgn,
                    return_change_map=True,
                    detach_memory=False,           # keep gradients through the gate
                )

                target_masks = _weighted_2ch_target(mgp, mgn)
                losses = _apply_channel_loss(
                    loss_fn=loss_fn,
                    pred_masks=out["masks"],
                    target_masks=target_masks,
                    change_logits=out.get("change_map"),
                    pre_masks=pm,
                    mid_masks=mm,
                    gtvn_weight=gtvn_weight,
                )

                # KL from variational change latent (annealed)
                kl = out.get("kl_loss")
                if kl is not None and kl_beta_val > 0:
                    total_with_kl = losses["total"] + kl_weight * kl_beta_val * kl
                    losses = dict(losses)
                    losses["total"] = total_with_kl
                    losses["kl"] = kl.detach()

                step_loss = losses["total"]
                seq_loss = step_loss if seq_loss is None else seq_loss + step_loss
                for k in seq_loss_components:
                    v = losses.get(k, None)
                    if v is not None:
                        seq_loss_components[k] += v.item() if torch.is_tensor(v) else float(v)
                n_in_seq += 1

            seq_loss = seq_loss / max(n_in_seq, 1)
            loss = seq_loss / accumulate

        scaler.scale(loss).backward()

        if (step + 1) % accumulate == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                cfg.training.gradient_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        total_losses["total"] += seq_loss.item()
        for k in ("dice", "bce", "change", "kl"):
            total_losses[k] += seq_loss_components[k] / max(n_in_seq, 1)
        n_steps += 1

        pbar.set_postfix({
            "loss": f"{seq_loss.item():.4f}",
            "S": S,
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    return {k: v / max(n_steps, 1) for k, v in total_losses.items()}


@torch.no_grad()
def validate(
    model: CSMSAM,
    loader,
    device: str,
    cfg,
) -> dict:
    """
    Validation: full 3D volume inference with per-channel (GTVp, GTVn) evaluation.

    Encodes pre-RT memory once, segments mid-RT slice by slice with the
    within-session memory propagating across slices.
    """
    model.eval()
    patient_metrics = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        # Volume-level batch (B=1 from val loader)
        pre_images = batch["pre_images"].squeeze(0).to(device)       # (N, 3, H, W)
        mid_images = batch["mid_images"].squeeze(0).to(device)       # (N, 3, H, W)
        pre_masks = batch["pre_masks"].squeeze(0).to(device)         # (N, 1, H, W) combined
        pre_gtvp = batch["pre_masks_gtvp"].squeeze(0).to(device)     # (N, 1, H, W)
        pre_gtvn = batch["pre_masks_gtvn"].squeeze(0).to(device)     # (N, 1, H, W)
        mid_masks_gtvp = batch["mid_masks_gtvp"].squeeze(0)          # (N, 1, H, W)
        mid_masks_gtvn = batch["mid_masks_gtvn"].squeeze(0)          # (N, 1, H, W)

        weeks = batch["weeks_elapsed"]
        if isinstance(weeks, (list, tuple)):
            weeks = weeks[0]
        weeks_scalar = int(weeks.item()) if torch.is_tensor(weeks) else int(weeks)
        weeks_t = torch.tensor([weeks_scalar], dtype=torch.long, device=device)

        N = pre_images.shape[0]

        # Encode pre-RT memory from all slices
        M_pre = model.encode_pre_rt(
            pre_images.unsqueeze(0),
            pre_masks.unsqueeze(0),
        )  # (1, N_mem, C)

        # Segment mid-RT slice by slice with within-session memory propagation.
        model.reset_mid_session_memory()
        pred_gtvp_slices = []
        pred_gtvn_slices = []

        threshold = cfg.evaluation.threshold

        for i in range(N):
            mid_slice = mid_images[i].unsqueeze(0)           # (1, 3, H, W)
            pre_slice = pre_images[i].unsqueeze(0)           # (1, 3, H, W)
            pgp = pre_gtvp[i].unsqueeze(0)
            pgn = pre_gtvn[i].unsqueeze(0)

            out = model(
                mid_images=mid_slice,
                M_pre=M_pre,
                pre_images=pre_slice,
                weeks_elapsed=weeks_t,
                pre_gtvp_mask=pgp,
                pre_gtvn_mask=pgn,
                return_change_map=False,
                detach_memory=True,           # save memory in eval
            )
            masks = out["masks"]                             # (1, 2, H, W)
            probs = torch.sigmoid(masks)
            binary = (probs > threshold).squeeze(0).cpu().numpy()   # (2, H, W)
            pred_gtvp_slices.append(binary[0])
            pred_gtvn_slices.append(binary[1])

        # Stack to volumes
        pred_gtvp = np.stack(pred_gtvp_slices)  # (N, H, W)
        pred_gtvn = np.stack(pred_gtvn_slices)  # (N, H, W)
        gt_gtvp = (mid_masks_gtvp > 0.5).squeeze(1).numpy()
        gt_gtvn = (mid_masks_gtvn > 0.5).squeeze(1).numpy()

        metrics = compute_agg_dsc(pred_gtvp, pred_gtvn, gt_gtvp, gt_gtvn)
        patient_metrics.append(metrics)

    agg = aggregate_metrics(patient_metrics)
    return agg


def save_checkpoint(
    model: CSMSAM,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    cfg,
    is_best: bool = False,
):
    out_dir = Path(cfg.checkpoint.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": OmegaConf.to_container(cfg),
    }

    # Latest checkpoint
    latest_path = out_dir / "latest.pth"
    torch.save(state, latest_path)

    # Periodic checkpoint
    if epoch % cfg.checkpoint.save_every_n_epochs == 0:
        torch.save(state, out_dir / f"epoch_{epoch:04d}.pth")

    # Best checkpoint
    if is_best:
        torch.save(state, out_dir / "best.pth")
        print(f"  New best model saved at epoch {epoch}")

    # Remove old checkpoints beyond keep_last_n
    periodic = sorted(out_dir.glob("epoch_*.pth"))
    for old in periodic[:-cfg.checkpoint.keep_last_n]:
        old.unlink()


def load_checkpoint(model, optimizer, scheduler, path: str, device: str) -> tuple[int, dict]:
    print(f"Loading checkpoint: {path}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and state.get("scheduler_state_dict"):
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return state["epoch"], state.get("metrics", {})


def _apply_kfold(
    cfg,
    fold: int,
    n_folds: int,
    seed: int,
) -> dict | None:
    """
    Override the default built-in train/val split with a k-fold split.

    Returns a replacement `loaders` dict, or None if the cv utility can't
    be imported (caller falls back to the default split).
    """
    try:
        from csmsam.utils.cv import kfold_split  # type: ignore
    except Exception as e:
        print(f"  [warn] csmsam.utils.cv.kfold_split unavailable ({e}); "
              f"falling back to built-in train/val split.")
        return None

    image_size = cfg.data.image_size

    # Use the slice dataset for training, volume dataset for validation.
    # We build both over split='train' then partition patient-wise via
    # the kfold utility.
    train_slice_ds = HNTSMRGSliceDataset(
        cfg.data.data_dir, split="train", image_size=image_size, augment=True
    )
    train_vol_ds = HNTSMRGDataset(
        cfg.data.data_dir, split="train", image_size=image_size
    )
    val_vol_ds = HNTSMRGDataset(
        cfg.data.data_dir, split="train", image_size=image_size
    )

    try:
        # kfold_split works on patient-ID lists: partition patients, then
        # map back to dataset indices for each split.
        patient_ids = [d.name for d in train_vol_ds.patient_dirs]
        train_ids, val_ids = kfold_split(patient_ids, fold=fold, n_folds=n_folds, seed=seed)
        train_set = set(train_ids)
        val_set = set(val_ids)

        # Volume datasets are indexed by patient, so partition directly.
        pid_to_vol_idx = {d.name: i for i, d in enumerate(train_vol_ds.patient_dirs)}
        train_vol_idx = [pid_to_vol_idx[p] for p in train_ids if p in pid_to_vol_idx]
        val_vol_idx = [pid_to_vol_idx[p] for p in val_ids if p in pid_to_vol_idx]

        # Slice dataset is indexed by (patient_dir, slice_idx) pairs.
        train_slice_idx = [
            i for i, (pdir, _) in enumerate(train_slice_ds.tumor_slices + train_slice_ds.bg_slices)
            if pdir.name in train_set
        ]
    except Exception as e:
        print(f"  [warn] kfold_split failed ({e}); falling back to built-in split.")
        return None

    train_slice_subset = Subset(train_slice_ds, train_slice_idx)
    train_vol_subset = Subset(train_vol_ds, train_vol_idx)
    val_vol_subset = Subset(val_vol_ds, val_vol_idx)

    train_loader = DataLoader(
        train_slice_subset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    train_vol_loader = DataLoader(
        train_vol_subset,
        batch_size=1,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    val_loader = DataLoader(
        val_vol_subset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    return {
        "train": train_loader,
        "train_volumes": train_vol_loader,
        "val": val_loader,
    }


def _build_train_volume_loader(cfg) -> DataLoader:
    """Build a batch-size-1 volume loader over the training split for sequence-mode."""
    ds = HNTSMRGDataset(cfg.data.data_dir, split="train", image_size=cfg.data.image_size)
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )


def main():
    parser = argparse.ArgumentParser(description="Train CSM-SAM")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sam2_checkpoint", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--sequence_train",
        dest="sequence_train",
        action="store_true",
        default=None,
        help="Enable sequence-mode training (default: controlled by config).",
    )
    parser.add_argument(
        "--no_sequence_train",
        dest="sequence_train",
        action="store_false",
        help="Disable sequence-mode training and use per-slice batches.",
    )
    parser.add_argument("--sequence_length", type=int, default=None,
                        help="Override training.sequence_length.")
    parser.add_argument("--fold", type=int, default=None, help="k-fold index (0-based)")
    parser.add_argument("--n_folds", type=int, default=5, help="number of folds")
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.data_dir is not None:
        cfg.data.data_dir = args.data_dir
    if args.output_dir is not None:
        cfg.checkpoint.output_dir = args.output_dir
    if args.sam2_checkpoint is not None:
        cfg.model.sam2_checkpoint = args.sam2_checkpoint
    if args.no_wandb:
        cfg.logging.use_wandb = False
    if args.sequence_train is not None:
        cfg.training.sequence_mode = bool(args.sequence_train)
    if args.sequence_length is not None:
        cfg.training.sequence_length = int(args.sequence_length)

    # Backfill defaults if the config file hasn't been regenerated.
    if "sequence_mode" not in cfg.training:
        cfg.training.sequence_mode = True
    if "sequence_length" not in cfg.training:
        cfg.training.sequence_length = 4
    if "gtvn_weight" not in cfg.loss:
        cfg.loss.gtvn_weight = 1.0

    # Setup
    set_seed(cfg.hardware.seed)
    device = cfg.hardware.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print("\n" + "=" * 60)
    print("CSM-SAM Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    # W&B logging
    wandb_run = None
    if cfg.logging.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.wandb_entity,
                config=OmegaConf.to_container(cfg),
            )
        except ImportError:
            print("wandb not installed, skipping")

    # Build model
    print("\nBuilding model...")
    temporal_encoder_type = cfg.model.get("temporal_encoder_type", "continuous") \
        if hasattr(cfg.model, "get") else getattr(cfg.model, "temporal_encoder_type", "continuous")
    temporal_hidden_dim = cfg.model.get("temporal_hidden_dim", 128) \
        if hasattr(cfg.model, "get") else getattr(cfg.model, "temporal_hidden_dim", 128)
    temporal_n_frequencies = cfg.model.get("temporal_n_frequencies", 6) \
        if hasattr(cfg.model, "get") else getattr(cfg.model, "temporal_n_frequencies", 6)

    model = CSMSAM.from_pretrained(
        sam2_checkpoint=cfg.model.sam2_checkpoint,
        sam2_cfg=cfg.model.sam2_cfg,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        n_memory_frames=cfg.model.n_memory_frames,
        spatial_pool_size=cfg.model.spatial_pool_size,
        max_weeks=cfg.model.max_weeks,
        dropout=cfg.model.dropout,
        temporal_encoder_type=temporal_encoder_type,
        temporal_hidden_dim=temporal_hidden_dim,
        temporal_n_frequencies=temporal_n_frequencies,
    ).to(device)

    # Print param counts
    counts = model.count_trainable_params()
    print(f"Trainable parameters:")
    for name, count in counts.items():
        print(f"  {name}: {count:,}")

    # Build data
    print("\nBuilding data loaders...")
    loaders = None
    if args.fold is not None:
        print(f"  Using k-fold split: fold={args.fold} / {args.n_folds}")
        loaders = _apply_kfold(cfg, args.fold, args.n_folds, cfg.hardware.seed)

    if loaders is None:
        loaders = build_dataloaders(
            data_dir=cfg.data.data_dir,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            image_size=cfg.data.image_size,
            pin_memory=cfg.data.pin_memory,
        )

    # For sequence mode we need a volume-level train loader.
    sequence_mode = bool(cfg.training.sequence_mode) and int(cfg.training.sequence_length) > 1
    if sequence_mode and "train_volumes" not in loaders:
        print("  Building volume loader for sequence training...")
        loaders["train_volumes"] = _build_train_volume_loader(cfg)

    # Build optimizer, scheduler, loss
    optimizer = build_optimizer(model, cfg)
    if sequence_mode:
        steps_per_epoch = len(loaders["train_volumes"])
    else:
        steps_per_epoch = len(loaders["train"])
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    loss_fn = CombinedLoss(
        lambda_dice=cfg.loss.lambda_dice,
        lambda_bce=cfg.loss.lambda_bce,
        lambda_change=cfg.loss.lambda_change,
        change_loss_weights=cfg.loss.change_class_weights,
        gtvn_weight=getattr(cfg.loss, "gtvn_weight", 1.0),
    )
    scaler = GradScaler(enabled=cfg.training.amp and device != "cpu")

    # Resume
    start_epoch = 1
    best_metric = 0.0
    if args.resume:
        start_epoch, prev_metrics = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        start_epoch += 1
        best_metric = prev_metrics.get(f"{cfg.evaluation.primary_metric}_mean", 0.0)

    # Training loop
    print(f"\nStarting training: epochs {start_epoch}-{cfg.training.epochs}")
    print(f"Mode: {'sequence (S=' + str(cfg.training.sequence_length) + ')' if sequence_mode else 'single-slice'}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Effective batch size: {cfg.data.batch_size * cfg.training.accumulate_grad_batches}")
    print()

    metrics_history = []

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        t0 = time.time()

        # Train
        if sequence_mode:
            train_metrics = train_sequence_epoch(
                model, loaders["train_volumes"], optimizer, scheduler,
                loss_fn, scaler, device, cfg, epoch,
            )
        else:
            train_metrics = train_one_epoch(
                model, loaders["train"], optimizer, scheduler, loss_fn, scaler, device, cfg, epoch
            )

        # Validate periodically
        val_metrics = {}
        is_best = False
        if epoch % cfg.evaluation.val_every_n_epochs == 0:
            val_metrics = validate(model, loaders["val"], device, cfg)
            current = val_metrics.get(f"{cfg.evaluation.primary_metric}_mean", 0.0)
            is_best = current > best_metric
            if is_best:
                best_metric = current

        # Log
        elapsed = time.time() - t0
        log_str = (
            f"Epoch {epoch:03d}/{cfg.training.epochs} | "
            f"loss={train_metrics['total']:.4f} dice={train_metrics['dice']:.4f}"
        )
        if val_metrics:
            agg = val_metrics.get("agg_dsc_mean", 0)
            log_str += f" | val_aggDSC={agg:.4f} (best={best_metric:.4f})"
        log_str += f" | {elapsed:.1f}s"
        print(log_str)

        # Save
        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, cfg, is_best)

        if wandb_run:
            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
            log_dict["epoch"] = epoch
            wandb_run.log(log_dict)

        metrics_history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

    # Save full metrics history
    out_dir = Path(cfg.checkpoint.output_dir)
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    print(f"\nTraining complete. Best {cfg.evaluation.primary_metric}: {best_metric:.4f}")
    print(f"Best model: {out_dir}/best.pth")
    print(f"\nNext step: python test.py --checkpoint {out_dir}/best.pth")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
