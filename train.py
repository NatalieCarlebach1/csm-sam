"""
CSM-SAM Training Script.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume checkpoints/csmsam/latest.pth
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
from tqdm import tqdm

from csmsam.datasets import build_dataloaders
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
    model.train()
    total_losses = {"total": 0.0, "dice": 0.0, "bce": 0.0, "change": 0.0}
    n_steps = 0
    accumulate = cfg.training.accumulate_grad_batches

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for step, batch in enumerate(pbar):
        mid_images = batch["mid_image"].to(device)       # (B, 3, H, W)
        pre_images = batch["pre_image"].to(device)        # (B, 3, H, W)
        mid_masks = batch["mid_mask"].to(device)          # (B, 1, H, W)
        pre_masks = batch["pre_mask"].to(device)          # (B, 1, H, W)
        weeks = batch["weeks_elapsed"]

        if isinstance(weeks, list):
            weeks = torch.tensor(weeks, dtype=torch.long)
        weeks = weeks.to(device)

        # Build per-sample pre-RT memory
        # During training we use single-slice memory (lightweight)
        with autocast(enabled=cfg.training.amp):
            # Encode pre-RT memory (batch of single slices)
            pre_imgs_5d = pre_images.unsqueeze(1)  # (B, 1, 3, H, W)
            pre_masks_5d = pre_masks.unsqueeze(1)  # (B, 1, 1, H, W)
            M_pre = model.encode_pre_rt(pre_imgs_5d, pre_masks_5d)

            # Reset within-session memory for training (each slice is independent)
            model.reset_mid_session_memory()

            # Forward pass
            out = model(
                mid_images=mid_images,
                M_pre=M_pre,
                pre_images=pre_images,
                weeks_elapsed=weeks,
                return_change_map=True,
            )

            losses = loss_fn(
                pred_masks=out["masks"],
                target_masks=mid_masks,
                change_logits=out.get("change_map"),
                pre_masks=pre_masks,
                mid_masks=mid_masks,
            )
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
            total_losses[k] += losses.get(k, torch.tensor(0.0)).item()
        n_steps += 1

        pbar.set_postfix({
            "loss": f"{losses['total'].item():.4f}",
            "dice": f"{losses.get('dice', 0):.4f}",
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
    Validation: full 3D volume inference.
    Encodes pre-RT memory once, segments mid-RT slice by slice.
    """
    model.eval()
    patient_metrics = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        # Volume-level batch (B=1 from val loader)
        pre_images = batch["pre_images"].squeeze(0).to(device)   # (N, 3, H, W)
        mid_images = batch["mid_images"].squeeze(0).to(device)   # (N, 3, H, W)
        mid_masks_gtvp = batch["mid_masks_gtvp"].squeeze(0)      # (N, 1, H, W)
        mid_masks_gtvn = batch["mid_masks_gtvn"].squeeze(0)      # (N, 1, H, W)
        pre_masks = batch["pre_masks"].squeeze(0).to(device)     # (N, 1, H, W)
        weeks = batch["weeks_elapsed"]
        if isinstance(weeks, (list, tuple)):
            weeks = weeks[0]
        weeks = torch.tensor([int(weeks)], dtype=torch.long, device=device)

        N = pre_images.shape[0]

        # Encode pre-RT memory from all slices
        M_pre = model.encode_pre_rt(
            pre_images.unsqueeze(0),
            pre_masks.unsqueeze(0),
        )  # (1, N_mem, C)

        # Segment mid-RT slice by slice
        model.reset_mid_session_memory()
        pred_gtvp_slices = []
        pred_gtvn_slices = []

        for i in range(N):
            mid_slice = mid_images[i].unsqueeze(0)   # (1, 3, H, W)
            pre_slice = pre_images[i].unsqueeze(0)   # (1, 3, H, W)

            out = model(
                mid_images=mid_slice,
                M_pre=M_pre,
                pre_images=pre_slice,
                weeks_elapsed=weeks,
                return_change_map=False,
            )
            # Binary prediction
            pred = (torch.sigmoid(out["masks"]) > cfg.evaluation.threshold).squeeze().cpu().numpy()
            pred_gtvp_slices.append(pred)
            pred_gtvn_slices.append(pred)  # simplified: same pred for both structures

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
    model = CSMSAM.from_pretrained(
        sam2_checkpoint=cfg.model.sam2_checkpoint,
        sam2_cfg=cfg.model.sam2_cfg,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        n_memory_frames=cfg.model.n_memory_frames,
        spatial_pool_size=cfg.model.spatial_pool_size,
        max_weeks=cfg.model.max_weeks,
        dropout=cfg.model.dropout,
    ).to(device)

    # Print param counts
    counts = model.count_trainable_params()
    print(f"Trainable parameters:")
    for name, count in counts.items():
        print(f"  {name}: {count:,}")

    # Build data
    print("\nBuilding data loaders...")
    loaders = build_dataloaders(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
        pin_memory=cfg.data.pin_memory,
    )

    # Build optimizer, scheduler, loss
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(loaders["train"])
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    loss_fn = CombinedLoss(
        lambda_dice=cfg.loss.lambda_dice,
        lambda_bce=cfg.loss.lambda_bce,
        lambda_change=cfg.loss.lambda_change,
        change_loss_weights=cfg.loss.change_class_weights,
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
    print(f"\nStarting training: epochs {start_epoch}–{cfg.training.epochs}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Effective batch size: {cfg.data.batch_size * cfg.training.accumulate_grad_batches}")
    print()

    metrics_history = []

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        t0 = time.time()

        # Train
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
