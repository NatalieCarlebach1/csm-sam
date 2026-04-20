"""
CSM-SAM Training on BraTS-GLI 2024 (longitudinal post-treatment glioma).

Differences from train.py (HNTS-MRG):
  - Single binary tumor mask (no dual GTVp / GTVn heads).
    Both decoder heads receive the same binary mask; loss is averaged.
  - Volume-level dataset uses keys pre_image / mid_image (not pre_images/mid_images).
  - Primary metric: volumetric DSC (not aggDSC).
  - weeks_elapsed is fixed from config (BraTS has no per-patient timing).

Single GPU:
    python train_brats.py --config configs/brats.yaml

Multi-GPU (8 GPUs):
    torchrun --nproc_per_node=8 train_brats.py --config configs/brats.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from csmsam.datasets.brats_gli import BraTSGLIDataset, BraTSGLISliceDataset
from csmsam.losses import CombinedLoss
from csmsam.modeling import CSMSAM
from csmsam.utils.metrics import compute_dice


# ---------------------------------------------------------------------------
# DDP helpers (identical to train.py)
# ---------------------------------------------------------------------------

def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank): return rank == 0


def unwrap(model):
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def reduce_scalar(value, world_size):
    if world_size == 1:
        return value
    t = torch.tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / world_size).item()


# ---------------------------------------------------------------------------
# Seed / optimiser / scheduler
# ---------------------------------------------------------------------------

def set_seed(seed, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def build_optimizer(model, cfg):
    groups = unwrap(model).get_trainable_params()
    base_lr = cfg.training.lr
    opt_groups = [
        {"params": [p for p in g["params"] if p.requires_grad],
         "lr": base_lr * g.get("lr_scale", 1.0)}
        for g in groups
    ]
    opt_groups = [g for g in opt_groups if g["params"]]
    return AdamW(opt_groups, weight_decay=cfg.training.weight_decay)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    warmup = cfg.training.warmup_epochs * steps_per_epoch
    total  = cfg.training.epochs * steps_per_epoch
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=total - warmup, eta_min=1e-6),
        ],
        milestones=[warmup],
    )


# ---------------------------------------------------------------------------
# BraTS-specific: single binary mask → both decoder channels
# ---------------------------------------------------------------------------

def _dual_target(mask):
    """Replicate binary mask into (B, 2, H, W) for the dual-head decoder."""
    return torch.cat([mask, mask], dim=1)


# ---------------------------------------------------------------------------
# Training epochs
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, loss_fn,
                    scaler, device, cfg, epoch, world_size):
    model.train()
    totals = {"total": 0.0, "dice": 0.0, "bce": 0.0, "change": 0.0}
    n_steps = 0
    accum = cfg.training.accumulate_grad_batches
    rank = dist.get_rank() if dist.is_initialized() else 0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False, disable=not is_main(rank))

    for step, batch in enumerate(pbar):
        mid_img  = batch["mid_image"].to(device)   # (B, 3, H, W)
        pre_img  = batch["pre_image"].to(device)
        mid_mask = batch["mid_mask"].to(device)    # (B, 1, H, W) binary
        pre_mask = batch["pre_mask"].to(device)
        weeks    = batch["weeks_elapsed"]
        if isinstance(weeks, list):
            weeks = torch.tensor(weeks, dtype=torch.long)
        weeks = weeks.to(device)

        target = _dual_target(mid_mask)            # (B, 2, H, W)

        with autocast(enabled=cfg.training.amp):
            M_pre = unwrap(model).encode_pre_rt(pre_img.unsqueeze(1), pre_mask.unsqueeze(1))
            unwrap(model).reset_mid_session_memory()

            out = model(
                mid_images=mid_img,
                M_pre=M_pre,
                pre_images=pre_img,
                weeks_elapsed=weeks,
                pre_gtvp_mask=pre_mask,
                pre_gtvn_mask=pre_mask,
                return_change_map=True,
                detach_memory=False,
                training_mode=True,
            )

            losses = loss_fn(
                pred_masks=out["masks"],
                target_masks=target,
                change_logits=out.get("change_map"),
                pre_masks=pre_mask,
                mid_masks=mid_mask,
            )
            loss = losses["total"] / accum

        is_sync_step = (step + 1) % accum == 0
        sync_ctx = contextlib.nullcontext() if (is_sync_step or not isinstance(model, nn.parallel.DistributedDataParallel)) else model.no_sync()
        with sync_ctx:
            scaler.scale(loss).backward()

        if is_sync_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                cfg.training.gradient_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if scheduler:
            scheduler.step()

        for k in totals:
            v = losses.get(k, torch.tensor(0.0))
            totals[k] += v.item() if torch.is_tensor(v) else float(v)
        n_steps += 1

        pbar.set_postfix({
            "loss": f"{losses['total'].item():.4f}",
            "dice": f"{float(losses.get('dice', 0)):.4f}",
            "lr":   f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    result = {k: v / max(n_steps, 1) for k, v in totals.items()}
    return {k: reduce_scalar(v, world_size) for k, v in result.items()}


def train_sequence_epoch(model, volume_loader, optimizer, scheduler, loss_fn,
                         scaler, device, cfg, epoch, world_size):
    """Sequence-mode: one full patient volume per step, M_mid accumulates."""
    model.train()
    totals = {"total": 0.0, "dice": 0.0, "bce": 0.0, "change": 0.0}
    n_steps = 0
    accum   = cfg.training.accumulate_grad_batches
    seq_len = max(1, int(cfg.training.sequence_length))
    rank = dist.get_rank() if dist.is_initialized() else 0

    optimizer.zero_grad()
    pbar = tqdm(volume_loader, desc=f"Epoch {epoch} [seq]", leave=False, disable=not is_main(rank))

    for step, batch in enumerate(pbar):
        # BraTS volume dataset: keys are pre_image / mid_image (N, 3, H, W)
        pre_images = batch["pre_image"].squeeze(0).to(device)   # (N, 3, H, W)
        mid_images = batch["mid_image"].squeeze(0).to(device)
        pre_masks  = batch["pre_mask"].squeeze(0).to(device)    # (N, 1, H, W)
        mid_masks  = batch["mid_mask"].squeeze(0).to(device)

        weeks = batch["weeks_elapsed"]
        if isinstance(weeks, (list, tuple)):
            weeks = weeks[0]
        weeks_scalar = int(weeks.item()) if torch.is_tensor(weeks) else int(weeks)

        N = mid_images.shape[0]
        if N == 0:
            continue

        S = min(seq_len, N)
        tumor_mask = (mid_masks.flatten(1).sum(dim=1) > 0).cpu().numpy()
        tumor_idx  = np.where(tumor_mask)[0]
        if len(tumor_idx) > 0:
            center = int(np.random.choice(tumor_idx))
            start  = max(0, min(center - S // 2, N - S))
        else:
            start = random.randint(0, max(0, N - S))
        window = slice(start, start + S)

        is_sync_step = (step + 1) % accum == 0
        sync_ctx = contextlib.nullcontext() if (is_sync_step or not isinstance(model, nn.parallel.DistributedDataParallel)) else model.no_sync()
        with sync_ctx:
            with autocast(enabled=cfg.training.amp):
                M_pre = unwrap(model).encode_pre_rt(
                    pre_images.unsqueeze(0), pre_masks.unsqueeze(0)
                )
                unwrap(model).reset_mid_session_memory()

                seq_loss = None
                seq_comps = {"dice": 0.0, "bce": 0.0, "change": 0.0}
                n_in_seq = 0

                for i in range(window.start, window.stop):
                    weeks_t = torch.tensor([weeks_scalar], dtype=torch.long, device=device)
                    pm = pre_masks[i].unsqueeze(0)
                    mm = mid_masks[i].unsqueeze(0)

                    out = model(
                        mid_images=mid_images[i].unsqueeze(0),
                        M_pre=M_pre,
                        pre_images=pre_images[i].unsqueeze(0),
                        weeks_elapsed=weeks_t,
                        pre_gtvp_mask=pm,
                        pre_gtvn_mask=pm,
                        return_change_map=True,
                        detach_memory=False,
                        training_mode=True,
                    )

                    target = _dual_target(mm)
                    losses = loss_fn(
                        pred_masks=out["masks"],
                        target_masks=target,
                        change_logits=out.get("change_map"),
                        pre_masks=pm,
                        mid_masks=mm,
                    )

                    step_loss = losses["total"]
                    seq_loss = step_loss if seq_loss is None else seq_loss + step_loss
                    for k in seq_comps:
                        v = losses.get(k, None)
                        if v is not None:
                            seq_comps[k] += v.item() if torch.is_tensor(v) else float(v)
                    n_in_seq += 1

                seq_loss = seq_loss / max(n_in_seq, 1)
                loss = seq_loss / accum

            scaler.scale(loss).backward()

        if is_sync_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                cfg.training.gradient_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if scheduler:
            scheduler.step()

        totals["total"] += seq_loss.item()
        for k in ("dice", "bce", "change"):
            totals[k] += seq_comps[k] / max(n_in_seq, 1)
        n_steps += 1

        pbar.set_postfix({
            "loss": f"{seq_loss.item():.4f}",
            "S": S,
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    result = {k: v / max(n_steps, 1) for k, v in totals.items()}
    return {k: reduce_scalar(v, world_size) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, cfg):
    model.eval()
    dsc_list = []
    threshold = cfg.evaluation.threshold

    for batch in tqdm(loader, desc="Validating", leave=False):
        pre_images = batch["pre_image"].squeeze(0).to(device)
        mid_images = batch["mid_image"].squeeze(0).to(device)
        pre_masks  = batch["pre_mask"].squeeze(0).to(device)
        gt_masks   = batch["mid_mask"].squeeze(0)              # (N, 1, H, W) CPU

        weeks = batch["weeks_elapsed"]
        if isinstance(weeks, (list, tuple)):
            weeks = weeks[0]
        weeks_scalar = int(weeks.item()) if torch.is_tensor(weeks) else int(weeks)
        weeks_t = torch.tensor([weeks_scalar], dtype=torch.long, device=device)

        N = pre_images.shape[0]
        M_pre = unwrap(model).encode_pre_rt(
            pre_images.unsqueeze(0), pre_masks.unsqueeze(0)
        )
        unwrap(model).reset_mid_session_memory()

        pred_slices = []
        for i in range(N):
            out = model(
                mid_images=mid_images[i].unsqueeze(0),
                M_pre=M_pre,
                pre_images=pre_images[i].unsqueeze(0),
                weeks_elapsed=weeks_t,
                pre_gtvp_mask=pre_masks[i].unsqueeze(0),
                pre_gtvn_mask=pre_masks[i].unsqueeze(0),
                return_change_map=False,
                detach_memory=True,
                training_mode=False,
            )
            # Use first channel (GTVp analog = tumor)
            prob   = torch.sigmoid(out["masks"][:, 0:1])
            binary = (prob > threshold).squeeze(0).cpu().numpy()  # (1, H, W)
            pred_slices.append(binary[0])

        pred_vol = np.stack(pred_slices)            # (N, H, W)
        gt_vol   = (gt_masks > 0.5).squeeze(1).numpy()  # (N, H, W)

        dsc = compute_dice(pred_vol, gt_vol)
        dsc_list.append(dsc)

    mean_dsc = float(np.mean(dsc_list)) if dsc_list else 0.0
    return {"dsc_mean": mean_dsc, "dsc_std": float(np.std(dsc_list)) if dsc_list else 0.0}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, cfg, is_best=False):
    out_dir = Path(cfg.checkpoint.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": OmegaConf.to_container(cfg),
    }
    torch.save(state, out_dir / "latest.pth")
    if epoch % cfg.checkpoint.save_every_n_epochs == 0:
        torch.save(state, out_dir / f"epoch_{epoch:04d}.pth")
    if is_best:
        torch.save(state, out_dir / "best.pth")
        print(f"  New best saved at epoch {epoch}")
    for old in sorted(out_dir.glob("epoch_*.pth"))[:-cfg.checkpoint.keep_last_n]:
        old.unlink()


def load_checkpoint(model, optimizer, scheduler, path, device):
    print(f"Loading checkpoint: {path}")
    state = torch.load(path, map_location=device)
    unwrap(model).load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and state.get("scheduler_state_dict"):
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return state["epoch"], state.get("metrics", {})


# ---------------------------------------------------------------------------
# Dataloader builders
# ---------------------------------------------------------------------------

def _make_distributed_loader(dataset, cfg, shuffle, world_size, rank):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle) \
        if world_size > 1 else None
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        shuffle=(shuffle and world_size == 1),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )


def _make_volume_loader(dataset, cfg, world_size, rank):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) \
        if world_size > 1 else None
    return DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=(world_size == 1),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )


def build_loaders(cfg, world_size, rank):
    data_dir     = cfg.data.data_dir
    image_size   = cfg.data.image_size
    modality     = getattr(cfg.data, "modality", "t2f")
    val_fraction = float(getattr(cfg.data, "val_fraction", 0.10))

    train_slice_ds = BraTSGLISliceDataset(
        data_dir, split="train", modality=modality,
        image_size=image_size, augment=True,
        tumor_ratio=cfg.data.tumor_ratio,
        val_fraction=val_fraction,
    )
    val_vol_ds = BraTSGLIDataset(
        data_dir, split="val", modality=modality,
        image_size=image_size, val_fraction=val_fraction,
    )
    train_vol_ds = BraTSGLIDataset(
        data_dir, split="train", modality=modality,
        image_size=image_size, val_fraction=val_fraction,
    )

    return {
        "train":         _make_distributed_loader(train_slice_ds, cfg, shuffle=True,  world_size=world_size, rank=rank),
        "train_volumes": _make_volume_loader(train_vol_ds,        cfg, world_size=world_size, rank=rank),
        "val":           DataLoader(val_vol_ds, batch_size=1, shuffle=False, num_workers=cfg.data.num_workers),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train CSM-SAM on BraTS-GLI (8-GPU DDP)")
    parser.add_argument("--config",          type=str,   default="configs/brats.yaml")
    parser.add_argument("--resume",          type=str,   default=None)
    parser.add_argument("--epochs",          type=int,   default=None)
    parser.add_argument("--batch_size",      type=int,   default=None)
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--data_dir",        type=str,   default=None)
    parser.add_argument("--output_dir",      type=str,   default=None)
    parser.add_argument("--sam2_checkpoint", type=str,   default=None)
    parser.add_argument("--no_wandb",        action="store_true")
    parser.add_argument("--sequence_train",  dest="sequence_train", action="store_true",  default=None)
    parser.add_argument("--no_sequence_train", dest="sequence_train", action="store_false")
    parser.add_argument("--sequence_length", type=int, default=None)
    args = parser.parse_args()

    local_rank, rank, world_size = setup_ddp()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    cfg = OmegaConf.load(args.config)
    if args.epochs          is not None: cfg.training.epochs         = args.epochs
    if args.batch_size      is not None: cfg.data.batch_size         = args.batch_size
    if args.lr              is not None: cfg.training.lr             = args.lr
    if args.data_dir        is not None: cfg.data.data_dir           = args.data_dir
    if args.output_dir      is not None: cfg.checkpoint.output_dir  = args.output_dir
    if args.sam2_checkpoint is not None: cfg.model.sam2_checkpoint  = args.sam2_checkpoint
    if args.no_wandb:                    cfg.logging.use_wandb       = False
    if args.sequence_train  is not None: cfg.training.sequence_mode = bool(args.sequence_train)
    if args.sequence_length is not None: cfg.training.sequence_length = int(args.sequence_length)

    if "sequence_mode"   not in cfg.training: cfg.training.sequence_mode   = True
    if "sequence_length" not in cfg.training: cfg.training.sequence_length = 4

    set_seed(cfg.hardware.seed, rank)

    if is_main(rank):
        print("\n" + "=" * 60)
        print(f"CSM-SAM BraTS Training  |  GPUs: {world_size}  |  rank 0/{world_size-1}")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))

    wandb_run = None
    if cfg.logging.use_wandb and is_main(rank):
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.logging.wandb_project,
                config=OmegaConf.to_container(cfg),
            )
        except ImportError:
            pass

    model = CSMSAM.from_pretrained(
        sam2_checkpoint=cfg.model.sam2_checkpoint,
        sam2_cfg=cfg.model.sam2_cfg,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        n_memory_frames=cfg.model.n_memory_frames,
        spatial_pool_size=cfg.model.spatial_pool_size,
        max_weeks=cfg.model.max_weeks,
        dropout=cfg.model.dropout,
        temporal_encoder_type=getattr(cfg.model, "temporal_encoder_type", "continuous"),
        temporal_hidden_dim=getattr(cfg.model,  "temporal_hidden_dim",    128),
        temporal_n_frequencies=getattr(cfg.model, "temporal_n_frequencies", 6),
    ).to(device)

    if is_main(rank):
        counts = unwrap(model).count_trainable_params()
        print("Trainable parameters:")
        for name, count in counts.items():
            print(f"  {name}: {count:,}")

    if world_size > 1:
        torch.cuda.empty_cache()
        # Skip NCCL param-shape verification — it OOMs on large SAM2 model
        _orig_verify = dist._verify_params_across_processes
        dist._verify_params_across_processes = lambda *a, **kw: None
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
            bucket_cap_mb=25,
        )
        dist._verify_params_across_processes = _orig_verify

    if is_main(rank):
        print("\nBuilding BraTS-GLI data loaders...")
    loaders = build_loaders(cfg, world_size, rank)

    sequence_mode = bool(cfg.training.sequence_mode) and int(cfg.training.sequence_length) > 1
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(loaders["train_volumes"] if sequence_mode else loaders["train"])
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)

    loss_fn = CombinedLoss(
        lambda_dice=cfg.loss.lambda_dice,
        lambda_bce=cfg.loss.lambda_bce,
        lambda_change=cfg.loss.lambda_change,
        change_loss_weights=cfg.loss.change_class_weights,
        gtvn_weight=1.0,
    )
    scaler = GradScaler(enabled=cfg.training.amp and device != "cpu")

    start_epoch = 1
    best_metric = 0.0
    if args.resume:
        start_epoch, prev = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        start_epoch += 1
        best_metric = prev.get("dsc_mean", 0.0)

    if is_main(rank):
        eff_bs = cfg.data.batch_size * cfg.training.accumulate_grad_batches * world_size
        print(f"\nStarting training: epochs {start_epoch}-{cfg.training.epochs}")
        print(f"Mode: {'sequence (S=' + str(cfg.training.sequence_length) + ')' if sequence_mode else 'single-slice'}")
        print(f"Effective batch size: {eff_bs}  ({world_size} GPU × {cfg.data.batch_size} × {cfg.training.accumulate_grad_batches} accum)")
        print()

    metrics_history = []

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        if world_size > 1:
            ldr = loaders["train_volumes"] if sequence_mode else loaders["train"]
            if hasattr(ldr.sampler, "set_epoch"):
                ldr.sampler.set_epoch(epoch)

        t0 = time.time()

        if sequence_mode:
            train_metrics = train_sequence_epoch(
                model, loaders["train_volumes"], optimizer, scheduler,
                loss_fn, scaler, device, cfg, epoch, world_size,
            )
        else:
            train_metrics = train_one_epoch(
                model, loaders["train"], optimizer, scheduler,
                loss_fn, scaler, device, cfg, epoch, world_size,
            )

        val_metrics = {}
        is_best = False
        if is_main(rank) and epoch % cfg.evaluation.val_every_n_epochs == 0:
            val_metrics = validate(model, loaders["val"], device, cfg)
            current = val_metrics.get("dsc_mean", 0.0)
            is_best = current > best_metric
            if is_best:
                best_metric = current

        if is_main(rank):
            elapsed = time.time() - t0
            log_str = (
                f"Epoch {epoch:03d}/{cfg.training.epochs} | "
                f"loss={train_metrics['total']:.4f}  dice={train_metrics['dice']:.4f}"
            )
            if val_metrics:
                log_str += f" | val_DSC={val_metrics['dsc_mean']:.4f} (best={best_metric:.4f})"
            log_str += f" | {elapsed:.1f}s"
            print(log_str)

            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, cfg, is_best)

            if wandb_run:
                log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
                log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
                log_dict["epoch"] = epoch
                wandb_run.log(log_dict)

            metrics_history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if world_size > 1:
            dist.barrier()

    if is_main(rank):
        out_dir = Path(cfg.checkpoint.output_dir)
        with open(out_dir / "training_history.json", "w") as f:
            json.dump(metrics_history, f, indent=2)
        print(f"\nDone. Best DSC: {best_metric:.4f}")
        print(f"Best model: {out_dir}/best.pth")
        if wandb_run:
            wandb_run.finish()

    cleanup_ddp()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        rank = int(os.environ.get("RANK", 0))
        log_path = f"/home/tal/natalie/tmp/rank_{rank}_error.log"
        with open(log_path, "w") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        raise
