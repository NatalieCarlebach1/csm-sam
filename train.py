"""
CSM-SAM Training Script — supports single-GPU and multi-GPU (DDP).

Single GPU:
    python train.py --config configs/default.yaml

Multi-GPU (8 GPUs, recommended):
    torchrun --nproc_per_node=8 train.py --config configs/default.yaml

With sequence training + k-fold:
    torchrun --nproc_per_node=8 train.py --config configs/default.yaml \\
        --sequence_train --fold 0 --n_folds 5
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
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

from csmsam.datasets import HNTSMRGDataset, HNTSMRGSliceDataset, build_dataloaders
from csmsam.losses import CombinedLoss
from csmsam.modeling import CSMSAM
from csmsam.utils.metrics import aggregate_metrics, compute_agg_dsc


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int]:
    """Initialise NCCL process group if torchrun env vars are set.

    Returns (local_rank, rank, world_size).
    Falls back to (0, 0, 1) for single-GPU runs.
    """
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


def is_main(rank: int) -> bool:
    return rank == 0


def unwrap(model: nn.Module) -> CSMSAM:
    """Strip DDP wrapper to access CSMSAM methods directly."""
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def reduce_scalar(value: float, world_size: int) -> float:
    """All-reduce a scalar (mean) across all ranks."""
    if world_size == 1:
        return value
    t = torch.tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / world_size).item()


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


# ---------------------------------------------------------------------------
# Optimiser / scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    param_groups = unwrap(model).get_trainable_params()
    base_lr = cfg.training.lr
    optimizer_groups = []
    for group in param_groups:
        lr = base_lr * group.get("lr_scale", 1.0)
        optimizer_groups.append({
            "params": [p for p in group["params"] if p.requires_grad],
            "lr": lr,
        })
    optimizer_groups = [g for g in optimizer_groups if len(g["params"]) > 0]
    return AdamW(optimizer_groups, weight_decay=cfg.training.weight_decay)


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    warmup_steps = cfg.training.warmup_epochs * steps_per_epoch
    total_steps  = cfg.training.epochs * steps_per_epoch
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


# ---------------------------------------------------------------------------
# KL beta schedule
# ---------------------------------------------------------------------------

def _get_kl_beta(epoch: int, cfg) -> float:
    cl = getattr(cfg, "change_latent", None)
    if cl is None or not getattr(cl, "enabled", False):
        return 0.0
    beta_max   = float(getattr(cl, "lambda_kl",        0.1))
    warmup_eps = int(getattr(cl,   "kl_warmup_epochs", 10))
    if warmup_eps <= 0:
        return beta_max
    return beta_max * min(1.0, epoch / warmup_eps)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _weighted_2ch_target(gtvp: torch.Tensor, gtvn: torch.Tensor) -> torch.Tensor:
    return torch.cat([gtvp, gtvn], dim=1)


def _apply_channel_loss(loss_fn, pred_masks, target_masks,
                        change_logits, pre_masks, mid_masks, gtvn_weight,
                        kl_loss=None, kl_beta_val: float = 0.0):
    del gtvn_weight  # handled inside CombinedLoss
    return loss_fn(
        pred_masks=pred_masks,
        target_masks=target_masks,
        change_logits=change_logits,
        pre_masks=pre_masks,
        mid_masks=mid_masks,
        kl_loss=kl_loss,
        kl_beta=kl_beta_val,
    )


# ---------------------------------------------------------------------------
# Training epochs
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, loss_fn,
                    scaler, device, cfg, epoch, world_size) -> dict:
    """Single-slice training (fallback when sequence_mode=false)."""
    model.train()
    total_losses = {"total": 0.0, "dice": 0.0, "bce": 0.0, "change": 0.0, "kl": 0.0}
    n_steps = 0
    accumulate  = cfg.training.accumulate_grad_batches
    gtvn_weight = getattr(cfg.loss, "gtvn_weight", 1.0)
    kl_beta_val = _get_kl_beta(epoch, cfg)
    rank = dist.get_rank() if dist.is_initialized() else 0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False, disable=not is_main(rank))

    for step, batch in enumerate(pbar):
        mid_images     = batch["mid_image"].to(device)
        pre_images     = batch["pre_image"].to(device)
        mid_masks      = batch["mid_mask"].to(device)
        pre_masks      = batch["pre_mask"].to(device)
        mid_masks_gtvp = batch["mid_mask_gtvp"].to(device)
        mid_masks_gtvn = batch["mid_mask_gtvn"].to(device)
        pre_masks_gtvp = batch["pre_mask_gtvp"].to(device)
        pre_masks_gtvn = batch["pre_mask_gtvn"].to(device)
        weeks = batch["weeks_elapsed"]
        if isinstance(weeks, list):
            weeks = torch.tensor(weeks, dtype=torch.long)
        weeks = weeks.to(device)

        target_masks = _weighted_2ch_target(mid_masks_gtvp, mid_masks_gtvn)

        with autocast(enabled=cfg.training.amp):
            M_pre = unwrap(model).encode_pre_rt(
                pre_images.unsqueeze(1), pre_masks.unsqueeze(1)
            )
            unwrap(model).reset_mid_session_memory()

            out = model(
                mid_images=mid_images,
                M_pre=M_pre,
                pre_images=pre_images,
                weeks_elapsed=weeks,
                pre_gtvp_mask=pre_masks_gtvp,
                pre_gtvn_mask=pre_masks_gtvn,
                return_change_map=True,
                detach_memory=False,
                training_mode=True,
            )

            kl = out.get("kl_loss")
            losses = _apply_channel_loss(
                loss_fn=loss_fn,
                pred_masks=out["masks"],
                target_masks=target_masks,
                change_logits=out.get("change_map"),
                pre_masks=pre_masks,
                mid_masks=mid_masks,
                gtvn_weight=gtvn_weight,
                kl_loss=kl,
                kl_beta_val=kl_beta_val,
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
            v = losses.get(k, torch.tensor(0.0))
            total_losses[k] += v.item() if torch.is_tensor(v) else float(v)
        n_steps += 1

        pbar.set_postfix({
            "loss": f"{losses['total'].item():.4f}",
            "dice": f"{float(losses.get('dice', 0)):.4f}",
            "lr":   f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    result = {k: v / max(n_steps, 1) for k, v in total_losses.items()}
    return {k: reduce_scalar(v, world_size) for k, v in result.items()}


def train_sequence_epoch(model, volume_loader, optimizer, scheduler, loss_fn,
                         scaler, device, cfg, epoch, world_size) -> dict:
    """Sequence-mode: accumulate M_mid across consecutive mid-RT slices per patient."""
    model.train()
    total_losses = {"total": 0.0, "dice": 0.0, "bce": 0.0, "change": 0.0, "kl": 0.0}
    n_steps = 0
    accumulate  = cfg.training.accumulate_grad_batches
    seq_len     = max(1, int(cfg.training.sequence_length))
    gtvn_weight = getattr(cfg.loss, "gtvn_weight", 1.0)
    kl_beta_val = _get_kl_beta(epoch, cfg)
    rank = dist.get_rank() if dist.is_initialized() else 0

    optimizer.zero_grad()
    pbar = tqdm(volume_loader, desc=f"Epoch {epoch} [seq]", leave=False, disable=not is_main(rank))

    for step, batch in enumerate(pbar):
        pre_images         = batch["pre_images"].squeeze(0).to(device)
        mid_images         = batch["mid_images"].squeeze(0).to(device)
        pre_masks_combined = batch["pre_masks"].squeeze(0).to(device)
        mid_masks_combined = batch["mid_masks"].squeeze(0).to(device)
        pre_gtvp = batch["pre_masks_gtvp"].squeeze(0).to(device)
        pre_gtvn = batch["pre_masks_gtvn"].squeeze(0).to(device)
        mid_gtvp = batch["mid_masks_gtvp"].squeeze(0).to(device)
        mid_gtvn = batch["mid_masks_gtvn"].squeeze(0).to(device)

        weeks = batch["weeks_elapsed"]
        if isinstance(weeks, (list, tuple)):
            weeks = weeks[0]
        weeks_scalar = int(weeks.item()) if torch.is_tensor(weeks) else int(weeks)

        N = mid_images.shape[0]
        if N == 0:
            continue

        S = min(seq_len, N)
        tumor_slice_mask = (mid_masks_combined.flatten(1).sum(dim=1) > 0).cpu().numpy()
        tumor_indices = np.where(tumor_slice_mask)[0]
        if len(tumor_indices) > 0:
            center = int(np.random.choice(tumor_indices))
            start  = max(0, min(center - S // 2, N - S))
        else:
            start = random.randint(0, max(0, N - S))
        window = slice(start, start + S)

        with autocast(enabled=cfg.training.amp):
            M_pre = unwrap(model).encode_pre_rt(
                pre_images.unsqueeze(0),
                pre_masks_combined.unsqueeze(0),
            )
            unwrap(model).reset_mid_session_memory()

            seq_loss = None
            seq_loss_components = {"dice": 0.0, "bce": 0.0, "change": 0.0, "kl": 0.0}
            n_in_seq = 0

            for i in range(window.start, window.stop):
                weeks_t = torch.tensor([weeks_scalar], dtype=torch.long, device=device)

                out = model(
                    mid_images=mid_images[i].unsqueeze(0),
                    M_pre=M_pre,
                    pre_images=pre_images[i].unsqueeze(0),
                    weeks_elapsed=weeks_t,
                    pre_gtvp_mask=pre_gtvp[i].unsqueeze(0),
                    pre_gtvn_mask=pre_gtvn[i].unsqueeze(0),
                    return_change_map=True,
                    detach_memory=False,
                    training_mode=True,
                )

                kl = out.get("kl_loss")
                target_masks = _weighted_2ch_target(mid_gtvp[i].unsqueeze(0), mid_gtvn[i].unsqueeze(0))
                losses = _apply_channel_loss(
                    loss_fn=loss_fn,
                    pred_masks=out["masks"],
                    target_masks=target_masks,
                    change_logits=out.get("change_map"),
                    pre_masks=pre_masks_combined[i].unsqueeze(0),
                    mid_masks=mid_masks_combined[i].unsqueeze(0),
                    gtvn_weight=gtvn_weight,
                    kl_loss=kl,
                    kl_beta_val=kl_beta_val,
                )

                step_loss = losses["total"]
                seq_loss  = step_loss if seq_loss is None else seq_loss + step_loss
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

    result = {k: v / max(n_steps, 1) for k, v in total_losses.items()}
    return {k: reduce_scalar(v, world_size) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Validation  (runs on rank-0 only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, cfg) -> dict:
    model.eval()
    patient_metrics = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        pre_images     = batch["pre_images"].squeeze(0).to(device)
        mid_images     = batch["mid_images"].squeeze(0).to(device)
        pre_masks      = batch["pre_masks"].squeeze(0).to(device)
        pre_gtvp       = batch["pre_masks_gtvp"].squeeze(0).to(device)
        pre_gtvn       = batch["pre_masks_gtvn"].squeeze(0).to(device)
        mid_masks_gtvp = batch["mid_masks_gtvp"].squeeze(0)
        mid_masks_gtvn = batch["mid_masks_gtvn"].squeeze(0)

        weeks = batch["weeks_elapsed"]
        if isinstance(weeks, (list, tuple)):
            weeks = weeks[0]
        weeks_scalar = int(weeks.item()) if torch.is_tensor(weeks) else int(weeks)
        weeks_t = torch.tensor([weeks_scalar], dtype=torch.long, device=device)

        N = pre_images.shape[0]
        M_pre = unwrap(model).encode_pre_rt(
            pre_images.unsqueeze(0),
            pre_masks.unsqueeze(0),
        )
        unwrap(model).reset_mid_session_memory()

        pred_gtvp_slices, pred_gtvn_slices = [], []
        threshold = cfg.evaluation.threshold

        for i in range(N):
            out = model(
                mid_images=mid_images[i].unsqueeze(0),
                M_pre=M_pre,
                pre_images=pre_images[i].unsqueeze(0),
                weeks_elapsed=weeks_t,
                pre_gtvp_mask=pre_gtvp[i].unsqueeze(0),
                pre_gtvn_mask=pre_gtvn[i].unsqueeze(0),
                return_change_map=False,
                detach_memory=True,
                training_mode=False,
            )
            probs  = torch.sigmoid(out["masks"])
            binary = (probs > threshold).squeeze(0).cpu().numpy()
            pred_gtvp_slices.append(binary[0])
            pred_gtvn_slices.append(binary[1])

        pred_gtvp = np.stack(pred_gtvp_slices)
        pred_gtvn = np.stack(pred_gtvn_slices)
        gt_gtvp   = (mid_masks_gtvp > 0.5).squeeze(1).numpy()
        gt_gtvn   = (mid_masks_gtvn > 0.5).squeeze(1).numpy()

        patient_metrics.append(compute_agg_dsc(pred_gtvp, pred_gtvn, gt_gtvp, gt_gtvn))

    return aggregate_metrics(patient_metrics)


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
        print(f"  New best model saved at epoch {epoch}")

    periodic = sorted(out_dir.glob("epoch_*.pth"))
    for old in periodic[:-cfg.checkpoint.keep_last_n]:
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
# DDP-aware dataloader builders
# ---------------------------------------------------------------------------

def _make_distributed_loader(dataset, cfg, shuffle, world_size, rank) -> DataLoader:
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


def _make_volume_loader(dataset, cfg, world_size, rank) -> DataLoader:
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


# ---------------------------------------------------------------------------
# K-fold
# ---------------------------------------------------------------------------

def _apply_kfold(cfg, fold, n_folds, seed, world_size, rank):
    try:
        from csmsam.utils.cv import kfold_split
    except Exception as e:
        print(f"  [warn] csmsam.utils.cv unavailable ({e}); falling back.")
        return None

    image_size = cfg.data.image_size
    train_slice_ds = HNTSMRGSliceDataset(cfg.data.data_dir, split="train", image_size=image_size, augment=True)
    train_vol_ds   = HNTSMRGDataset(cfg.data.data_dir, split="train", image_size=image_size)
    val_vol_ds     = HNTSMRGDataset(cfg.data.data_dir, split="train", image_size=image_size)

    try:
        patient_ids = [d.name for d in train_vol_ds.patient_dirs]
        train_ids, val_ids = kfold_split(patient_ids, fold=fold, n_folds=n_folds, seed=seed)
        train_set = set(train_ids)

        pid_to_idx    = {d.name: i for i, d in enumerate(train_vol_ds.patient_dirs)}
        train_vol_idx = [pid_to_idx[p] for p in train_ids if p in pid_to_idx]
        val_vol_idx   = [pid_to_idx[p] for p in val_ids   if p in pid_to_idx]

        train_slice_idx = [
            i for i, (pdir, _) in enumerate(train_slice_ds.tumor_slices + train_slice_ds.bg_slices)
            if pdir.name in train_set
        ]
    except Exception as e:
        print(f"  [warn] kfold_split failed ({e}); falling back.")
        return None

    train_slice_sub = Subset(train_slice_ds, train_slice_idx)
    train_vol_sub   = Subset(train_vol_ds,   train_vol_idx)
    val_vol_sub     = Subset(val_vol_ds,     val_vol_idx)

    return {
        "train":         _make_distributed_loader(train_slice_sub, cfg, shuffle=True,  world_size=world_size, rank=rank),
        "train_volumes": _make_volume_loader(train_vol_sub, cfg, world_size=world_size, rank=rank),
        "val":           DataLoader(val_vol_sub, batch_size=1, shuffle=False, num_workers=cfg.data.num_workers),
    }


def _build_train_volume_loader(cfg, world_size, rank) -> DataLoader:
    ds = HNTSMRGDataset(cfg.data.data_dir, split="train", image_size=cfg.data.image_size)
    return _make_volume_loader(ds, cfg, world_size=world_size, rank=rank)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train CSM-SAM (single or multi-GPU via torchrun)")
    parser.add_argument("--config",            type=str,   default="configs/default.yaml")
    parser.add_argument("--resume",            type=str,   default=None)
    parser.add_argument("--epochs",            type=int,   default=None)
    parser.add_argument("--batch_size",        type=int,   default=None)
    parser.add_argument("--lr",                type=float, default=None)
    parser.add_argument("--data_dir",          type=str,   default=None)
    parser.add_argument("--output_dir",        type=str,   default=None)
    parser.add_argument("--sam2_checkpoint",   type=str,   default=None)
    parser.add_argument("--no_wandb",          action="store_true")
    parser.add_argument("--sequence_train",    dest="sequence_train", action="store_true",  default=None)
    parser.add_argument("--no_sequence_train", dest="sequence_train", action="store_false")
    parser.add_argument("--sequence_length",   type=int,   default=None)
    parser.add_argument("--fold",              type=int,   default=None)
    parser.add_argument("--n_folds",           type=int,   default=5)
    args = parser.parse_args()

    # ── DDP ───────────────────────────────────────────────────────────────
    local_rank, rank, world_size = setup_ddp()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # ── Config ────────────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)

    if args.epochs          is not None: cfg.training.epochs          = args.epochs
    if args.batch_size      is not None: cfg.data.batch_size          = args.batch_size
    if args.lr              is not None: cfg.training.lr              = args.lr
    if args.data_dir        is not None: cfg.data.data_dir            = args.data_dir
    if args.output_dir      is not None: cfg.checkpoint.output_dir   = args.output_dir
    if args.sam2_checkpoint is not None: cfg.model.sam2_checkpoint   = args.sam2_checkpoint
    if args.no_wandb:                    cfg.logging.use_wandb        = False
    if args.sequence_train  is not None: cfg.training.sequence_mode  = bool(args.sequence_train)
    if args.sequence_length is not None: cfg.training.sequence_length = int(args.sequence_length)

    if "sequence_mode"   not in cfg.training: cfg.training.sequence_mode   = True
    if "sequence_length" not in cfg.training: cfg.training.sequence_length = 4
    if "gtvn_weight"     not in cfg.loss:     cfg.loss.gtvn_weight         = 1.0

    set_seed(cfg.hardware.seed, rank)

    if is_main(rank):
        print("\n" + "=" * 60)
        print(f"CSM-SAM Training  |  GPUs: {world_size}  |  rank 0/{world_size-1}")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))

    # ── W&B ───────────────────────────────────────────────────────────────
    wandb_run = None
    if cfg.logging.use_wandb and is_main(rank):
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.wandb_entity,
                config=OmegaConf.to_container(cfg),
            )
        except ImportError:
            print("wandb not installed, skipping")

    # ── Model ─────────────────────────────────────────────────────────────
    cl_cfg = getattr(cfg, "change_latent", None)
    use_change_latent = bool(getattr(cl_cfg, "enabled",       False)) if cl_cfg else False
    d_z               = int(getattr(cl_cfg,  "d_z",           64))    if cl_cfg else 64
    latent_alpha      = float(getattr(cl_cfg, "alpha",        0.5))   if cl_cfg else 0.5
    retrieval_tau     = float(getattr(cl_cfg, "retrieval_tau", 0.1))  if cl_cfg else 0.1

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
        temporal_hidden_dim=getattr(cfg.model, "temporal_hidden_dim", 128),
        temporal_n_frequencies=getattr(cfg.model, "temporal_n_frequencies", 6),
        use_change_latent=use_change_latent,
        d_z=d_z,
        latent_alpha=latent_alpha,
        latent_retrieval_tau=retrieval_tau,
    ).to(device)

    if is_main(rank):
        counts = unwrap(model).count_trainable_params()
        print("Trainable parameters:")
        for name, count in counts.items():
            print(f"  {name}: {count:,}")

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # ── Data ──────────────────────────────────────────────────────────────
    if is_main(rank):
        print("\nBuilding data loaders...")

    loaders = None
    if args.fold is not None:
        if is_main(rank):
            print(f"  k-fold: fold={args.fold}/{args.n_folds}")
        loaders = _apply_kfold(cfg, args.fold, args.n_folds, cfg.hardware.seed, world_size, rank)

    if loaders is None:
        raw = build_dataloaders(
            data_dir=cfg.data.data_dir,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            image_size=cfg.data.image_size,
            pin_memory=cfg.data.pin_memory,
        )
        if world_size > 1:
            loaders = {
                "train": _make_distributed_loader(raw["train"].dataset, cfg, shuffle=True, world_size=world_size, rank=rank),
                "val":   raw["val"],
            }
            if "train_volumes" in raw:
                loaders["train_volumes"] = _make_volume_loader(raw["train_volumes"].dataset, cfg, world_size=world_size, rank=rank)
        else:
            loaders = raw

    sequence_mode = bool(cfg.training.sequence_mode) and int(cfg.training.sequence_length) > 1
    if sequence_mode and "train_volumes" not in loaders:
        if is_main(rank):
            print("  Building volume loader for sequence training...")
        loaders["train_volumes"] = _build_train_volume_loader(cfg, world_size, rank)

    # ── Optimiser / scheduler / loss ──────────────────────────────────────
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(loaders["train_volumes"] if sequence_mode else loaders["train"])
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)

    loss_fn = CombinedLoss(
        lambda_dice=cfg.loss.lambda_dice,
        lambda_bce=cfg.loss.lambda_bce,
        lambda_change=cfg.loss.lambda_change,
        change_loss_weights=cfg.loss.change_class_weights,
        gtvn_weight=getattr(cfg.loss, "gtvn_weight", 1.0),
        lambda_kl=float(getattr(cl_cfg, "lambda_kl", 0.0)) if cl_cfg else 0.0,
    )
    scaler = GradScaler(enabled=cfg.training.amp and device != "cpu")

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    best_metric = 0.0
    if args.resume:
        start_epoch, prev = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        start_epoch += 1
        best_metric = prev.get(f"{cfg.evaluation.primary_metric}_mean", 0.0)

    if is_main(rank):
        eff_bs = cfg.data.batch_size * cfg.training.accumulate_grad_batches * world_size
        print(f"\nStarting training: epochs {start_epoch}-{cfg.training.epochs}")
        print(f"Mode: {'sequence (S=' + str(cfg.training.sequence_length) + ')' if sequence_mode else 'single-slice'}")
        print(f"Steps per epoch (per GPU): {steps_per_epoch}")
        print(f"Effective batch size: {eff_bs}  ({world_size} GPU × {cfg.data.batch_size} × {cfg.training.accumulate_grad_batches} accum)")
        print()

    # ── Training loop ─────────────────────────────────────────────────────
    metrics_history = []

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        if world_size > 1:
            loader_for_epoch = loaders["train_volumes"] if sequence_mode else loaders["train"]
            if hasattr(loader_for_epoch.sampler, "set_epoch"):
                loader_for_epoch.sampler.set_epoch(epoch)

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
            current = val_metrics.get(f"{cfg.evaluation.primary_metric}_mean", 0.0)
            is_best = current > best_metric
            if is_best:
                best_metric = current

        if is_main(rank):
            elapsed = time.time() - t0
            log_str = (
                f"Epoch {epoch:03d}/{cfg.training.epochs} | "
                f"loss={train_metrics['total']:.4f}  dice={train_metrics['dice']:.4f}"
            )
            if train_metrics.get("kl", 0) > 0:
                log_str += f"  kl={train_metrics['kl']:.4f}"
            if val_metrics:
                agg = val_metrics.get("agg_dsc_mean", 0)
                log_str += f" | val_aggDSC={agg:.4f} (best={best_metric:.4f})"
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

    # ── Finish ────────────────────────────────────────────────────────────
    if is_main(rank):
        out_dir = Path(cfg.checkpoint.output_dir)
        with open(out_dir / "training_history.json", "w") as f:
            json.dump(metrics_history, f, indent=2)

        print(f"\nTraining complete. Best {cfg.evaluation.primary_metric}: {best_metric:.4f}")
        print(f"Best model: {out_dir}/best.pth")
        print(f"\nNext step: python test.py --checkpoint {out_dir}/best.pth")

        if wandb_run:
            wandb_run.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
