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
torch.backends.cudnn.enabled = False
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
from csmsam.utils.metrics import compute_dice, compute_hd95


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
# BraTS multi-class loss helpers
# ---------------------------------------------------------------------------

# Per-class loss weights: [NETC, SNFH, ET, RC]
BRATS_CLASS_WEIGHTS = [1.0, 1.0, 1.5, 0.5]


def _multiclass_loss(pred_masks, target_classes, class_weights=None):
    """
    Weighted sum of per-class (Dice + BCE) losses.

    Args:
        pred_masks    : (B, 4, H, W) logits
        target_classes: (B, 4, H, W) float32 binary targets per class
        class_weights : list of 4 floats
    """
    if class_weights is None:
        class_weights = BRATS_CLASS_WEIGHTS

    total_loss = pred_masks.new_zeros(1)[0]
    total_dice = pred_masks.new_zeros(1)[0]
    total_bce  = pred_masks.new_zeros(1)[0]

    for k, w in enumerate(class_weights):
        pred_k = pred_masks[:, k:k+1]
        tgt_k  = target_classes[:, k:k+1].float()

        prob_k = torch.sigmoid(pred_k)
        inter  = (prob_k * tgt_k).sum(dim=(1, 2, 3))
        union  = prob_k.sum(dim=(1, 2, 3)) + tgt_k.sum(dim=(1, 2, 3)) + 1e-6
        dice_k = 1.0 - (2.0 * inter / union).mean()
        bce_k  = torch.nn.functional.binary_cross_entropy_with_logits(pred_k, tgt_k)

        total_dice  = total_dice  + w * dice_k
        total_bce   = total_bce   + w * bce_k
        total_loss  = total_loss  + w * (dice_k + bce_k)

    return {"total": total_loss, "dice": total_dice, "bce": total_bce}


# ---------------------------------------------------------------------------
# Training epochs
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, loss_fn,
                    scaler, device, cfg, epoch, world_size):
    model.train()
    totals = {"total": 0.0, "dice": 0.0, "bce": 0.0, "change": 0.0}
    n_steps = 0
    accum       = cfg.training.accumulate_grad_batches
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

        pre_class_masks = batch.get("pre_mask_classes")
        mid_class_masks = batch.get("mid_mask_classes")
        if pre_class_masks is not None:
            pre_class_masks = pre_class_masks.to(device)   # (B, 4, H, W)
        if mid_class_masks is not None:
            mid_class_masks = mid_class_masks.to(device)   # (B, 4, H, W)

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
                pre_class_masks=pre_class_masks,
            )

            if mid_class_masks is not None and out["masks"].shape[1] == 4:
                mc_losses = _multiclass_loss(out["masks"], mid_class_masks)
                losses = {
                    "total": mc_losses["total"],
                    "dice":  mc_losses["dice"],
                    "bce":   mc_losses["bce"],
                    "change": torch.tensor(0.0, device=device),
                }
            else:
                target = torch.cat([mid_mask, mid_mask], dim=1)
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
    """Sequence-mode: one full patient volume per step, M_mid accumulates.

    True TBPTT: one backward() per sequence step so the autograd graph never
    spans more than one slice.  Gradients accumulate in .grad buffers across
    the sequence exactly like they would with a single accumulated loss, but
    no parameter tensor is held live across multiple forward passes.
    """
    model.train()
    for _m in model.modules():
        if isinstance(_m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            _m.eval()
    totals = {"total": 0.0, "dice": 0.0, "bce": 0.0, "change": 0.0}
    n_steps = 0
    accum       = cfg.training.accumulate_grad_batches
    seq_len     = max(1, int(cfg.training.sequence_length))
    rank = dist.get_rank() if dist.is_initialized() else 0
    _grad_norms_last = {}

    optimizer.zero_grad()
    pbar = tqdm(volume_loader, desc=f"Epoch {epoch} [seq]", leave=False, disable=not is_main(rank))

    for step, batch in enumerate(pbar):
        pre_images = batch["pre_image"].squeeze(0).to(device)
        mid_images = batch["mid_image"].squeeze(0).to(device)
        pre_masks  = batch["pre_mask"].squeeze(0).to(device)
        mid_masks  = batch["mid_mask"].squeeze(0).to(device)

        weeks = batch["weeks_elapsed"]
        if isinstance(weeks, (list, tuple)):
            weeks = weeks[0]
        weeks_scalar = int(weeks.item()) if torch.is_tensor(weeks) else int(weeks)

        pre_class_masks_vol = batch.get("pre_mask_classes")
        mid_class_masks_vol = batch.get("mid_mask_classes")
        if pre_class_masks_vol is not None:
            pre_class_masks_vol = pre_class_masks_vol.squeeze(0).to(device)
        if mid_class_masks_vol is not None:
            mid_class_masks_vol = mid_class_masks_vol.squeeze(0).to(device)

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
        n_steps_in_window = window.stop - window.start

        is_sync_step = (step + 1) % accum == 0

        # Encode pre-RT memory (no-grad; SAM2 encoder frozen)
        with autocast(enabled=cfg.training.amp):
            M_pre = unwrap(model).encode_pre_rt(
                pre_images.unsqueeze(0), pre_masks.unsqueeze(0)
            )
        unwrap(model).reset_mid_session_memory()

        seq_total_loss_val = 0.0
        seq_comps = {"dice": 0.0, "bce": 0.0, "change": 0.0}
        n_in_seq = 0

        for local_i, i in enumerate(range(window.start, window.stop)):
            weeks_t  = torch.tensor([weeks_scalar], dtype=torch.long, device=device)
            pm       = pre_masks[i].unsqueeze(0)
            mm       = mid_masks[i].unsqueeze(0)
            pm_cls   = pre_class_masks_vol[i].unsqueeze(0) if pre_class_masks_vol is not None else None
            mm_cls   = mid_class_masks_vol[i].unsqueeze(0) if mid_class_masks_vol is not None else None

            # DDP: only sync gradients on the last backward of the last accum step
            is_last  = (local_i == n_steps_in_window - 1)
            do_sync  = is_last and is_sync_step
            inner_ctx = (
                contextlib.nullcontext()
                if (do_sync or not isinstance(model, nn.parallel.DistributedDataParallel))
                else model.no_sync()
            )

            with inner_ctx:
                with autocast(enabled=cfg.training.amp):
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
                        pre_class_masks=pm_cls,
                    )

                    if mm_cls is not None and out["masks"].shape[1] == 4:
                        mc_losses = _multiclass_loss(out["masks"], mm_cls)
                        losses = {
                            "total":  mc_losses["total"],
                            "dice":   mc_losses["dice"],
                            "bce":    mc_losses["bce"],
                            "change": torch.tensor(0.0, device=device),
                        }
                    else:
                        target = torch.cat([mm, mm], dim=1)
                        losses = loss_fn(
                            pred_masks=out["masks"],
                            target_masks=target,
                            change_logits=out.get("change_map"),
                            pre_masks=pm,
                            mid_masks=mm,
                        )

                    # Divide by sequence length and accum so gradients accumulate correctly
                    step_loss = losses["total"] / n_steps_in_window / accum

                scaler.scale(step_loss).backward(retain_graph=not is_last)

            # Detach _M_mid after each backward to prevent gradient double-counting
            # through within-session memory. retain_graph=not is_last is still
            # required to keep M_pre's graph alive for subsequent slice forwards.
            _m = unwrap(model)
            if _m._M_mid is not None:
                _m._M_mid = _m._M_mid.detach()

            seq_total_loss_val += losses["total"].item() / n_steps_in_window
            for k in seq_comps:
                v = losses.get(k, None)
                if v is not None:
                    seq_comps[k] += v.item() if torch.is_tensor(v) else float(v)
            n_in_seq += 1

        if is_sync_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                cfg.training.gradient_clip_norm,
            )
            for name, module in unwrap(model).named_children():
                norm = sum(p.grad.detach().norm().item() ** 2
                           for p in module.parameters() if p.grad is not None) ** 0.5
                if norm > 0:
                    _grad_norms_last[name] = norm
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if scheduler:
            scheduler.step()

        totals["total"] += seq_total_loss_val
        for k in ("dice", "bce", "change"):
            totals[k] += seq_comps[k] / max(n_in_seq, 1)
        n_steps += 1

        pbar.set_postfix({
            "loss": f"{seq_total_loss_val:.4f}",
            "S": n_steps_in_window,
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    result = {k: v / max(n_steps, 1) for k, v in totals.items()}
    result["_grad_norms"] = _grad_norms_last
    return {k: reduce_scalar(v, world_size) if k != "_grad_norms" else v
            for k, v in result.items()}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, cfg):
    """
    Full-volume BraTS validation: WT/TC/ET Dice.
      WT = NETC | SNFH | ET  (channels 0,1,2)
      TC = NETC | ET          (channels 0,2)
      ET = ET                 (channel 2)
    Primary metric: dsc_mean = (WT + TC + ET) / 3
    """
    model.eval()
    threshold = cfg.evaluation.threshold
    voxel_spacing = tuple(cfg.evaluation.voxel_spacing)
    dsc_wt_list, dsc_tc_list, dsc_et_list, hd95_list = [], [], [], []

    for batch in tqdm(loader, desc="Validating", leave=False):
        pre_images = batch["pre_image"].squeeze(0).to(device)
        mid_images = batch["mid_image"].squeeze(0).to(device)
        pre_masks  = batch["pre_mask"].squeeze(0).to(device)

        have_cls = "mid_mask_classes" in batch
        if have_cls:
            gt_cls = batch["mid_mask_classes"].squeeze(0).numpy()  # (N, 4, H, W)
        else:
            gt_bin = batch["mid_mask"].squeeze(0).numpy()          # (N, 1, H, W)

        pre_cls_vol = None
        if "pre_mask_classes" in batch:
            pre_cls_vol = batch["pre_mask_classes"].squeeze(0).to(device)  # (N, 4, H, W)

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
            pm_cls = pre_cls_vol[i].unsqueeze(0) if pre_cls_vol is not None else None
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
                pre_class_masks=pm_cls,
            )
            prob = torch.sigmoid(out["masks"])  # (1, C, H, W)
            pred_slices.append((prob > threshold).squeeze(0).cpu().numpy())  # (C, H, W)

        pred_vol = np.stack(pred_slices)  # (N, C, H, W)

        if pred_vol.shape[1] == 4 and have_cls:
            pred_wt = (pred_vol[:, 0] | pred_vol[:, 1] | pred_vol[:, 2]).astype(float)
            gt_wt   = ((gt_cls[:, 0] | gt_cls[:, 1] | gt_cls[:, 2]) > 0.5).astype(float)
            pred_tc = (pred_vol[:, 0] | pred_vol[:, 2]).astype(float)
            gt_tc   = ((gt_cls[:, 0] | gt_cls[:, 2]) > 0.5).astype(float)
            pred_et = pred_vol[:, 2].astype(float)
            gt_et   = (gt_cls[:, 2] > 0.5).astype(float)
        else:
            pred_bin = pred_vol[:, 0] if pred_vol.ndim == 4 else pred_vol
            pred_wt = pred_tc = pred_et = pred_bin.astype(float)
            if have_cls:
                gt_all = ((gt_cls[:, 0] | gt_cls[:, 1] | gt_cls[:, 2]) > 0.5).astype(float)
                gt_wt = gt_tc = gt_et = gt_all
            else:
                gt_wt = gt_tc = gt_et = (gt_bin.squeeze(1) > 0.5).astype(float)

        dsc_wt_list.append(compute_dice(pred_wt, gt_wt))
        dsc_tc_list.append(compute_dice(pred_tc, gt_tc))
        dsc_et_list.append(compute_dice(pred_et, gt_et))
        hd95_list.append(compute_hd95(pred_wt, gt_wt, voxel_spacing=voxel_spacing))

    dsc_wt   = float(np.mean(dsc_wt_list))  if dsc_wt_list  else 0.0
    dsc_tc   = float(np.mean(dsc_tc_list))  if dsc_tc_list  else 0.0
    dsc_et   = float(np.mean(dsc_et_list))  if dsc_et_list  else 0.0
    dsc_mean = (dsc_wt + dsc_tc + dsc_et) / 3.0
    hd95_wt  = float(np.mean([h for h in hd95_list if np.isfinite(h)])) if hd95_list else float("nan")
    return {
        "dsc_wt":       dsc_wt,
        "dsc_tc":       dsc_tc,
        "dsc_et":       dsc_et,
        "dsc_mean":     dsc_mean,
        "hd95_wt":      hd95_wt,
        "dsc_wt_std":   float(np.std(dsc_wt_list))  if dsc_wt_list  else 0.0,
        "dsc_tc_std":   float(np.std(dsc_tc_list))  if dsc_tc_list  else 0.0,
        "dsc_et_std":   float(np.std(dsc_et_list))  if dsc_et_list  else 0.0,
    }

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


def build_loaders(cfg, world_size, rank, fold: int = 0, n_folds: int = 1):
    data_dir     = cfg.data.data_dir
    image_size   = cfg.data.image_size
    modality     = getattr(cfg.data, "modality", "t2f")
    val_fraction = float(getattr(cfg.data, "val_fraction", 0.10))

    train_slice_ds = BraTSGLISliceDataset(
        data_dir, split="train", modality=modality,
        image_size=image_size, augment=True,
        tumor_ratio=cfg.data.tumor_ratio,
        val_fraction=val_fraction,
        n_folds=n_folds, fold=fold,
    )
    val_vol_ds = BraTSGLIDataset(
        data_dir, split="val", modality=modality,
        image_size=image_size, val_fraction=val_fraction,
        n_folds=n_folds, fold=fold,
    )
    train_vol_ds = BraTSGLIDataset(
        data_dir, split="train", modality=modality,
        image_size=image_size, val_fraction=val_fraction,
        n_folds=n_folds, fold=fold,
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
    parser.add_argument("--fold",            type=int, default=0,  help="CV fold (0-based)")
    parser.add_argument("--n_folds",         type=int, default=1,  help="CV folds (1=no CV)")
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
        in_chans=int(getattr(cfg.model, "in_chans", 3)),
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
    loaders = build_loaders(cfg, world_size, rank, fold=args.fold, n_folds=args.n_folds)

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

        if world_size > 1:
            dist.barrier()

        val_metrics = {}
        is_best = False
        if is_main(rank) and epoch % cfg.evaluation.val_every_n_epochs == 0:
            val_metrics = validate(unwrap(model), loaders["val"], device, cfg)
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
                log_str += (
                    f" | WT={val_metrics.get('dsc_wt', 0):.4f}"
                    f" HD95={val_metrics.get('hd95_wt', float('nan')):.2f}mm"
                    f" (best={best_metric:.4f})"
                )
            log_str += f" | {elapsed:.1f}s"
            print(log_str)

            if True:  # log grad norms every epoch
                grad_info = train_metrics.get("_grad_norms", {})
                if grad_info:
                    grad_str = "  grad norms: " + " ".join(f"{k}={v:.3e}" for k, v in grad_info.items())
                    print(grad_str)

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
