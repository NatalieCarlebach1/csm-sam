"""
Visualization utilities for CSM-SAM.

Functions:
    visualize_slice        — single slice: pre-RT / mid-RT GT / mid-RT pred overlay
    visualize_change_map   — predicted change map with legend
    make_slice_gallery     — multi-row grid of slices through a 3D volume
    visualize_patient      — combined figure: gallery + change map + metrics
    save_random_test_samples — randomly sample N patients from test set and save figures
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F


# Color scheme
COLORS = {
    "gtvp": (1.0, 0.2, 0.2, 0.5),   # Red for primary tumor
    "gtvn": (0.2, 0.6, 1.0, 0.5),   # Blue for nodal
    "pred": (0.0, 1.0, 0.0, 0.5),   # Green for prediction
    "change_stable": (1.0, 1.0, 0.0),    # Yellow: stable tumor
    "change_grown":  (1.0, 0.0, 0.0),    # Red: grown region
    "change_shrunk": (0.0, 0.5, 1.0),    # Blue: shrunk region
}

CHANGE_CMAP = {
    0: [0, 0, 0],        # Background: black
    1: [255, 255, 0],    # Stable: yellow
    2: [255, 50, 50],    # Grown: red
    3: [50, 100, 255],   # Shrunk: blue
}


def tensor_to_display(t: torch.Tensor) -> np.ndarray:
    """
    Convert a SAM2-normalized image tensor to a displayable [0,1] numpy array.

    Args:
        t : (3, H, W) or (H, W) tensor

    Returns:
        (H, W, 3) or (H, W) float32 numpy array in [0, 1]
    """
    SAM2_MEAN = np.array([0.485, 0.456, 0.406])
    SAM2_STD = np.array([0.229, 0.224, 0.225])

    if t.dim() == 3 and t.shape[0] == 3:
        arr = t.permute(1, 2, 0).cpu().numpy()
        arr = arr * SAM2_STD[None, None, :] + SAM2_MEAN[None, None, :]
        return arr.clip(0, 1).astype(np.float32)
    elif t.dim() == 2:
        return t.cpu().numpy().clip(0, 1).astype(np.float32)
    else:
        return t.squeeze().cpu().numpy().clip(0, 1).astype(np.float32)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay a binary mask on an image.

    Args:
        image : (H, W, 3) float [0, 1]
        mask  : (H, W) binary
        color : (R, G, B) floats
        alpha : overlay opacity

    Returns:
        (H, W, 3) composited image
    """
    out = image.copy()
    mask_bool = mask.astype(bool)
    r, g, b = color[:3]
    out[mask_bool, 0] = (1 - alpha) * image[mask_bool, 0] + alpha * r
    out[mask_bool, 1] = (1 - alpha) * image[mask_bool, 1] + alpha * g
    out[mask_bool, 2] = (1 - alpha) * image[mask_bool, 2] + alpha * b
    return out


def change_map_to_rgb(change_map: np.ndarray) -> np.ndarray:
    """
    Convert 4-class change map to RGB image.

    Args:
        change_map : (H, W) int array with values in {0, 1, 2, 3}

    Returns:
        (H, W, 3) uint8 RGB image
    """
    rgb = np.zeros((*change_map.shape, 3), dtype=np.uint8)
    for cls, color in CHANGE_CMAP.items():
        mask = change_map == cls
        rgb[mask] = color
    return rgb


def visualize_slice(
    pre_image: torch.Tensor,
    mid_image: torch.Tensor,
    mid_mask_gt: torch.Tensor,
    mid_mask_pred: torch.Tensor,
    pre_mask_gt: Optional[torch.Tensor] = None,
    change_map_pred: Optional[torch.Tensor] = None,
    gate_vals: Optional[torch.Tensor] = None,
    title: str = "",
    figsize: tuple = (16, 5),
) -> plt.Figure:
    """
    Visualize one slice: pre-RT (with GT) | mid-RT GT | mid-RT prediction | optionals.

    Returns a matplotlib Figure.
    """
    pre_img = tensor_to_display(pre_image)
    mid_img = tensor_to_display(mid_image)

    gt_mask = mid_mask_gt.squeeze().cpu().numpy() > 0.5
    pred_mask = (torch.sigmoid(mid_mask_pred) > 0.5).squeeze().cpu().numpy() if mid_mask_pred.dim() >= 2 else mid_mask_pred.squeeze().cpu().numpy() > 0.5

    n_cols = 4 if change_map_pred is not None else 3
    if gate_vals is not None:
        n_cols += 1

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes)

    # Panel 1: Pre-RT with GT mask
    ax = axes[0]
    pre_vis = overlay_mask(pre_img, pre_mask_gt.squeeze().cpu().numpy() > 0.5 if pre_mask_gt is not None else np.zeros_like(gt_mask), COLORS["gtvp"][:3])
    ax.imshow(pre_vis)
    ax.set_title("Pre-RT (GT)", fontsize=10)
    ax.axis("off")

    # Panel 2: Mid-RT GT
    ax = axes[1]
    mid_gt_vis = overlay_mask(mid_img, gt_mask, COLORS["gtvp"][:3])
    ax.imshow(mid_gt_vis)
    ax.set_title("Mid-RT GT", fontsize=10)
    ax.axis("off")

    # Panel 3: Mid-RT Prediction
    ax = axes[2]
    mid_pred_vis = overlay_mask(mid_img, pred_mask, COLORS["pred"][:3])
    ax.imshow(mid_pred_vis)

    # Compute per-slice Dice for title
    intersection = (pred_mask & gt_mask).sum()
    dice = (2 * intersection + 1e-6) / (pred_mask.sum() + gt_mask.sum() + 1e-6)
    ax.set_title(f"Prediction (DSC={dice:.3f})", fontsize=10)
    ax.axis("off")

    col = 3
    # Panel 4: Change map (optional)
    if change_map_pred is not None:
        ax = axes[col]
        col += 1
        change = change_map_pred.argmax(dim=0).cpu().numpy() if change_map_pred.dim() == 3 else change_map_pred.cpu().numpy()
        ax.imshow(change_map_to_rgb(change))
        legend_patches = [
            mpatches.Patch(color=np.array(CHANGE_CMAP[1]) / 255, label="Stable"),
            mpatches.Patch(color=np.array(CHANGE_CMAP[2]) / 255, label="Grown"),
            mpatches.Patch(color=np.array(CHANGE_CMAP[3]) / 255, label="Shrunk"),
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=7)
        ax.set_title("Change Map (pred)", fontsize=10)
        ax.axis("off")

    # Panel 5: Gate values (optional)
    if gate_vals is not None:
        ax = axes[col]
        gate = gate_vals.squeeze().cpu().numpy() if hasattr(gate_vals, "cpu") else gate_vals
        im = ax.imshow(gate, cmap="plasma", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Cross-session gate\n(1=pre-RT, 0=within)", fontsize=9)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def visualize_change_map(
    pre_image: torch.Tensor,
    mid_image: torch.Tensor,
    change_pred: np.ndarray,
    change_gt: Optional[np.ndarray] = None,
    figsize: tuple = (14, 5),
    title: str = "",
) -> plt.Figure:
    """
    Side-by-side: pre-RT | mid-RT | predicted change map | (optional) GT change map.
    """
    pre_img = tensor_to_display(pre_image)
    mid_img = tensor_to_display(mid_image)

    n_cols = 3 if change_gt is None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    axes[0].imshow(pre_img)
    axes[0].set_title("Pre-RT MRI", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(mid_img)
    axes[1].set_title("Mid-RT MRI", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(change_map_to_rgb(change_pred))
    axes[2].set_title("Predicted Change", fontsize=10)
    axes[2].axis("off")

    if change_gt is not None:
        axes[3].imshow(change_map_to_rgb(change_gt))
        axes[3].set_title("GT Change", fontsize=10)
        axes[3].axis("off")

    legend_patches = [
        mpatches.Patch(color=np.array(CHANGE_CMAP[1]) / 255, label="Stable tumor"),
        mpatches.Patch(color=np.array(CHANGE_CMAP[2]) / 255, label="Grown region"),
        mpatches.Patch(color=np.array(CHANGE_CMAP[3]) / 255, label="Shrunk region"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.05))

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def make_slice_gallery(
    mid_images: torch.Tensor,
    mid_masks_gt: torch.Tensor,
    mid_masks_pred: torch.Tensor,
    n_slices: int = 8,
    figsize: tuple | None = None,
    title: str = "",
) -> plt.Figure:
    """
    Create a gallery showing n_slices evenly-spaced slices through a volume.

    Each column = one slice. Three rows: GT / Prediction / Overlay.

    Args:
        mid_images     : (N, 3, H, W)
        mid_masks_gt   : (N, 1, H, W)
        mid_masks_pred : (N, 1, H, W) — logits or probabilities
    """
    N = mid_images.shape[0]
    n_slices = min(n_slices, N)
    indices = np.linspace(0, N - 1, n_slices, dtype=int)

    if figsize is None:
        figsize = (n_slices * 2.5, 7)

    fig, axes = plt.subplots(3, n_slices, figsize=figsize)
    row_titles = ["Mid-RT + GT", "Mid-RT + Pred", "GT vs Pred"]

    for col, idx in enumerate(indices):
        img = tensor_to_display(mid_images[idx])
        gt = (mid_masks_gt[idx].squeeze().cpu().numpy() > 0.5)
        pred_raw = mid_masks_pred[idx].squeeze().cpu()
        pred = (torch.sigmoid(pred_raw) > 0.5).numpy() if pred_raw.min() < 0 else (pred_raw > 0.5).numpy()

        # Row 0: GT overlay
        axes[0, col].imshow(overlay_mask(img, gt, COLORS["gtvp"][:3]))
        axes[0, col].axis("off")
        if col == 0:
            axes[0, col].set_ylabel(row_titles[0], fontsize=8)

        # Row 1: Pred overlay
        axes[1, col].imshow(overlay_mask(img, pred, COLORS["pred"][:3]))
        axes[1, col].axis("off")
        if col == 0:
            axes[1, col].set_ylabel(row_titles[1], fontsize=8)

        # Row 2: GT (red) vs Pred (green) vs overlap (yellow)
        combined = np.zeros((*img.shape[:2], 3))
        combined[gt, 0] = 0.8
        combined[pred, 1] = 0.8
        combined[gt & pred] = [0.8, 0.8, 0.0]
        axes[2, col].imshow(img * 0.4 + combined * 0.6)
        axes[2, col].axis("off")
        if col == 0:
            axes[2, col].set_ylabel(row_titles[2], fontsize=8)

        axes[0, col].set_title(f"z={idx}", fontsize=8)

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLORS["gtvp"][:3], label="GT tumor"),
        mpatches.Patch(color=COLORS["pred"][:3], label="Prediction"),
        mpatches.Patch(color=(0.8, 0.8, 0.0), label="Overlap"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


def visualize_patient(
    patient_data: dict,
    predictions: dict,
    metrics: dict,
    output_path: Path,
    n_gallery_slices: int = 8,
):
    """
    Generate and save all visualization figures for one patient.

    Creates:
        {output_path}/{patient_id}_overlay.png
        {output_path}/{patient_id}_change_map.png
        {output_path}/{patient_id}_gallery.png

    Args:
        patient_data : dict from HNTSMRGDataset.__getitem__
        predictions  : dict with 'masks', 'change_map', 'gate_vals'
        metrics      : dict from evaluate_patient
        output_path  : directory to save figures
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    pid = patient_data.get("patient_id", "unknown")

    mid_images = patient_data["mid_images"]   # (N, 3, H, W)
    pre_images = patient_data["pre_images"]
    mid_masks_gt = patient_data["mid_masks"]  # (N, 1, H, W)
    mid_masks_pred = predictions["masks"]      # (N, 1, H, W)

    N = mid_images.shape[0]

    # Find the most informative slice (largest tumor in GT)
    tumor_sizes = [mid_masks_gt[i].sum().item() for i in range(N)]
    best_slice = int(np.argmax(tumor_sizes)) if max(tumor_sizes) > 0 else N // 2

    # 1. Single-slice overlay
    gate_slice = None
    if "gate_vals" in predictions and predictions["gate_vals"] is not None:
        gate_slice = predictions["gate_vals"][best_slice] if predictions["gate_vals"].dim() > 2 else predictions["gate_vals"]

    change_slice = None
    if "change_map" in predictions and predictions["change_map"] is not None:
        change_slice = predictions["change_map"][best_slice]

    title_str = (
        f"{pid} | aggDSC={metrics.get('agg_dsc', 0):.3f} | "
        f"GTVp DSC={metrics.get('dsc_gtvp', 0):.3f} | "
        f"GTVn DSC={metrics.get('dsc_gtvn', 0):.3f}"
    )

    fig1 = visualize_slice(
        pre_images[best_slice],
        mid_images[best_slice],
        mid_masks_gt[best_slice],
        mid_masks_pred[best_slice],
        pre_mask_gt=patient_data.get("pre_masks", None)[best_slice] if "pre_masks" in patient_data else None,
        change_map_pred=change_slice,
        gate_vals=gate_slice,
        title=title_str,
    )
    fig1.savefig(output_path / f"{pid}_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # 2. Change map gallery (best slice)
    if "change_map" in predictions and predictions["change_map"] is not None:
        change_np = predictions["change_map"][best_slice].argmax(dim=0).cpu().numpy()
        fig2 = visualize_change_map(
            pre_images[best_slice],
            mid_images[best_slice],
            change_np,
            title=f"{pid} — Change Map",
        )
        fig2.savefig(output_path / f"{pid}_change_map.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)

    # 3. Slice gallery
    fig3 = make_slice_gallery(
        mid_images,
        mid_masks_gt,
        mid_masks_pred,
        n_slices=n_gallery_slices,
        title=f"{pid} — Slice Gallery | aggDSC={metrics.get('agg_dsc', 0):.3f}",
    )
    fig3.savefig(output_path / f"{pid}_gallery.png", dpi=120, bbox_inches="tight")
    plt.close(fig3)


def save_random_test_samples(
    model,
    test_loader,
    output_dir: str | Path,
    n_samples: int = 10,
    device: str = "cuda",
    seed: int = 42,
):
    """
    Run inference on N randomly selected test patients and save visualizations.

    Args:
        model      : CSMSAM model (in eval mode)
        test_loader: DataLoader[HNTSMRGDataset]
        output_dir : directory for output figures
        n_samples  : number of random patients to visualize
        device     : torch device
        seed       : random seed for reproducibility
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Randomly select patient indices
    random.seed(seed)
    all_indices = list(range(len(test_loader.dataset)))
    selected = random.sample(all_indices, min(n_samples, len(all_indices)))

    model.eval()

    for i, patient_idx in enumerate(selected):
        print(f"  Visualizing patient {i + 1}/{len(selected)} (idx={patient_idx})...")

        patient_data = test_loader.dataset[patient_idx]
        pid = patient_data["patient_id"]

        pre_images = patient_data["pre_images"].unsqueeze(0).to(device)  # (1, N, 3, H, W)
        mid_images = patient_data["mid_images"]  # (N, 3, H, W)
        mid_masks_gt = patient_data["mid_masks"]  # (N, 1, H, W)
        weeks = torch.tensor([patient_data["weeks_elapsed"]], dtype=torch.long, device=device)

        with torch.no_grad():
            # Encode pre-RT memory
            M_pre = model.encode_pre_rt(
                pre_images,
                pre_masks=patient_data["pre_masks"].unsqueeze(0).to(device) if "pre_masks" in patient_data else None,
            )

            # Segment mid-RT slice by slice
            model.reset_mid_session_memory()
            pred_masks_list = []
            change_maps_list = []
            gate_vals_list = []

            N = mid_images.shape[0]
            for j in range(N):
                batch_mid = mid_images[j].unsqueeze(0).to(device)  # (1, 3, H, W)
                batch_pre = pre_images[0, min(j, pre_images.shape[1] - 1)].unsqueeze(0)  # (1, 3, H, W)

                out = model(
                    mid_images=batch_mid,
                    M_pre=M_pre,
                    pre_images=batch_pre,
                    weeks_elapsed=weeks,
                    return_change_map=True,
                )
                pred_masks_list.append(out["masks"].squeeze(0).cpu())
                if "change_map" in out:
                    change_maps_list.append(out["change_map"].squeeze(0).cpu())
                if "gate_vals" in out:
                    gate_vals_list.append(out["gate_vals"].squeeze(0).cpu())

        pred_masks = torch.stack(pred_masks_list)  # (N, 1, H, W)
        change_maps = torch.stack(change_maps_list) if change_maps_list else None
        gate_vals = torch.stack(gate_vals_list) if gate_vals_list else None

        # Compute metrics (simplified — use combined mask)
        from csmsam.utils.metrics import compute_dice, compute_agg_dsc
        pred_np = (torch.sigmoid(pred_masks) > 0.5).squeeze(1).numpy()
        gt_np = (mid_masks_gt > 0.5).squeeze(1).numpy()

        metrics = {
            "agg_dsc": compute_dice(pred_np, gt_np),
            "dsc_gtvp": compute_dice(
                (torch.sigmoid(pred_masks) > 0.5).squeeze(1).numpy(),
                (patient_data.get("mid_masks_gtvp", mid_masks_gt) > 0.5).squeeze(1).numpy(),
            ),
            "dsc_gtvn": compute_dice(
                (torch.sigmoid(pred_masks) > 0.5).squeeze(1).numpy(),
                (patient_data.get("mid_masks_gtvn", mid_masks_gt) > 0.5).squeeze(1).numpy(),
            ),
        }

        predictions = {
            "masks": pred_masks,
            "change_map": change_maps,
            "gate_vals": gate_vals,
        }

        visualize_patient(patient_data, predictions, metrics, output_dir)
        print(f"    Saved: {output_dir}/{pid}_*.png | aggDSC={metrics['agg_dsc']:.3f}")

    print(f"\nVisualization complete. {len(selected)} patients saved to {output_dir}")
