"""
CSM-SAM Visualization Script.

Generates publication-quality figures for N random test patients:
  1. Overlay figure: pre-RT | mid-RT GT | mid-RT prediction | change map | gate vals
  2. Slice gallery: multi-row grid through the 3D volume
  3. Change map figure: pre-RT | mid-RT | predicted change | GT change
  4. Per-structure figure: GTVp pred vs GT and GTVn pred vs GT side by side

Usage:
    python visualize.py --checkpoint checkpoints/best.pth --n_samples 10
    python visualize.py --checkpoint checkpoints/best.pth --patient_ids patient_001 patient_042
    python visualize.py --checkpoint checkpoints/best.pth --worst_n 5  # worst aggDSC patients
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from csmsam.datasets import HNTSMRGDataset
from csmsam.modeling import CSMSAM
from csmsam.utils.metrics import evaluate_patient, compute_dice
from csmsam.utils.visualization import (
    COLORS,
    overlay_mask,
    tensor_to_display,
    visualize_patient,
    visualize_slice,
    make_slice_gallery,
    visualize_change_map,
)


@torch.no_grad()
def run_inference_for_patient(
    model: CSMSAM,
    patient_data: dict,
    device: str,
    threshold: float = 0.5,
) -> dict:
    """Run full 3D inference and return predictions dict with per-structure outputs."""
    model.eval()

    pre_images = patient_data["pre_images"].to(device)
    mid_images = patient_data["mid_images"].to(device)
    pre_masks = patient_data["pre_masks"].to(device) if "pre_masks" in patient_data else None
    pre_masks_gtvp = patient_data["pre_masks_gtvp"].to(device)
    pre_masks_gtvn = patient_data["pre_masks_gtvn"].to(device)
    weeks = patient_data["weeks_elapsed"]
    weeks_t = torch.tensor([int(weeks)], dtype=torch.long, device=device)

    N = pre_images.shape[0]

    M_pre = model.encode_pre_rt(
        pre_images.unsqueeze(0),
        pre_masks.unsqueeze(0) if pre_masks is not None else None,
    )

    model.reset_mid_session_memory()
    pred_masks, change_maps, gate_vals = [], [], []

    for i in range(N):
        out = model(
            mid_images=mid_images[i].unsqueeze(0),
            M_pre=M_pre,
            pre_images=pre_images[i].unsqueeze(0),
            weeks_elapsed=weeks_t,
            pre_gtvp_mask=pre_masks_gtvp[i].unsqueeze(0),
            pre_gtvn_mask=pre_masks_gtvn[i].unsqueeze(0),
            return_change_map=True,
            detach_memory=True,
        )
        pred_masks.append(out["masks"].squeeze(0).cpu())  # (2, H, W)
        if "change_map" in out and out["change_map"] is not None:
            change_maps.append(out["change_map"].squeeze(0).cpu())
        if "gate_vals" in out and out["gate_vals"] is not None:
            gate_vals.append(out["gate_vals"].squeeze(0).cpu())

    pred_masks_t = torch.stack(pred_masks)                # (N, 2, H, W)
    probs = torch.sigmoid(pred_masks_t)
    pred_gtvp = (probs[:, 0] > threshold).numpy()         # (N, H, W)
    pred_gtvn = (probs[:, 1] > threshold).numpy()         # (N, H, W)
    pred_combined = (pred_gtvp | pred_gtvn)               # (N, H, W)

    # Per-structure ground truth
    gt_gtvp = (patient_data["mid_masks_gtvp"] > 0.5).squeeze(1).numpy()
    gt_gtvn = (patient_data["mid_masks_gtvn"] > 0.5).squeeze(1).numpy()

    dsc_gtvp = compute_dice(pred_gtvp, gt_gtvp)
    dsc_gtvn = compute_dice(pred_gtvn, gt_gtvn)
    agg_dsc = (dsc_gtvp + dsc_gtvn) / 2.0

    metrics = {
        "dsc_gtvp": dsc_gtvp,
        "dsc_gtvn": dsc_gtvn,
        "agg_dsc": agg_dsc,
    }

    return {
        "masks": pred_masks_t,                            # (N, 2, H, W) logits
        "masks_gtvp": pred_gtvp,                          # (N, H, W) bool
        "masks_gtvn": pred_gtvn,                          # (N, H, W) bool
        "masks_binary": pred_combined,                    # (N, H, W) bool, combined
        "change_map": torch.stack(change_maps) if change_maps else None,
        "gate_vals": torch.stack(gate_vals) if gate_vals else None,
        "metrics": metrics,
    }


def visualize_per_structure(
    patient_data: dict,
    predictions: dict,
    output_path: Path,
):
    """
    Side-by-side GTVp and GTVn pred vs GT figure for the best slice of each
    structure (largest GT area). Saved to ``{patient_id}_per_structure.png``.

    Layout:
        Row 0 (GTVp): mid-RT+GT | mid-RT+pred | GT vs pred diff
        Row 1 (GTVn): mid-RT+GT | mid-RT+pred | GT vs pred diff
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    pid = patient_data.get("patient_id", "unknown")

    mid_images = patient_data["mid_images"]                         # (N, 3, H, W)
    gt_gtvp_vol = (patient_data["mid_masks_gtvp"] > 0.5).squeeze(1).numpy()
    gt_gtvn_vol = (patient_data["mid_masks_gtvn"] > 0.5).squeeze(1).numpy()
    pred_gtvp_vol = predictions["masks_gtvp"]
    pred_gtvn_vol = predictions["masks_gtvn"]

    def _best_slice(vol: np.ndarray) -> int:
        sums = vol.reshape(vol.shape[0], -1).sum(axis=1)
        return int(np.argmax(sums)) if sums.max() > 0 else vol.shape[0] // 2

    idx_p = _best_slice(gt_gtvp_vol)
    idx_n = _best_slice(gt_gtvn_vol)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    structures = [
        ("GTVp", idx_p, gt_gtvp_vol, pred_gtvp_vol, COLORS["gtvp"][:3]),
        ("GTVn", idx_n, gt_gtvn_vol, pred_gtvn_vol, COLORS["gtvn"][:3]),
    ]

    for row, (name, idx, gt_vol, pred_vol, color) in enumerate(structures):
        img = tensor_to_display(mid_images[idx])
        gt = gt_vol[idx].astype(bool)
        pr = pred_vol[idx].astype(bool)

        # Col 0: GT overlay
        axes[row, 0].imshow(overlay_mask(img, gt, color))
        axes[row, 0].set_title(f"{name} GT (z={idx})", fontsize=10)
        axes[row, 0].axis("off")

        # Col 1: Prediction overlay
        axes[row, 1].imshow(overlay_mask(img, pr, COLORS["pred"][:3]))
        inter = (gt & pr).sum()
        dsc = (2 * inter + 1e-6) / (gt.sum() + pr.sum() + 1e-6)
        axes[row, 1].set_title(f"{name} Pred (slice DSC={dsc:.3f})", fontsize=10)
        axes[row, 1].axis("off")

        # Col 2: GT (red) vs pred (green) overlap (yellow)
        combined = np.zeros((*img.shape[:2], 3))
        combined[gt, 0] = 0.8
        combined[pr, 1] = 0.8
        combined[gt & pr] = [0.8, 0.8, 0.0]
        axes[row, 2].imshow(img * 0.4 + combined * 0.6)
        axes[row, 2].set_title(f"{name} GT vs Pred", fontsize=10)
        axes[row, 2].axis("off")

    legend_patches = [
        mpatches.Patch(color=(0.8, 0.0, 0.0), label="GT only"),
        mpatches.Patch(color=(0.0, 0.8, 0.0), label="Pred only"),
        mpatches.Patch(color=(0.8, 0.8, 0.0), label="Overlap"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    m = predictions.get("metrics", {})
    fig.suptitle(
        f"{pid} — Per-Structure | "
        f"GTVp DSC={m.get('dsc_gtvp', 0):.3f} | "
        f"GTVn DSC={m.get('dsc_gtvn', 0):.3f} | "
        f"aggDSC={m.get('agg_dsc', 0):.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path / f"{pid}_per_structure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _visualize_all(patient_data, predictions, output_path, n_gallery_slices):
    """Call shared visualizer + per-structure figure."""
    # The shared helper expects a (N, 1, H, W) "masks" for gallery/overlay.
    # Re-pack combined predictions into a logits-shaped tensor so existing
    # helpers (which call sigmoid) still work, without mutating the original
    # predictions dict.
    combined_logits = predictions["masks"].max(dim=1, keepdim=True).values  # (N, 1, H, W)
    legacy_predictions = {
        **predictions,
        "masks": combined_logits,
    }
    visualize_patient(
        patient_data=patient_data,
        predictions=legacy_predictions,
        metrics=predictions["metrics"],
        output_path=output_path,
        n_gallery_slices=n_gallery_slices,
    )
    visualize_per_structure(patient_data, predictions, output_path)


def plot_summary_histogram(all_metrics: list[dict], output_path: Path):
    """
    Plot aggDSC distribution across all test patients.
    Useful for identifying patterns (e.g., GTVp vs GTVn difficulty).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, key, label in zip(
        axes,
        ["agg_dsc", "dsc_gtvp", "dsc_gtvn"],
        ["aggDSC", "GTVp DSC", "GTVn DSC"],
    ):
        values = [m[key] for m in all_metrics if not np.isnan(m.get(key, float("nan")))]
        if not values:
            continue
        ax.hist(values, bins=20, range=(0, 1), color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(values), color="red", linestyle="--", label=f"Mean={np.mean(values):.3f}")
        ax.axvline(np.median(values), color="orange", linestyle="--", label=f"Median={np.median(values):.3f}")
        ax.set_title(f"{label} Distribution", fontsize=11)
        ax.set_xlabel("DSC", fontsize=9)
        ax.set_ylabel("# Patients", fontsize=9)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("CSM-SAM Test Set Performance Distribution", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path / "dsc_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved DSC distribution plot: {output_path}/dsc_distribution.png")


def plot_scatter_gtvp_vs_gtvn(all_metrics: list[dict], output_path: Path):
    """Scatter plot GTVp DSC vs GTVn DSC to identify correlation."""
    gtvp = [m.get("dsc_gtvp", float("nan")) for m in all_metrics]
    gtvn = [m.get("dsc_gtvn", float("nan")) for m in all_metrics]

    valid = [(p, n) for p, n in zip(gtvp, gtvn) if not np.isnan(p) and not np.isnan(n)]
    if not valid:
        return

    ps, ns = zip(*valid)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(ps, ns, alpha=0.6, s=40, c="steelblue", edgecolors="white", linewidths=0.5)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="y=x")
    ax.set_xlabel("GTVp DSC", fontsize=11)
    ax.set_ylabel("GTVn DSC", fontsize=11)
    ax.set_title(f"GTVp vs GTVn DSC\n(n={len(ps)} patients)", fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "gtvp_vs_gtvn_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved GTVp vs GTVn scatter: {output_path}/gtvp_vs_gtvn_scatter.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize CSM-SAM predictions")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default="results/visualizations")
    parser.add_argument("--n_samples", type=int, default=10, help="Random patients to visualize")
    parser.add_argument("--patient_ids", nargs="*", default=None, help="Specific patient IDs")
    parser.add_argument("--worst_n", type=int, default=0, help="Also visualize N worst-performing patients")
    parser.add_argument("--best_n", type=int, default=0, help="Also visualize N best-performing patients")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gallery_slices", type=int, default=8)
    parser.add_argument("--all_patients", action="store_true", help="Visualize all test patients")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config from checkpoint
    state = torch.load(args.checkpoint, map_location="cpu")
    if "config" in state:
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(state["config"])
    else:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(args.config or "configs/default.yaml")

    if args.data_dir:
        cfg.data.data_dir = args.data_dir

    print("=" * 60)
    print("CSM-SAM Visualization")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Build model
    print("\nLoading model...")
    model = CSMSAM.from_pretrained(
        sam2_checkpoint=cfg.model.sam2_checkpoint,
        sam2_cfg=cfg.model.sam2_cfg,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        n_memory_frames=cfg.model.n_memory_frames,
        spatial_pool_size=cfg.model.spatial_pool_size,
        max_weeks=cfg.model.max_weeks,
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print("Model loaded.")

    # Build dataset
    dataset = HNTSMRGDataset(
        data_dir=cfg.data.data_dir,
        split=args.split,
        image_size=cfg.data.image_size,
    )

    # Select patients to visualize
    if args.all_patients:
        selected_indices = list(range(len(dataset)))
    elif args.patient_ids:
        pid_to_idx = {dataset.patient_dirs[i].name: i for i in range(len(dataset))}
        selected_indices = [pid_to_idx[pid] for pid in args.patient_ids if pid in pid_to_idx]
        if not selected_indices:
            print(f"No matching patient IDs found. Available: {list(pid_to_idx.keys())[:10]}")
            return
    else:
        random.seed(args.seed)
        selected_indices = random.sample(range(len(dataset)), min(args.n_samples, len(dataset)))

    all_metrics = []
    all_results = []  # (idx, metrics) for worst/best selection

    print(f"\nProcessing {len(selected_indices)} patients...")

    for idx in tqdm(selected_indices, desc="Patients"):
        patient_data = dataset[idx]
        pid = patient_data["patient_id"]

        try:
            predictions = run_inference_for_patient(model, patient_data, device, args.threshold)
            metrics = predictions["metrics"]
            all_metrics.append(metrics)
            all_results.append((idx, metrics))

            _visualize_all(patient_data, predictions, output_dir, args.gallery_slices)
            print(f"  {pid}: aggDSC={metrics['agg_dsc']:.3f} "
                  f"(GTVp={metrics['dsc_gtvp']:.3f}, GTVn={metrics['dsc_gtvn']:.3f})")

        except Exception as e:
            print(f"  {pid}: ERROR — {e}")

    # Worst/best patients
    if args.worst_n > 0 and all_results:
        all_results_sorted = sorted(all_results, key=lambda x: x[1].get("agg_dsc", 1.0))
        worst_idxs = [idx for idx, _ in all_results_sorted[:args.worst_n]]
        remaining = [i for i in worst_idxs if i not in selected_indices]
        print(f"\nVisualizing {len(remaining)} worst patients...")
        for idx in remaining:
            patient_data = dataset[idx]
            predictions = run_inference_for_patient(model, patient_data, device, args.threshold)
            _visualize_all(patient_data, predictions, output_dir / "worst", args.gallery_slices)

    if args.best_n > 0 and all_results:
        all_results_sorted = sorted(all_results, key=lambda x: x[1].get("agg_dsc", 0.0), reverse=True)
        best_idxs = [idx for idx, _ in all_results_sorted[:args.best_n]]
        remaining = [i for i in best_idxs if i not in selected_indices]
        print(f"\nVisualizing {len(remaining)} best patients...")
        for idx in remaining:
            patient_data = dataset[idx]
            predictions = run_inference_for_patient(model, patient_data, device, args.threshold)
            _visualize_all(patient_data, predictions, output_dir / "best", args.gallery_slices)

    # Summary plots
    if all_metrics:
        print("\nGenerating summary plots...")
        plot_summary_histogram(all_metrics, output_dir)
        plot_scatter_gtvp_vs_gtvn(all_metrics, output_dir)

        # Print summary
        agg_vals = [m["agg_dsc"] for m in all_metrics if not np.isnan(m.get("agg_dsc", float("nan")))]
        if agg_vals:
            print(f"\nVisualized {len(agg_vals)} patients:")
            print(f"  Mean aggDSC : {np.mean(agg_vals):.4f}")
            print(f"  Median      : {np.median(agg_vals):.4f}")
            print(f"  Min / Max   : {np.min(agg_vals):.4f} / {np.max(agg_vals):.4f}")

    print(f"\nAll figures saved to: {output_dir}")
    print(f"Files per patient: {{patient_id}}_overlay.png, _gallery.png, _change_map.png, _per_structure.png")


if __name__ == "__main__":
    main()
