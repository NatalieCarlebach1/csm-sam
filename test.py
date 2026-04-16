"""
CSM-SAM Test / Evaluation Script.

Runs inference on the test set, computes metrics, and saves predictions.

Usage:
    python test.py --checkpoint checkpoints/csmsam_default/best.pth
    python test.py --checkpoint checkpoints/best.pth --split val
    python test.py --checkpoint checkpoints/best.pth --compare baselines/results/
    python test.py --checkpoint checkpoints/best.pth --tta
    python test.py --checkpoint checkpoints/best.pth --fold 0 --n_folds 5
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

from csmsam.datasets import HNTSMRGDataset
from csmsam.modeling import CSMSAM
from csmsam.utils.metrics import (
    evaluate_patient,
    aggregate_metrics,
    format_results_table,
)


def _forward_logits(
    model: CSMSAM,
    mid_slice: torch.Tensor,
    pre_slice: torch.Tensor,
    M_pre: torch.Tensor,
    weeks_t: torch.Tensor,
    pre_gtvp_slice: torch.Tensor | None,
    pre_gtvn_slice: torch.Tensor | None,
    return_change_map: bool,
) -> dict:
    """Single forward through the per-structure model. Returns raw dict."""
    return model(
        mid_images=mid_slice,
        M_pre=M_pre,
        pre_images=pre_slice,
        weeks_elapsed=weeks_t,
        pre_gtvp_mask=pre_gtvp_slice,
        pre_gtvn_mask=pre_gtvn_slice,
        return_change_map=return_change_map,
        detach_memory=True,
    )


@torch.no_grad()
def inference_patient(
    model: CSMSAM,
    patient_data: dict,
    device: str,
    threshold: float = 0.5,
    voxel_spacing: tuple = (3.0, 1.0, 1.0),
    tta: bool = False,
) -> tuple[dict, dict]:
    """
    Run full 3D inference on one patient.

    Emits per-structure (GTVp, GTVn) predictions using the new two-channel
    CSM-SAM API. If ``tta`` is set, averages logits with the horizontal-flipped
    pass (only H-flip — V-flip is anatomically unsafe).

    Returns:
        predictions: dict of numpy volume predictions
        metrics:     dict of evaluation metrics
    """
    model.eval()

    pre_images = patient_data["pre_images"].to(device)              # (N, 3, H, W)
    mid_images = patient_data["mid_images"].to(device)              # (N, 3, H, W)
    pre_masks = patient_data["pre_masks"].to(device)                # (N, 1, H, W) combined
    pre_masks_gtvp = patient_data["pre_masks_gtvp"].to(device)      # (N, 1, H, W)
    pre_masks_gtvn = patient_data["pre_masks_gtvn"].to(device)      # (N, 1, H, W)
    weeks = patient_data["weeks_elapsed"]
    weeks_t = torch.tensor([int(weeks)], dtype=torch.long, device=device)

    N = pre_images.shape[0]

    # Encode pre-RT memory (original orientation)
    M_pre = model.encode_pre_rt(
        pre_images.unsqueeze(0),
        pre_masks.unsqueeze(0),
    )  # (1, N_mem, C)

    # Encode pre-RT memory for H-flipped pass, if using TTA.
    if tta:
        pre_images_flip = torch.flip(pre_images, dims=[-1])
        pre_masks_flip = torch.flip(pre_masks, dims=[-1])
        M_pre_flip = model.encode_pre_rt(
            pre_images_flip.unsqueeze(0),
            pre_masks_flip.unsqueeze(0),
        )

    # Slice-by-slice mid-RT segmentation
    model.reset_mid_session_memory()

    gtvp_logits_list: list[torch.Tensor] = []
    gtvn_logits_list: list[torch.Tensor] = []
    change_maps_list: list[torch.Tensor] = []
    gate_vals_list: list[torch.Tensor] = []

    for i in tqdm(range(N), desc="Slices", leave=False):
        mid_slice = mid_images[i].unsqueeze(0)
        pre_slice = pre_images[i].unsqueeze(0)
        pre_gtvp_slice = pre_masks_gtvp[i].unsqueeze(0)
        pre_gtvn_slice = pre_masks_gtvn[i].unsqueeze(0)

        out = _forward_logits(
            model,
            mid_slice=mid_slice,
            pre_slice=pre_slice,
            M_pre=M_pre,
            weeks_t=weeks_t,
            pre_gtvp_slice=pre_gtvp_slice,
            pre_gtvn_slice=pre_gtvn_slice,
            return_change_map=True,
        )

        logits = out["masks"].squeeze(0).cpu()  # (2, H, W)

        if tta:
            # Reset within-session memory before the flipped pass so the two
            # sequences don't contaminate each other.
            saved_mid = model._M_mid
            model._M_mid = None

            mid_slice_f = torch.flip(mid_slice, dims=[-1])
            pre_slice_f = torch.flip(pre_slice, dims=[-1])
            pre_gtvp_f = torch.flip(pre_gtvp_slice, dims=[-1])
            pre_gtvn_f = torch.flip(pre_gtvn_slice, dims=[-1])

            out_f = _forward_logits(
                model,
                mid_slice=mid_slice_f,
                pre_slice=pre_slice_f,
                M_pre=M_pre_flip,
                weeks_t=weeks_t,
                pre_gtvp_slice=pre_gtvp_f,
                pre_gtvn_slice=pre_gtvn_f,
                return_change_map=False,
            )
            logits_f = torch.flip(out_f["masks"].squeeze(0).cpu(), dims=[-1])
            logits = 0.5 * (logits + logits_f)

            # Restore the original-orientation within-session memory so it keeps
            # accumulating correctly across slices.
            model._M_mid = saved_mid

        gtvp_logits_list.append(logits[0])
        gtvn_logits_list.append(logits[1])
        if "change_map" in out and out["change_map"] is not None:
            change_maps_list.append(out["change_map"].squeeze(0).cpu())
        if "gate_vals" in out and out["gate_vals"] is not None:
            gate_vals_list.append(out["gate_vals"].squeeze(0).cpu())

    gtvp_logits = torch.stack(gtvp_logits_list)               # (N, H, W)
    gtvn_logits = torch.stack(gtvn_logits_list)               # (N, H, W)
    pred_masks_2ch = torch.stack(
        [gtvp_logits, gtvn_logits], dim=1
    )                                                         # (N, 2, H, W)

    pred_gtvp_vol = (torch.sigmoid(gtvp_logits) > threshold).numpy()  # (N, H, W)
    pred_gtvn_vol = (torch.sigmoid(gtvn_logits) > threshold).numpy()  # (N, H, W)
    pred_combined = (pred_gtvp_vol | pred_gtvn_vol).astype(bool)

    # Ground truth (real per-structure)
    gt_gtvp = (patient_data["mid_masks_gtvp"] > 0.5).squeeze(1).numpy()
    gt_gtvn = (patient_data["mid_masks_gtvn"] > 0.5).squeeze(1).numpy()

    metrics = evaluate_patient(
        pred_masks=pred_combined,
        pred_gtvp=pred_gtvp_vol,
        pred_gtvn=pred_gtvn_vol,
        target_gtvp=gt_gtvp,
        target_gtvn=gt_gtvn,
        voxel_spacing=voxel_spacing,
    )

    predictions = {
        "masks": pred_masks_2ch,                              # (N, 2, H, W) logits
        "masks_gtvp": pred_gtvp_vol,                          # (N, H, W) bool
        "masks_gtvn": pred_gtvn_vol,                          # (N, H, W) bool
        "masks_binary": pred_combined,                        # (N, H, W) combined
        "change_map": torch.stack(change_maps_list) if change_maps_list else None,
        "gate_vals": torch.stack(gate_vals_list) if gate_vals_list else None,
    }

    return predictions, metrics


def save_predictions_nifti(
    predictions: dict,
    patient_data: dict,
    output_dir: Path,
):
    """Save binary per-structure prediction masks as NIfTI files."""
    if not HAS_SITK:
        print("  Warning: SimpleITK not available, skipping NIfTI save")
        return

    pid = patient_data["patient_id"]
    patient_out = output_dir / "predictions" / pid
    patient_out.mkdir(parents=True, exist_ok=True)

    spacing = (1.0, 1.0, 3.0)

    gtvp_arr = predictions["masks_gtvp"].astype(np.uint8)
    gtvp_img = sitk.GetImageFromArray(gtvp_arr)
    gtvp_img.SetSpacing(spacing)
    sitk.WriteImage(gtvp_img, str(patient_out / "pred_mid_GTVp.nii.gz"))

    gtvn_arr = predictions["masks_gtvn"].astype(np.uint8)
    gtvn_img = sitk.GetImageFromArray(gtvn_arr)
    gtvn_img.SetSpacing(spacing)
    sitk.WriteImage(gtvn_img, str(patient_out / "pred_mid_GTVn.nii.gz"))


def load_checkpoint_cfg(checkpoint_path: str, config_path: str | None) -> OmegaConf:
    """Load config from checkpoint or fallback to provided config file."""
    state = torch.load(checkpoint_path, map_location="cpu")
    if "config" in state:
        return OmegaConf.create(state["config"])
    elif config_path:
        return OmegaConf.load(config_path)
    else:
        return OmegaConf.load("configs/default.yaml")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CSM-SAM on test set")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default="results/csmsam")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_nifti", action="store_true", help="Save prediction NIfTI files")
    parser.add_argument("--compare", type=str, default=None, help="Path to baselines results dir for comparison table")
    parser.add_argument("--tta", action="store_true", help="Horizontal-flip test-time augmentation")
    parser.add_argument("--fold", type=int, default=None, help="K-fold index (0-based). Requires --n_folds.")
    parser.add_argument("--n_folds", type=int, default=None, help="Total folds for K-fold CV.")
    args = parser.parse_args()

    # Setup
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg = load_checkpoint_cfg(args.checkpoint, args.config)
    if args.data_dir:
        cfg.data.data_dir = args.data_dir

    print("=" * 60)
    print(f"CSM-SAM Evaluation — split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    if args.tta:
        print("TTA: horizontal-flip enabled")
    if args.fold is not None:
        print(f"Fold: {args.fold} / {args.n_folds}")
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

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print("Model loaded.")

    # Build dataset
    dataset = HNTSMRGDataset(
        data_dir=cfg.data.data_dir,
        split=args.split,
        image_size=cfg.data.image_size,
    )

    # Optional K-fold selection. The cv util is being added by another agent;
    # fall back gracefully if unavailable.
    selected_indices: list[int] | None = None
    if args.fold is not None:
        if args.n_folds is None:
            raise ValueError("--fold requires --n_folds")
        try:
            from csmsam.utils.cv import kfold_split  # type: ignore
            patient_ids = [d.name for d in dataset.patient_dirs]
            _, eval_ids = kfold_split(patient_ids, n_folds=args.n_folds, fold=args.fold)
            pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}
            selected_indices = [pid_to_idx[p] for p in eval_ids if p in pid_to_idx]
            print(f"K-fold split: {len(selected_indices)} patients in fold {args.fold}")
        except ImportError:
            warnings.warn(
                "csmsam.utils.cv.kfold_split not available — falling back to full split."
            )
            selected_indices = None

    iter_indices = selected_indices if selected_indices is not None else list(range(len(dataset)))

    voxel_spacing = tuple(cfg.evaluation.get("voxel_spacing", [3.0, 1.0, 1.0]))

    # Inference
    print(f"\nRunning inference on {len(iter_indices)} patients...")
    per_patient_metrics = []
    per_patient_predictions = []

    for idx in tqdm(iter_indices, desc="Patients"):
        patient_data = dataset[idx]
        pid = patient_data["patient_id"]

        try:
            predictions, metrics = inference_patient(
                model, patient_data, device, args.threshold, voxel_spacing,
                tta=args.tta,
            )
            metrics["patient_id"] = pid
            per_patient_metrics.append(metrics)
            per_patient_predictions.append((patient_data, predictions, metrics))

            if args.save_nifti:
                save_predictions_nifti(predictions, patient_data, output_dir)

        except Exception as e:
            print(f"  Error processing {pid}: {e}")
            per_patient_metrics.append({
                "patient_id": pid,
                "agg_dsc": float("nan"),
                "dsc_gtvp": float("nan"),
                "dsc_gtvn": float("nan"),
                "hd95": float("inf"),
            })

    # Aggregate metrics
    agg_metrics = aggregate_metrics(per_patient_metrics)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"aggDSC  : {agg_metrics.get('agg_dsc_mean', 0):.4f} ± {agg_metrics.get('agg_dsc_std', 0):.4f}")
    print(f"DSC_GTVp: {agg_metrics.get('dsc_gtvp_mean', 0):.4f} ± {agg_metrics.get('dsc_gtvp_std', 0):.4f}")
    print(f"DSC_GTVn: {agg_metrics.get('dsc_gtvn_mean', 0):.4f} ± {agg_metrics.get('dsc_gtvn_std', 0):.4f}")
    print(f"HD95    : {agg_metrics.get('hd95_mean', 0):.2f} ± {agg_metrics.get('hd95_std', 0):.2f} mm")

    # Save metrics
    results = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "tta": args.tta,
        "fold": args.fold,
        "n_folds": args.n_folds,
        "aggregate": agg_metrics,
        "per_patient": per_patient_metrics,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved: {output_dir}/metrics.json")

    # Generate comparison table
    methods = {"CSM-SAM (ours)": agg_metrics}

    # Add published baselines
    methods["HNTS-MRG 2024 Winner"] = {
        "agg_dsc_mean": 0.727,
        "dsc_gtvp_mean": float("nan"),
        "dsc_gtvn_mean": float("nan"),
        "hd95_mean": float("nan"),
    }

    # Load external baseline results if provided
    if args.compare:
        compare_dir = Path(args.compare)
        for method_dir in sorted(compare_dir.iterdir()):
            metrics_file = method_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    baseline_results = json.load(f)
                methods[method_dir.name] = baseline_results.get("aggregate", {})

    table = format_results_table(methods)
    print(table)

    with open(output_dir / "metrics_table.txt", "w") as f:
        f.write(table)

    print(f"Table saved: {output_dir}/metrics_table.txt")
    print(f"\nNext step: python visualize.py --checkpoint {args.checkpoint}")


if __name__ == "__main__":
    main()
