"""
CSM-SAM Test / Evaluation Script.

Runs inference on the test set, computes metrics, and saves predictions.

Usage:
    python test.py --checkpoint checkpoints/csmsam_default/best.pth
    python test.py --checkpoint checkpoints/best.pth --split val
    python test.py --checkpoint checkpoints/best.pth --compare baselines/results/
"""

from __future__ import annotations

import argparse
import json
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


@torch.no_grad()
def inference_patient(
    model: CSMSAM,
    patient_data: dict,
    device: str,
    threshold: float = 0.5,
    voxel_spacing: tuple = (3.0, 1.0, 1.0),
) -> tuple[dict, dict]:
    """
    Run full 3D inference on one patient.

    Returns:
        predictions: dict of numpy volume predictions
        metrics:     dict of evaluation metrics
    """
    model.eval()

    pre_images = patient_data["pre_images"].to(device)   # (N, 3, H, W)
    mid_images = patient_data["mid_images"].to(device)   # (N, 3, H, W)
    pre_masks = patient_data["pre_masks"].to(device)     # (N, 1, H, W)
    weeks = patient_data["weeks_elapsed"]
    weeks_t = torch.tensor([int(weeks)], dtype=torch.long, device=device)

    N = pre_images.shape[0]

    # Encode pre-RT memory
    M_pre = model.encode_pre_rt(
        pre_images.unsqueeze(0),
        pre_masks.unsqueeze(0),
    )  # (1, N_mem, C)

    # Slice-by-slice mid-RT segmentation
    model.reset_mid_session_memory()

    pred_masks_list = []
    change_maps_list = []
    gate_vals_list = []

    for i in tqdm(range(N), desc="Slices", leave=False):
        mid_slice = mid_images[i].unsqueeze(0)
        pre_slice = pre_images[i].unsqueeze(0)

        out = model(
            mid_images=mid_slice,
            M_pre=M_pre,
            pre_images=pre_slice,
            weeks_elapsed=weeks_t,
            return_change_map=True,
        )

        pred_masks_list.append(out["masks"].squeeze(0).cpu())
        if "change_map" in out:
            change_maps_list.append(out["change_map"].squeeze(0).cpu())
        if "gate_vals" in out:
            gate_vals_list.append(out["gate_vals"].squeeze(0).cpu())

    pred_masks = torch.stack(pred_masks_list)   # (N, 1, H, W)
    pred_binary = (torch.sigmoid(pred_masks) > threshold).squeeze(1).numpy()  # (N, H, W)

    # For GTVp / GTVn: use same prediction (can be refined with separate heads)
    pred_gtvp = pred_binary
    pred_gtvn = pred_binary

    # Ground truth
    gt_gtvp = (patient_data["mid_masks_gtvp"] > 0.5).squeeze(1).numpy()
    gt_gtvn = (patient_data["mid_masks_gtvn"] > 0.5).squeeze(1).numpy()

    metrics = evaluate_patient(
        pred_masks=pred_binary,
        pred_gtvp=pred_gtvp,
        pred_gtvn=pred_gtvn,
        target_gtvp=gt_gtvp,
        target_gtvn=gt_gtvn,
        voxel_spacing=voxel_spacing,
    )

    predictions = {
        "masks": pred_masks,
        "masks_binary": pred_binary,
        "change_map": torch.stack(change_maps_list) if change_maps_list else None,
        "gate_vals": torch.stack(gate_vals_list) if gate_vals_list else None,
    }

    return predictions, metrics


def save_predictions_nifti(
    predictions: dict,
    patient_data: dict,
    output_dir: Path,
):
    """Save binary prediction masks as NIfTI files."""
    if not HAS_SITK:
        print("  Warning: SimpleITK not available, skipping NIfTI save")
        return

    pid = patient_data["patient_id"]
    patient_out = output_dir / "predictions" / pid
    patient_out.mkdir(parents=True, exist_ok=True)

    pred_arr = predictions["masks_binary"].astype(np.uint8)
    img = sitk.GetImageFromArray(pred_arr)
    img.SetSpacing((1.0, 1.0, 3.0))
    sitk.WriteImage(img, str(patient_out / "pred_mid_mask.nii.gz"))


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

    voxel_spacing = tuple(cfg.evaluation.get("voxel_spacing", [3.0, 1.0, 1.0]))

    # Inference
    print(f"\nRunning inference on {len(dataset)} patients...")
    per_patient_metrics = []
    per_patient_predictions = []

    for idx in tqdm(range(len(dataset)), desc="Patients"):
        patient_data = dataset[idx]
        pid = patient_data["patient_id"]

        try:
            predictions, metrics = inference_patient(
                model, patient_data, device, args.threshold, voxel_spacing
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
