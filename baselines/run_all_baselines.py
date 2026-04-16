"""
Run all baselines and generate a comparison table.

Usage:
    python baselines/run_all_baselines.py \
        --data_dir data/processed \
        --sam2_checkpoint checkpoints/sam2/sam2.1_hiera_large.pt \
        --output_dir results/baselines

    # After training CSM-SAM:
    python baselines/run_all_baselines.py \
        --data_dir data/processed \
        --csmsam_checkpoint checkpoints/csmsam_default/best.pth \
        --output_dir results/baselines
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from csmsam.utils.metrics import format_results_table


def load_metrics(results_dir: Path) -> dict | None:
    """Load metrics.json from a results directory."""
    metrics_file = results_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    with open(metrics_file) as f:
        data = json.load(f)
    return data.get("aggregate", {})


def main():
    parser = argparse.ArgumentParser(description="Run all baselines and compare")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/baselines")
    parser.add_argument("--sam2_checkpoint", type=str, default="checkpoints/sam2/sam2.1_hiera_large.pt")
    parser.add_argument("--csmsam_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--skip_medsam2", action="store_true")
    parser.add_argument("--skip_nnunet", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CSM-SAM Baseline Runner")
    print("=" * 60)

    methods = {}

    # ─── Published baselines (from HNTS-MRG 2024 paper) ─────────────────────
    # These are the official challenge results from Wahid et al., 2024
    methods["HNTS-MRG 2024 Winner†"] = {
        "agg_dsc_mean": 0.727,
        "dsc_gtvp_mean": float("nan"),
        "dsc_gtvn_mean": float("nan"),
        "hd95_mean": float("nan"),
    }
    methods["Challenge Mean†"] = {
        "agg_dsc_mean": 0.688,
        "dsc_gtvp_mean": float("nan"),
        "dsc_gtvn_mean": float("nan"),
        "hd95_mean": float("nan"),
    }

    # ─── MedSAM2 Baseline ────────────────────────────────────────────────────
    medsam2_dir = output_dir / "medsam2"
    if not args.skip_medsam2:
        print("\n[1/2] Running MedSAM2 baseline...")
        from baselines.medsam2_baseline import run_medsam2_baseline
        run_medsam2_baseline(
            data_dir=args.data_dir,
            sam2_checkpoint=args.sam2_checkpoint,
            output_dir=str(medsam2_dir),
            split=args.split,
            device=args.device,
        )

    medsam2_metrics = load_metrics(medsam2_dir)
    if medsam2_metrics:
        methods["MedSAM2 (within-session)"] = medsam2_metrics

    # ─── nnUNet Baseline ─────────────────────────────────────────────────────
    nnunet_dir = output_dir / "nnunet"
    if not args.skip_nnunet:
        print("\n[2/2] Running nnUNet baseline...")
        try:
            from baselines.nnunet_baseline import (
                check_nnunet,
                convert_to_nnunet_format,
                run_nnunet_training,
                run_nnunet_inference,
                evaluate_nnunet_predictions,
            )
            if check_nnunet():
                nnunet_raw_dir = Path("data/nnunet")
                convert_to_nnunet_format(Path(args.data_dir), nnunet_raw_dir)
                run_nnunet_training(nnunet_raw_dir)
                predictions_dir = run_nnunet_inference(nnunet_raw_dir, Path(args.data_dir), nnunet_dir)
                evaluate_nnunet_predictions(predictions_dir, Path(args.data_dir), nnunet_dir)
            else:
                print("nnUNet not installed, skipping. pip install nnunetv2")
        except Exception as e:
            print(f"nnUNet baseline failed: {e}")

    nnunet_metrics = load_metrics(nnunet_dir)
    if nnunet_metrics:
        methods["nnUNet 3d_fullres"] = nnunet_metrics

    # ─── CSM-SAM (Ours) ──────────────────────────────────────────────────────
    if args.csmsam_checkpoint:
        print("\nLoading CSM-SAM results...")
        csmsam_dir = output_dir.parent / "csmsam"
        csmsam_metrics = load_metrics(csmsam_dir)
        if csmsam_metrics is None:
            print("CSM-SAM metrics not found. Run: python test.py --checkpoint ...")
        else:
            methods["CSM-SAM (Ours)"] = csmsam_metrics

    # ─── Generate comparison table ───────────────────────────────────────────
    print("\n")
    table = format_results_table(methods)
    print(table)

    table_path = output_dir / "comparison_table.txt"
    with open(table_path, "w") as f:
        f.write(table)
        f.write("\n† Published results from HNTS-MRG 2024 challenge paper.\n")
        f.write("  Wahid et al., Head and Neck Tumor Segmentation for MR-Guided Radiotherapy. arXiv 2024.\n")

    print(f"Table saved: {table_path}")

    # Also save raw numbers
    all_results = {name: metrics for name, metrics in methods.items()}
    with open(output_dir / "all_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: None if np.isnan(x) else float(x))

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
