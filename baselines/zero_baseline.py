"""
Zero-Prediction Baseline.

What this baseline does:
    Predicts an all-zero mask for every mid-RT slice. Establishes the absolute
    metric floor — aggDSC == 0 whenever the GT is non-empty.

How CSM-SAM differs:
    CSM-SAM actually produces spatial predictions grounded in the pre-RT
    memory bank; this baseline produces none. Exists to verify the evaluation
    pipeline and set an unconditional lower bound against which every learned
    method (including CSM-SAM) must improve.

Usage:
    python baselines/zero_baseline.py \
        --data_dir data/processed \
        --output_dir results/baselines/zero
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from csmsam.datasets import HNTSMRGDataset
from csmsam.utils.metrics import aggregate_metrics, evaluate_patient


def run_zero_baseline(
    data_dir: str,
    output_dir: str,
    split: str = "test",
    image_size: int = 1024,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Zero Baseline (all-zero prediction)")
    print(f"Split: {split}")
    print("=" * 60)

    dataset = HNTSMRGDataset(data_dir=data_dir, split=split, image_size=image_size)
    per_patient_metrics = []

    for idx in tqdm(range(len(dataset)), desc="Patients"):
        patient_data = dataset[idx]
        pid = patient_data["patient_id"]

        try:
            gt_gtvp = (patient_data["mid_masks_gtvp"] > 0.5).squeeze(1).numpy()
            gt_gtvn = (patient_data["mid_masks_gtvn"] > 0.5).squeeze(1).numpy()

            pred_zero = np.zeros_like(gt_gtvp, dtype=np.float32)

            metrics = evaluate_patient(
                pred_masks=pred_zero,
                pred_gtvp=pred_zero,
                pred_gtvn=pred_zero,
                target_gtvp=gt_gtvp,
                target_gtvn=gt_gtvn,
            )
            metrics["patient_id"] = pid
            per_patient_metrics.append(metrics)

        except Exception as e:
            print(f"  Error on {pid}: {e}")

    agg = aggregate_metrics(per_patient_metrics)

    print(f"\nZero Baseline Results ({split}):")
    print(f"  aggDSC  : {agg.get('agg_dsc_mean', 0):.4f} +/- {agg.get('agg_dsc_std', 0):.4f}")
    print(f"  GTVp DSC: {agg.get('dsc_gtvp_mean', 0):.4f}")
    print(f"  GTVn DSC: {agg.get('dsc_gtvn_mean', 0):.4f}")

    results = {"aggregate": agg, "per_patient": per_patient_metrics}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {output_dir}/metrics.json")
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/baselines/zero")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--image_size", type=int, default=1024)
    args = parser.parse_args()

    run_zero_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        image_size=args.image_size,
    )
