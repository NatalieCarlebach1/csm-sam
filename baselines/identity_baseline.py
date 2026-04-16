"""
Identity Propagation Baseline.

What this baseline does:
    Uses the (registered) pre-RT mask verbatim as the mid-RT prediction.
    No learning, no adaptation — just copies pre-RT GTVp/GTVn masks onto the
    mid-RT scan.

How CSM-SAM differs:
    CSM-SAM uses cross-session memory to ADAPT the pre-RT segmentation to the
    mid-RT scan, modelling treatment-induced shrinkage/shape-change instead of
    asserting the tumor is unchanged. Identity propagation is the key "no-
    learning" floor that proves cross-session memory must DO something beyond
    copying.

Usage:
    python baselines/identity_baseline.py \
        --data_dir data/processed \
        --output_dir results/baselines/identity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from csmsam.datasets import HNTSMRGDataset
from csmsam.utils.metrics import aggregate_metrics, evaluate_patient


def run_identity_baseline(
    data_dir: str,
    output_dir: str,
    split: str = "test",
    image_size: int = 1024,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Identity Propagation Baseline (pre-RT mask == mid-RT pred)")
    print(f"Split: {split}")
    print("=" * 60)

    dataset = HNTSMRGDataset(data_dir=data_dir, split=split, image_size=image_size)
    per_patient_metrics = []

    for idx in tqdm(range(len(dataset)), desc="Patients"):
        patient_data = dataset[idx]
        pid = patient_data["patient_id"]

        try:
            pre_gtvp = (patient_data["pre_masks_gtvp"] > 0.5).squeeze(1).numpy()
            pre_gtvn = (patient_data["pre_masks_gtvn"] > 0.5).squeeze(1).numpy()
            gt_gtvp = (patient_data["mid_masks_gtvp"] > 0.5).squeeze(1).numpy()
            gt_gtvn = (patient_data["mid_masks_gtvn"] > 0.5).squeeze(1).numpy()

            # Align slice count (pre and mid may differ)
            N = min(pre_gtvp.shape[0], gt_gtvp.shape[0])
            pred_gtvp = pre_gtvp[:N].astype(np.float32)
            pred_gtvn = pre_gtvn[:N].astype(np.float32)
            gt_gtvp = gt_gtvp[:N]
            gt_gtvn = gt_gtvn[:N]
            pred_combined = ((pred_gtvp + pred_gtvn) > 0).astype(np.float32)

            metrics = evaluate_patient(
                pred_masks=pred_combined,
                pred_gtvp=pred_gtvp,
                pred_gtvn=pred_gtvn,
                target_gtvp=gt_gtvp,
                target_gtvn=gt_gtvn,
            )
            metrics["patient_id"] = pid
            per_patient_metrics.append(metrics)

        except Exception as e:
            print(f"  Error on {pid}: {e}")

    agg = aggregate_metrics(per_patient_metrics)

    print(f"\nIdentity Baseline Results ({split}):")
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
    parser.add_argument("--output_dir", type=str, default="results/baselines/identity")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--image_size", type=int, default=1024)
    args = parser.parse_args()

    run_identity_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        image_size=args.image_size,
    )
