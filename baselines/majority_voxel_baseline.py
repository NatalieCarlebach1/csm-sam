"""
Majority-Voxel (Dataset-Level Shape Prior) Baseline.

What this baseline does:
    Pools the pre-RT GTVp/GTVn masks across all TRAINING patients into a
    single voxel-wise UNION prior, then applies that same fixed prior as the
    prediction for EVERY mid-RT scan in the evaluation split. A patient-
    agnostic shape prior — no conditioning on imaging, no per-patient info.

How CSM-SAM differs:
    CSM-SAM conditions on THIS patient's pre-RT scan + mask, attends to
    image-matched memory, and produces a patient-specific prediction. This
    baseline shows that a dataset-level anatomical prior — "where tumors
    tend to be" — is fundamentally too coarse for per-patient segmentation.

Usage:
    python baselines/majority_voxel_baseline.py \
        --data_dir data/processed \
        --output_dir results/baselines/majority_voxel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from csmsam.datasets import HNTSMRGDataset
from csmsam.utils.metrics import aggregate_metrics, evaluate_patient


def build_union_prior(prior_dataset: HNTSMRGDataset) -> tuple[np.ndarray, np.ndarray]:
    """Build voxel-wise union masks over the prior split (GTVp, GTVn)."""
    prior_gtvp = None
    prior_gtvn = None
    for idx in tqdm(range(len(prior_dataset)), desc="Building union prior"):
        pd = prior_dataset[idx]
        p_gtvp = (pd["pre_masks_gtvp"] > 0.5).squeeze(1).numpy().astype(bool)
        p_gtvn = (pd["pre_masks_gtvn"] > 0.5).squeeze(1).numpy().astype(bool)
        if prior_gtvp is None:
            prior_gtvp = np.zeros_like(p_gtvp, dtype=bool)
            prior_gtvn = np.zeros_like(p_gtvn, dtype=bool)
        N = min(prior_gtvp.shape[0], p_gtvp.shape[0])
        prior_gtvp[:N] |= p_gtvp[:N]
        prior_gtvn[:N] |= p_gtvn[:N]
    return prior_gtvp.astype(np.float32), prior_gtvn.astype(np.float32)


def run_majority_voxel_baseline(
    data_dir: str,
    output_dir: str,
    split: str = "test",
    prior_split: str = "train",
    image_size: int = 1024,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Majority-Voxel Baseline (prior={prior_split}, eval={split})")
    print("=" * 60)

    prior_ds = HNTSMRGDataset(data_dir=data_dir, split=prior_split, image_size=image_size)
    prior_gtvp, prior_gtvn = build_union_prior(prior_ds)
    prior_D = prior_gtvp.shape[0]

    dataset = HNTSMRGDataset(data_dir=data_dir, split=split, image_size=image_size)
    per_patient_metrics = []

    for idx in tqdm(range(len(dataset)), desc="Patients"):
        patient_data = dataset[idx]
        pid = patient_data["patient_id"]

        try:
            gt_gtvp = (patient_data["mid_masks_gtvp"] > 0.5).squeeze(1).numpy()
            gt_gtvn = (patient_data["mid_masks_gtvn"] > 0.5).squeeze(1).numpy()
            N = gt_gtvp.shape[0]

            # Shape-match: tile or truncate prior along slice axis
            def _match(prior: np.ndarray) -> np.ndarray:
                if prior.shape[0] >= N:
                    return prior[:N]
                reps = (N + prior.shape[0] - 1) // prior.shape[0]
                return np.tile(prior, (reps, 1, 1))[:N]

            pred_gtvp = _match(prior_gtvp)
            pred_gtvn = _match(prior_gtvn)
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

    print(f"\nMajority-Voxel Baseline Results ({split}):")
    print(f"  aggDSC  : {agg.get('agg_dsc_mean', 0):.4f} +/- {agg.get('agg_dsc_std', 0):.4f}")
    print(f"  GTVp DSC: {agg.get('dsc_gtvp_mean', 0):.4f}")
    print(f"  GTVn DSC: {agg.get('dsc_gtvn_mean', 0):.4f}")

    results = {
        "aggregate": agg,
        "per_patient": per_patient_metrics,
        "config": {"prior_split": prior_split, "prior_slice_count": int(prior_D)},
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {output_dir}/metrics.json")
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/baselines/majority_voxel")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--prior_split", type=str, default="train")
    parser.add_argument("--image_size", type=int, default=1024)
    args = parser.parse_args()

    run_majority_voxel_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        prior_split=args.prior_split,
        image_size=args.image_size,
    )
