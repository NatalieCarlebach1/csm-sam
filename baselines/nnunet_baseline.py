"""
nnUNet Baseline for HNTS-MRG 2024 mid-RT segmentation.

nnUNet is the strongest traditional baseline and is used by many HNTS-MRG
challenge methods. This script:
  1. Converts HNTS-MRG preprocessed data to nnUNet format
  2. Runs nnUNet planning and preprocessing
  3. Trains nnUNet 3d_fullres model
  4. Runs inference and evaluation

Requirements:
    pip install nnunetv2

Usage:
    python baselines/nnunet_baseline.py \
        --data_dir data/processed \
        --nnunet_dir data/nnunet \
        --output_dir results/baselines/nnunet \
        --task_name HNTSMRG2024

Note: This is a script wrapper around nnUNetv2 CLI commands.
For nnUNetv2 installation: https://github.com/MIC-DKFZ/nnUNet
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


DATASET_ID = "001"
DATASET_NAME = f"Dataset{DATASET_ID}_HNTSMRG2024"


def _emit_fallback_metrics(data_dir: Path, output_dir: Path) -> None:
    """Write random-prediction metrics.json so the sweep can aggregate — same convention other baselines use when their library is missing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from csmsam.datasets import HNTSMRGDataset
        from csmsam.utils.metrics import evaluate_patient, aggregate_metrics
        ds = HNTSMRGDataset(data_dir=str(data_dir), split="test", image_size=128)
        per_patient = []
        for i in range(len(ds)):
            d = ds[i]
            mid_gtvp = (d["mid_masks_gtvp"] > 0.5).squeeze(1).numpy()
            mid_gtvn = (d["mid_masks_gtvn"] > 0.5).squeeze(1).numpy()
            pred = (np.random.rand(*mid_gtvp.shape) > 0.95).astype(np.float32)
            m = evaluate_patient(pred_masks=pred, pred_gtvp=pred, pred_gtvn=pred,
                                 target_gtvp=mid_gtvp, target_gtvn=mid_gtvn)
            m["patient_id"] = d["patient_id"]
            per_patient.append(m)
        agg = aggregate_metrics(per_patient)
    except Exception as exc:
        agg = {"note": f"fallback (nnUNet missing); dataset load also failed: {exc}"}
        per_patient = []
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"aggregate": agg, "per_patient": per_patient, "fallback": True}, f, indent=2)


def check_nnunet():
    """Check if nnUNetv2 is installed."""
    try:
        result = subprocess.run(["nnUNetv2_plan_and_preprocess", "--help"],
                               capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def convert_to_nnunet_format(
    data_dir: Path,
    nnunet_raw_dir: Path,
    dataset_name: str = DATASET_NAME,
):
    """
    Convert preprocessed HNTS-MRG data to nnUNetv2 dataset format.

    nnUNetv2 expects:
        nnUNet_raw/
            DatasetXXX_Name/
                imagesTr/
                    case_0000_0000.nii.gz  (channel 0)
                labelsTr/
                    case_0000.nii.gz
                dataset.json
    """
    if not HAS_SITK:
        print("ERROR: SimpleITK required: pip install SimpleITK")
        return

    dataset_dir = nnunet_raw_dir / dataset_name
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting data to nnUNet format: {dataset_dir}")

    cases = []
    for split in ["train", "val"]:  # nnUNet trains on train+val internally
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        for patient_dir in sorted(split_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            pid = patient_dir.name

            mid_image = patient_dir / "mid_image.nii.gz"
            mid_gtvp = patient_dir / "mid_GTVp.nii.gz"
            mid_gtvn = patient_dir / "mid_GTVn.nii.gz"

            if not mid_image.exists():
                continue

            case_id = pid.replace("-", "_")
            cases.append(case_id)

            # Copy mid-RT image as channel 0
            shutil.copy(mid_image, images_dir / f"{case_id}_0000.nii.gz")

            # Merge GTVp + GTVn into one label (GTVp=1, GTVn=2)
            label_arr = None
            if mid_gtvp.exists():
                gtvp = sitk.GetArrayFromImage(sitk.ReadImage(str(mid_gtvp)))
                if label_arr is None:
                    label_arr = np.zeros_like(gtvp, dtype=np.uint8)
                label_arr[gtvp > 0.5] = 1
            if mid_gtvn.exists():
                gtvn = sitk.GetArrayFromImage(sitk.ReadImage(str(mid_gtvn)))
                if label_arr is None:
                    label_arr = np.zeros_like(gtvn, dtype=np.uint8)
                label_arr[gtvn > 0.5] = 2

            if label_arr is not None:
                label_img = sitk.GetImageFromArray(label_arr)
                label_img.SetSpacing((3.0, 1.0, 1.0))
                sitk.WriteImage(label_img, str(labels_dir / f"{case_id}.nii.gz"))

    # dataset.json
    dataset_json = {
        "channel_names": {"0": "T2w MRI (mid-RT)"},
        "labels": {
            "background": 0,
            "GTVp": 1,
            "GTVn": 2,
        },
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "dataset_name": dataset_name,
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Converted {len(cases)} training cases")
    print(f"Dataset JSON: {dataset_dir}/dataset.json")
    return dataset_dir


def run_nnunet_training(nnunet_raw_dir: Path, fold: int = 0):
    """Run nnUNetv2 planning, preprocessing, and training."""
    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw_dir)
    env["nnUNet_preprocessed"] = str(nnunet_raw_dir.parent / "nnunet_preprocessed")
    env["nnUNet_results"] = str(nnunet_raw_dir.parent / "nnunet_results")

    print("\n--- nnUNet Planning & Preprocessing ---")
    subprocess.run([
        "nnUNetv2_plan_and_preprocess",
        "-d", DATASET_ID,
        "--verify_dataset_integrity",
    ], env=env, check=True)

    print("\n--- nnUNet Training ---")
    subprocess.run([
        "nnUNetv2_train",
        DATASET_ID,
        "3d_fullres",
        str(fold),
    ], env=env, check=True)


def run_nnunet_inference(
    nnunet_raw_dir: Path,
    data_dir: Path,
    output_dir: Path,
    fold: int = 0,
):
    """Run nnUNetv2 inference on test set."""
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw_dir)
    env["nnUNet_preprocessed"] = str(nnunet_raw_dir.parent / "nnunet_preprocessed")
    env["nnUNet_results"] = str(nnunet_raw_dir.parent / "nnunet_results")

    # Prepare test images
    test_images_dir = output_dir / "test_images"
    test_images_dir.mkdir(exist_ok=True)

    if HAS_SITK:
        for patient_dir in sorted((data_dir / "test").iterdir()):
            mid_image = patient_dir / "mid_image.nii.gz"
            if mid_image.exists():
                case_id = patient_dir.name.replace("-", "_")
                shutil.copy(mid_image, test_images_dir / f"{case_id}_0000.nii.gz")

    predictions_dir = output_dir / "predictions"
    subprocess.run([
        "nnUNetv2_predict",
        "-i", str(test_images_dir),
        "-o", str(predictions_dir),
        "-d", DATASET_ID,
        "-c", "3d_fullres",
        "-f", str(fold),
    ], env=env, check=True)

    print(f"Predictions saved to {predictions_dir}")
    return predictions_dir


def evaluate_nnunet_predictions(
    predictions_dir: Path,
    data_dir: Path,
    output_dir: Path,
):
    """Compare nnUNet predictions to ground truth and compute metrics."""
    from csmsam.utils.metrics import evaluate_patient, aggregate_metrics

    per_patient_metrics = []

    for pred_file in sorted(predictions_dir.glob("*.nii.gz")):
        case_id = pred_file.stem.replace(".nii", "")
        patient_id = case_id

        # Find ground truth
        gt_dir = data_dir / "test" / patient_id
        if not gt_dir.exists():
            # Try with dashes
            patient_id = case_id.replace("_", "-")
            gt_dir = data_dir / "test" / patient_id

        if not gt_dir.exists() or not HAS_SITK:
            continue

        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_file)))
        pred_gtvp = (pred == 1).astype(np.float32)
        pred_gtvn = (pred == 2).astype(np.float32)

        gt_gtvp_path = gt_dir / "mid_GTVp.nii.gz"
        gt_gtvn_path = gt_dir / "mid_GTVn.nii.gz"

        gt_gtvp = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_gtvp_path))) > 0.5 if gt_gtvp_path.exists() else np.zeros_like(pred_gtvp)
        gt_gtvn = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_gtvn_path))) > 0.5 if gt_gtvn_path.exists() else np.zeros_like(pred_gtvn)

        metrics = evaluate_patient(
            pred_masks=(pred_gtvp + pred_gtvn) > 0,
            pred_gtvp=pred_gtvp,
            pred_gtvn=pred_gtvn,
            target_gtvp=gt_gtvp,
            target_gtvn=gt_gtvn,
        )
        metrics["patient_id"] = patient_id
        per_patient_metrics.append(metrics)

    agg = aggregate_metrics(per_patient_metrics)
    results = {"aggregate": agg, "per_patient": per_patient_metrics}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nnnUNet Results:")
    print(f"  aggDSC  : {agg.get('agg_dsc_mean', 0):.4f}")
    print(f"  GTVp DSC: {agg.get('dsc_gtvp_mean', 0):.4f}")
    print(f"  GTVn DSC: {agg.get('dsc_gtvn_mean', 0):.4f}")
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nnUNet Baseline for HNTS-MRG 2024")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--nnunet_dir", type=str, default="data/nnunet")
    parser.add_argument("--output_dir", type=str, default="results/baselines/nnunet")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--predictions_dir", type=str, default=None, help="Existing predictions dir")
    parser.add_argument("--split", type=str, default="test", help="Unused — kept for sweep compat")
    parser.add_argument("--image_size", type=int, default=128, help="Unused — nnUNet plans its own patches")
    args = parser.parse_args()

    if not check_nnunet():
        print("nnUNetv2 not installed — emitting fallback random metrics so the sweep can aggregate.")
        print("To run the real baseline: pip install nnunetv2 (https://github.com/MIC-DKFZ/nnUNet)")
        _emit_fallback_metrics(Path(args.data_dir), Path(args.output_dir))
        sys.exit(0)

    data_dir = Path(args.data_dir)
    nnunet_dir = Path(args.nnunet_dir)
    output_dir = Path(args.output_dir)

    # Convert data
    dataset_dir = convert_to_nnunet_format(data_dir, nnunet_dir)

    # Train
    if not args.skip_training:
        run_nnunet_training(nnunet_dir, fold=args.fold)

    # Inference + evaluation
    if args.predictions_dir:
        evaluate_nnunet_predictions(Path(args.predictions_dir), data_dir, output_dir)
    else:
        predictions_dir = run_nnunet_inference(nnunet_dir, data_dir, output_dir, args.fold)
        evaluate_nnunet_predictions(predictions_dir, data_dir, output_dir)
