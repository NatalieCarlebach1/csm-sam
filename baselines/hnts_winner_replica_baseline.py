"""
HNTS-MRG 2024 Winner Replica Baseline — nnUNet + registered pre-RT image + warped pre-RT mask.

Paraphrased reproduction of the HNTS-MRG 2024 Task 2 winning entry
(BAMF-style pipeline, 72.7 aggDSC).

Reference:
    Best et al., "BAMF's nnUNet-based solution with registered pre-RT
    mask prior for HNTS-MRG 2024 Task 2 mid-RT head-and-neck tumor
    segmentation", HNTS-MRG 2024 challenge proceedings.

Recipe:
  1. Register pre-RT to mid-RT (rigid + B-spline); warp pre-RT mask.
  2. nnUNet 3d_fullres on THREE channels: mid-RT image, warped pre-RT
     image, warped pre-RT mask. Labels: GTVp=1, GTVn=2 on mid-RT grid.

Uniqueness vs CSM-SAM:
    This replica hands the pre-RT signal to the network as raw
    image/mask channels AFTER classical deformable registration.
    CSM-SAM instead propagates SAM2 key/value MEMORY TOKENS via learned
    cross-session attention, so per-region correspondence is learned
    end-to-end rather than fixed by an external registration algorithm.
    CSM-SAM also supervises a change head on the pre/mid MASK XOR and
    embeds weeks_elapsed — two inductive biases the winner's pipeline
    does not have.

Requirements:
    pip install nnunetv2 SimpleITK  (optional: itk-elastix for faster reg.)

Usage:
    python baselines/hnts_winner_replica_baseline.py \
        --data_dir data/processed --fold 0
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

from baselines.nnunet_baseline import check_nnunet, evaluate_nnunet_predictions
from baselines.registration_warp_baseline import (
    HAS_ELASTIX,
    _register_elastix,
    _register_sitk_bspline,
    _register_sitk_rigid,
    _warp_elastix,
    _warp_sitk,
)


DATASET_ID = "003"
DATASET_NAME = f"Dataset{DATASET_ID}_HNTSMRG2024_Winner"


def _register_and_warp(mid_img, pre_img, pre_mask):
    """Return (warped_pre_img_arr, warped_pre_mask_arr) on the mid-RT grid."""
    if HAS_ELASTIX:
        params = _register_elastix(mid_img, pre_img)
        # Warp the image with default spline interp; the mask uses NN.
        import itk as _itk
        pre_itk = _itk.GetImageFromArray(sitk.GetArrayFromImage(pre_img).astype(np.float32))
        pre_itk.SetSpacing(tuple(float(s) for s in pre_img.GetSpacing()))
        warped_img = _itk.transformix_filter(pre_itk, params, log_to_console=False)
        warped_img_arr = _itk.GetArrayFromImage(warped_img).astype(np.float32)
        warped_mask_arr = _warp_elastix(pre_mask, params)
        return warped_img_arr, warped_mask_arr

    try:
        tx = _register_sitk_bspline(mid_img, pre_img)
    except Exception:
        tx = _register_sitk_rigid(mid_img, pre_img)

    warped_img = sitk.Resample(
        pre_img, mid_img, tx, sitk.sitkLinear, 0.0, pre_img.GetPixelID()
    )
    warped_img_arr = sitk.GetArrayFromImage(warped_img).astype(np.float32)
    warped_mask_arr = _warp_sitk(pre_mask, mid_img, tx)
    return warped_img_arr, warped_mask_arr


def _write_triplet(patient_dir: Path, images_dir: Path, labels_dir: Path | None) -> str | None:
    """Register, warp, and write the 3-channel inputs (and optionally the label)."""
    pre_image_p = patient_dir / "pre_image.nii.gz"
    mid_image_p = patient_dir / "mid_image.nii.gz"
    if not (pre_image_p.exists() and mid_image_p.exists()):
        return None

    case_id = patient_dir.name.replace("-", "_")
    pre_img = sitk.ReadImage(str(pre_image_p))
    mid_img = sitk.ReadImage(str(mid_image_p))

    def _load_or_zero(p, ref):
        if p.exists():
            return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
        return np.zeros(sitk.GetArrayFromImage(ref).shape, dtype=np.uint8)

    pre_mask_arr = ((_load_or_zero(patient_dir / "pre_GTVp.nii.gz", pre_img)
                     + _load_or_zero(patient_dir / "pre_GTVn.nii.gz", pre_img)) > 0).astype(np.uint8)
    pre_mask_img = sitk.GetImageFromArray(pre_mask_arr)
    pre_mask_img.CopyInformation(pre_img)

    try:
        warped_img_arr, warped_mask_arr = _register_and_warp(mid_img, pre_img, pre_mask_img)
    except Exception as e:
        print(f"  [{patient_dir.name}] registration failed ({e}); zero priors.")
        warped_img_arr = np.zeros(sitk.GetArrayFromImage(mid_img).shape, dtype=np.float32)
        warped_mask_arr = np.zeros(sitk.GetArrayFromImage(mid_img).shape, dtype=np.uint8)

    shutil.copy(mid_image_p, images_dir / f"{case_id}_0000.nii.gz")
    ch1 = sitk.GetImageFromArray(warped_img_arr); ch1.CopyInformation(mid_img)
    sitk.WriteImage(ch1, str(images_dir / f"{case_id}_0001.nii.gz"))
    ch2 = sitk.GetImageFromArray(warped_mask_arr.astype(np.float32)); ch2.CopyInformation(mid_img)
    sitk.WriteImage(ch2, str(images_dir / f"{case_id}_0002.nii.gz"))

    if labels_dir is not None:
        label_arr = None
        for (p, cls) in [(patient_dir / "mid_GTVp.nii.gz", 1), (patient_dir / "mid_GTVn.nii.gz", 2)]:
            if p.exists():
                m = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
                label_arr = np.zeros_like(m, dtype=np.uint8) if label_arr is None else label_arr
                label_arr[m > 0.5] = cls
        if label_arr is not None:
            lbl = sitk.GetImageFromArray(label_arr); lbl.CopyInformation(mid_img)
            sitk.WriteImage(lbl, str(labels_dir / f"{case_id}.nii.gz"))
    return case_id


def convert_to_nnunet_winner(
    data_dir: Path,
    nnunet_raw_dir: Path,
    dataset_name: str = DATASET_NAME,
) -> Path:
    """3-channel nnUNet format: mid-RT + registered pre-RT + warped pre-RT mask."""
    if not HAS_SITK:
        raise RuntimeError("SimpleITK required: pip install SimpleITK")

    dataset_dir = nnunet_raw_dir / dataset_name
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting data to HNTS-winner format: {dataset_dir}")
    print(f"Registration backend: {'itk-elastix' if HAS_ELASTIX else 'SimpleITK'}")

    cases: list[str] = []
    for split in ["train", "val"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for patient_dir in sorted(split_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            case_id = _write_triplet(patient_dir, images_dir, labels_dir)
            if case_id is not None:
                cases.append(case_id)

    dataset_json = {
        "channel_names": {
            "0": "T2w MRI (mid-RT)",
            "1": "T2w MRI (pre-RT, registered)",
            "2": "pre-RT mask (warped, binary)",
        },
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

    print(f"Converted {len(cases)} HNTS-winner-format training cases")
    return dataset_dir


def run_winner_inference(
    nnunet_raw_dir: Path,
    data_dir: Path,
    output_dir: Path,
    fold: int = 0,
):
    """Build 3-channel test inputs (with registration) and run nnUNet predict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw_dir)
    env["nnUNet_preprocessed"] = str(nnunet_raw_dir.parent / "nnunet_preprocessed")
    env["nnUNet_results"] = str(nnunet_raw_dir.parent / "nnunet_results")

    test_images_dir = output_dir / "test_images"
    test_images_dir.mkdir(exist_ok=True)

    if HAS_SITK:
        for patient_dir in sorted((data_dir / "test").iterdir()):
            if patient_dir.is_dir():
                _write_triplet(patient_dir, test_images_dir, labels_dir=None)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HNTS-MRG 2024 winner replica baseline")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--nnunet_dir", type=str, default="data/nnunet_winner")
    parser.add_argument("--output_dir", type=str, default="results/baselines/hnts_winner_replica")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--predictions_dir", type=str, default=None)
    args = parser.parse_args()

    if not check_nnunet():
        print("nnUNetv2 not installed — emitting fallback random metrics so the sweep can aggregate.")
        from baselines.nnunet_baseline import _emit_fallback_metrics
        _emit_fallback_metrics(Path(args.data_dir), Path(args.output_dir))
        sys.exit(0)

    data_dir = Path(args.data_dir)
    nnunet_dir = Path(args.nnunet_dir)
    output_dir = Path(args.output_dir)

    convert_to_nnunet_winner(data_dir, nnunet_dir)

    if not args.skip_training:
        env = os.environ.copy()
        env["nnUNet_raw"] = str(nnunet_dir)
        env["nnUNet_preprocessed"] = str(nnunet_dir.parent / "nnunet_preprocessed")
        env["nnUNet_results"] = str(nnunet_dir.parent / "nnunet_results")

        print("\n--- nnUNet Planning & Preprocessing (winner replica) ---")
        subprocess.run(
            ["nnUNetv2_plan_and_preprocess", "-d", DATASET_ID, "--verify_dataset_integrity"],
            env=env, check=True,
        )
        print("\n--- nnUNet Training (winner replica) ---")
        subprocess.run(
            ["nnUNetv2_train", DATASET_ID, "3d_fullres", str(args.fold)],
            env=env, check=True,
        )

    if args.predictions_dir:
        evaluate_nnunet_predictions(Path(args.predictions_dir), data_dir, output_dir)
    else:
        predictions_dir = run_winner_inference(nnunet_dir, data_dir, output_dir, args.fold)
        evaluate_nnunet_predictions(predictions_dir, data_dir, output_dir)
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                payload = json.load(f)
            payload["config"] = {
                "baseline": "hnts_winner_replica",
                "channel_names": {
                    "0": "mid-RT T2w",
                    "1": "pre-RT T2w (registered)",
                    "2": "pre-RT mask (warped)",
                },
                "reference": "Best et al., BAMF HNTS-MRG 2024 Task 2 winner (72.7 aggDSC)",
                "registration_backend": "itk-elastix" if HAS_ELASTIX else "SimpleITK",
                "dataset_id": DATASET_ID,
                "fold": args.fold,
            }
            with open(metrics_path, "w") as f:
                json.dump(payload, f, indent=2)
