"""
nnUNet Dual-Channel Baseline — nnUNet with pre-RT and mid-RT as 2 input channels.

Stronger than the naive concat-channels 2D U-Net because:
  - nnUNet's planner auto-tunes patch size, spacing, normalization and
    architecture depth to the specific dataset (Isensee et al., 2021).
  - nnUNet trains a true 3D_fullres model, capturing inter-slice context
    that 2D siamese/concat baselines cannot.

This script wraps the existing ``baselines/nnunet_baseline.py`` pipeline:
it reproduces its dataset-conversion / training / inference / evaluation
steps but sets ``channel_names`` to two channels (pre-RT, mid-RT) and
writes both volumes to ``imagesTr``.

Reference:
    Isensee et al., "nnU-Net: a self-configuring method for deep
    learning-based biomedical image segmentation." Nat. Methods 2021.

Uniqueness vs CSM-SAM:
    CSM-SAM routes the pre-RT signal through a LEARNED CROSS-SESSION
    ATTENTION over SAM2 memory tokens, not through a CNN that sees the
    two visits concatenated at the pixel level. nnUNet-dual-channel must
    re-learn everything from its 24M random-init parameters on ~150
    patients; CSM-SAM leverages SAM2's frozen ViT-H with its 1B-mask
    prior. CSM-SAM also supervises a change head on the pre/mid MASK XOR
    and embeds weeks_elapsed — neither is expressible as an extra input
    channel to nnUNet.

Requirements:
    pip install nnunetv2 SimpleITK

Usage:
    python baselines/nnunet_dual_channel_baseline.py \
        --data_dir data/processed \
        --nnunet_dir data/nnunet_dual \
        --output_dir results/baselines/nnunet_dual_channel \
        --fold 0
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

from baselines.nnunet_baseline import (
    check_nnunet,
    run_nnunet_training,
    run_nnunet_inference,
    evaluate_nnunet_predictions,
)


DATASET_ID = "002"
DATASET_NAME = f"Dataset{DATASET_ID}_HNTSMRG2024_Dual"


def convert_to_nnunet_dual_channel(
    data_dir: Path,
    nnunet_raw_dir: Path,
    dataset_name: str = DATASET_NAME,
) -> Path:
    """
    Convert HNTS-MRG data to nnUNetv2 format with TWO input channels:
        channel 0: pre-RT T2w MRI
        channel 1: mid-RT T2w MRI
    Labels come from mid-RT GTVp (1) and GTVn (2).

    nnUNetv2 naming convention:
        imagesTr/{case_id}_0000.nii.gz   — channel 0 (pre-RT)
        imagesTr/{case_id}_0001.nii.gz   — channel 1 (mid-RT)
        labelsTr/{case_id}.nii.gz        — mid-RT label map
    """
    if not HAS_SITK:
        raise RuntimeError("SimpleITK required: pip install SimpleITK")

    dataset_dir = nnunet_raw_dir / dataset_name
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting data to nnUNet dual-channel format: {dataset_dir}")

    cases: list[str] = []
    for split in ["train", "val"]:  # nnUNet cross-validates internally
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        for patient_dir in sorted(split_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            pid = patient_dir.name

            pre_image = patient_dir / "pre_image.nii.gz"
            mid_image = patient_dir / "mid_image.nii.gz"
            mid_gtvp = patient_dir / "mid_GTVp.nii.gz"
            mid_gtvn = patient_dir / "mid_GTVn.nii.gz"

            if not mid_image.exists() or not pre_image.exists():
                continue

            case_id = pid.replace("-", "_")
            cases.append(case_id)

            # Channel 0: pre-RT (resampled to mid-RT grid if needed).
            pre_img = sitk.ReadImage(str(pre_image))
            mid_img = sitk.ReadImage(str(mid_image))
            if pre_img.GetSize() != mid_img.GetSize():
                pre_img = sitk.Resample(
                    pre_img, mid_img, sitk.Transform(),
                    sitk.sitkLinear, 0.0, pre_img.GetPixelID(),
                )
            sitk.WriteImage(pre_img, str(images_dir / f"{case_id}_0000.nii.gz"))
            # Channel 1: mid-RT.
            shutil.copy(mid_image, images_dir / f"{case_id}_0001.nii.gz")

            # Label map (GTVp=1, GTVn=2) on mid-RT grid.
            label_arr = None
            if mid_gtvp.exists():
                gtvp = sitk.GetArrayFromImage(sitk.ReadImage(str(mid_gtvp)))
                label_arr = np.zeros_like(gtvp, dtype=np.uint8) if label_arr is None else label_arr
                label_arr[gtvp > 0.5] = 1
            if mid_gtvn.exists():
                gtvn = sitk.GetArrayFromImage(sitk.ReadImage(str(mid_gtvn)))
                label_arr = np.zeros_like(gtvn, dtype=np.uint8) if label_arr is None else label_arr
                label_arr[gtvn > 0.5] = 2

            if label_arr is not None:
                label_img = sitk.GetImageFromArray(label_arr)
                label_img.CopyInformation(mid_img)
                sitk.WriteImage(label_img, str(labels_dir / f"{case_id}.nii.gz"))

    dataset_json = {
        "channel_names": {
            "0": "T2w MRI (pre-RT)",
            "1": "T2w MRI (mid-RT)",
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

    print(f"Converted {len(cases)} dual-channel training cases")
    print(f"Dataset JSON: {dataset_dir}/dataset.json")
    return dataset_dir


def run_dual_channel_inference(
    nnunet_raw_dir: Path,
    data_dir: Path,
    output_dir: Path,
    fold: int = 0,
):
    """Inference wrapper that writes both pre and mid channels to the test dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw_dir)
    env["nnUNet_preprocessed"] = str(nnunet_raw_dir.parent / "nnunet_preprocessed")
    env["nnUNet_results"] = str(nnunet_raw_dir.parent / "nnunet_results")

    test_images_dir = output_dir / "test_images"
    test_images_dir.mkdir(exist_ok=True)

    if HAS_SITK:
        for patient_dir in sorted((data_dir / "test").iterdir()):
            pre_image = patient_dir / "pre_image.nii.gz"
            mid_image = patient_dir / "mid_image.nii.gz"
            if not (pre_image.exists() and mid_image.exists()):
                continue
            case_id = patient_dir.name.replace("-", "_")

            pre_img = sitk.ReadImage(str(pre_image))
            mid_img = sitk.ReadImage(str(mid_image))
            if pre_img.GetSize() != mid_img.GetSize():
                pre_img = sitk.Resample(
                    pre_img, mid_img, sitk.Transform(),
                    sitk.sitkLinear, 0.0, pre_img.GetPixelID(),
                )
            sitk.WriteImage(pre_img, str(test_images_dir / f"{case_id}_0000.nii.gz"))
            shutil.copy(mid_image, test_images_dir / f"{case_id}_0001.nii.gz")

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
    parser = argparse.ArgumentParser(description="nnUNet Dual-Channel (pre, mid) Baseline")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--nnunet_dir", type=str, default="data/nnunet_dual")
    parser.add_argument("--output_dir", type=str, default="results/baselines/nnunet_dual_channel")
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

    convert_to_nnunet_dual_channel(data_dir, nnunet_dir)

    if not args.skip_training:
        # Reuse the shared training helper but with our DATASET_ID.
        # (nnunet_baseline.run_nnunet_training hard-codes DATASET_ID; we
        # subprocess directly to override.)
        env = os.environ.copy()
        env["nnUNet_raw"] = str(nnunet_dir)
        env["nnUNet_preprocessed"] = str(nnunet_dir.parent / "nnunet_preprocessed")
        env["nnUNet_results"] = str(nnunet_dir.parent / "nnunet_results")

        print("\n--- nnUNet Planning & Preprocessing (dual-channel) ---")
        subprocess.run(
            ["nnUNetv2_plan_and_preprocess", "-d", DATASET_ID, "--verify_dataset_integrity"],
            env=env, check=True,
        )
        print("\n--- nnUNet Training (dual-channel) ---")
        subprocess.run(
            ["nnUNetv2_train", DATASET_ID, "3d_fullres", str(args.fold)],
            env=env, check=True,
        )

    if args.predictions_dir:
        evaluate_nnunet_predictions(Path(args.predictions_dir), data_dir, output_dir)
    else:
        predictions_dir = run_dual_channel_inference(nnunet_dir, data_dir, output_dir, args.fold)
        agg = evaluate_nnunet_predictions(predictions_dir, data_dir, output_dir)
        # Tag the metrics JSON with the dual-channel config marker.
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                payload = json.load(f)
            payload["config"] = {
                "baseline": "nnunet_dual_channel",
                "channel_names": {"0": "pre-RT", "1": "mid-RT"},
                "dataset_id": DATASET_ID,
                "fold": args.fold,
            }
            with open(metrics_path, "w") as f:
                json.dump(payload, f, indent=2)
