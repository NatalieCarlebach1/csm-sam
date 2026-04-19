"""
nnUNet Baseline for BraTS-GLI 2024 longitudinal glioma segmentation.

Trains nnUNet 3d_fullres on BraTS-GLI mid-RT (TP=001) images using all 4
modalities (t1c, t1n, t2f, t2w). Binary label: tumor vs background
(resection cavity label 4 excluded, matching BraTSGLIDataset convention).

Steps:
  1. Convert BraTS-GLI NIfTI data to nnUNetv2 dataset format
  2. Run nnUNet planning and preprocessing
  3. Train nnUNet 3d_fullres fold 0
  4. Inference on val split
  5. Evaluate DSC and write metrics.json

Requirements:
    pip install nnunetv2 SimpleITK

Usage:
    python baselines/nnunet_brats_baseline.py \
        --data_dir /media/data1/natalie/BraTS_GLI \
        --nnunet_dir /media/data1/natalie/nnunet_brats \
        --output_dir /media/data1/natalie/results/brats_baselines/nnunet_brats
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

DATASET_ID = "002"
DATASET_NAME = f"Dataset{DATASET_ID}_BraTSGLI"
RESECTION_LABEL = 4
MODALITIES = ("t1c", "t1n", "t2f", "t2w")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_nnunet() -> bool:
    try:
        r = subprocess.run(["nnUNetv2_plan_and_preprocess", "--help"],
                           capture_output=True, text=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def _parse_patient_folder(name: str):
    parts = name.split("-")
    if len(parts) < 4 or parts[0] != "BraTS" or parts[1] != "GLI":
        return None
    try:
        return int(parts[2]), int(parts[3])
    except ValueError:
        return None


def _discover_pairs(training_dir: Path, val_fraction: float = 0.10, seed: int = 1234):
    """Return (train_pairs, val_pairs) as lists of (pid, pre_dir, mid_dir)."""
    patients: dict[int, dict[int, Path]] = {}
    for d in sorted(training_dir.iterdir()):
        if not d.is_dir():
            continue
        parsed = _parse_patient_folder(d.name)
        if parsed is None:
            continue
        pid, tp = parsed
        patients.setdefault(pid, {})[tp] = d

    pair_pids = sorted(pid for pid, tps in patients.items() if 0 in tps and 1 in tps)
    rng = random.Random(seed)
    shuffled = pair_pids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_set = set(shuffled[:n_val])

    def _record(pid):
        return (f"BraTS-GLI-{pid:05d}", patients[pid][0], patients[pid][1])

    train_pairs = [_record(p) for p in pair_pids if p not in val_set]
    val_pairs   = [_record(p) for p in pair_pids if p in val_set]
    return train_pairs, val_pairs


def _write_case(pid: str, mid_dir: Path, images_dir: Path, labels_dir: Path):
    """Write 4-channel image + binary label for one patient."""
    base = mid_dir.name
    case_id = pid.replace("-", "_")

    for ch_idx, mod in enumerate(MODALITIES):
        src = mid_dir / f"{base}-{mod}.nii.gz"
        if not src.exists():
            print(f"  WARNING: missing {src}, skipping patient")
            return False
        shutil.copy(src, images_dir / f"{case_id}_{ch_idx:04d}.nii.gz")

    seg_path = mid_dir / f"{base}-seg.nii.gz"
    if not seg_path.exists():
        print(f"  WARNING: missing seg for {pid}, skipping")
        return False

    seg_img = sitk.ReadImage(str(seg_path))
    seg_arr = sitk.GetArrayFromImage(seg_img)
    label = ((seg_arr > 0) & (seg_arr != RESECTION_LABEL)).astype(np.uint8)
    label_img = sitk.GetImageFromArray(label)
    label_img.CopyInformation(seg_img)
    sitk.WriteImage(label_img, str(labels_dir / f"{case_id}.nii.gz"))
    return True


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------

def convert_to_nnunet_format(data_dir: Path, nnunet_raw_dir: Path) -> Path:
    if not HAS_SITK:
        raise ImportError("pip install SimpleITK")

    training_dir = data_dir / "Training"
    if not training_dir.exists():
        raise FileNotFoundError(f"BraTS-GLI Training/ not found in {data_dir}")

    dataset_dir = nnunet_raw_dir / DATASET_NAME
    images_dir  = dataset_dir / "imagesTr"
    labels_dir  = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    train_pairs, val_pairs = _discover_pairs(training_dir)
    print(f"[nnUNet-BraTS] {len(train_pairs)} train pairs, {len(val_pairs)} val pairs")

    cases = []
    for pid, pre_dir, mid_dir in train_pairs + val_pairs:
        if _write_case(pid, mid_dir, images_dir, labels_dir):
            cases.append(pid.replace("-", "_"))

    dataset_json = {
        "channel_names": {str(i): f"{m} MRI (mid-RT)" for i, m in enumerate(MODALITIES)},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "dataset_name": DATASET_NAME,
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"[nnUNet-BraTS] Converted {len(cases)} cases → {dataset_dir}")
    return dataset_dir


# ---------------------------------------------------------------------------
# Train / infer / evaluate
# ---------------------------------------------------------------------------

def run_nnunet_training(nnunet_raw_dir: Path, fold: int = 0):
    env = os.environ.copy()
    env["nnUNet_raw"]          = str(nnunet_raw_dir)
    env["nnUNet_preprocessed"] = str(nnunet_raw_dir.parent / "nnunet_preprocessed_brats")
    env["nnUNet_results"]      = str(nnunet_raw_dir.parent / "nnunet_results_brats")

    print("\n--- nnUNet Planning & Preprocessing ---")
    subprocess.run([
        "nnUNetv2_plan_and_preprocess",
        "-d", DATASET_ID,
        "--verify_dataset_integrity",
    ], env=env, check=True)

    print("\n--- nnUNet Training ---")
    subprocess.run([
        "nnUNetv2_train",
        DATASET_ID, "3d_fullres", str(fold),
    ], env=env, check=True)


def run_nnunet_inference(nnunet_raw_dir: Path, data_dir: Path,
                         output_dir: Path, fold: int = 0) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["nnUNet_raw"]          = str(nnunet_raw_dir)
    env["nnUNet_preprocessed"] = str(nnunet_raw_dir.parent / "nnunet_preprocessed_brats")
    env["nnUNet_results"]      = str(nnunet_raw_dir.parent / "nnunet_results_brats")

    training_dir = data_dir / "Training"
    _, val_pairs = _discover_pairs(training_dir)

    test_images_dir = output_dir / "test_images"
    test_images_dir.mkdir(exist_ok=True)

    for pid, pre_dir, mid_dir in val_pairs:
        base = mid_dir.name
        case_id = pid.replace("-", "_")
        for ch_idx, mod in enumerate(MODALITIES):
            src = mid_dir / f"{base}-{mod}.nii.gz"
            if src.exists():
                shutil.copy(src, test_images_dir / f"{case_id}_{ch_idx:04d}.nii.gz")

    predictions_dir = output_dir / "predictions"
    subprocess.run([
        "nnUNetv2_predict",
        "-i", str(test_images_dir),
        "-o", str(predictions_dir),
        "-d", DATASET_ID,
        "-c", "3d_fullres",
        "-f", str(fold),
    ], env=env, check=True)

    return predictions_dir


def evaluate_predictions(predictions_dir: Path, data_dir: Path,
                          output_dir: Path) -> dict:
    from csmsam.utils.metrics import compute_dice

    training_dir = data_dir / "Training"
    _, val_pairs = _discover_pairs(training_dir)

    dsc_list = []
    per_patient = []

    for pid, pre_dir, mid_dir in val_pairs:
        case_id = pid.replace("-", "_")
        pred_path = predictions_dir / f"{case_id}.nii.gz"
        if not pred_path.exists():
            print(f"  WARNING: no prediction for {pid}")
            continue

        base = mid_dir.name
        seg_path = mid_dir / f"{base}-seg.nii.gz"
        if not seg_path.exists():
            continue

        pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))) > 0
        seg_arr  = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path)))
        gt_arr   = ((seg_arr > 0) & (seg_arr != RESECTION_LABEL))

        dsc = compute_dice(pred_arr.astype(np.float32), gt_arr.astype(np.float32))
        dsc_list.append(dsc)
        per_patient.append({"patient_id": pid, "dsc": float(dsc)})
        print(f"  {pid}: DSC={dsc:.4f}")

    metrics = {
        "dsc_mean": float(np.mean(dsc_list)) if dsc_list else 0.0,
        "dsc_std":  float(np.std(dsc_list))  if dsc_list else 0.0,
        "n_patients": len(dsc_list),
    }
    result = {"aggregate": metrics, "per_patient": per_patient, "fallback": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    return metrics


# ---------------------------------------------------------------------------
# Fallback (nnUNet not installed)
# ---------------------------------------------------------------------------

def _emit_fallback(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"dsc_mean": 0.0, "dsc_std": 0.0, "note": "nnUNet not installed", "fallback": True}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"aggregate": metrics, "per_patient": [], "fallback": True}, f, indent=2)
    print("  nnUNet not installed — wrote fallback metrics.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="nnUNet BraTS-GLI baseline")
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--nnunet_dir", type=str, default="/media/data1/natalie/nnunet_brats")
    parser.add_argument("--output_dir", type=str, default="/media/data1/natalie/results/brats_baselines/nnunet_brats")
    parser.add_argument("--fold",       type=int, default=0)
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    nnunet_dir  = Path(args.nnunet_dir)
    output_dir  = Path(args.output_dir)

    if not _check_nnunet():
        print("nnUNet not found — install with: pip install nnunetv2")
        _emit_fallback(data_dir, output_dir)
        return

    if not HAS_SITK:
        print("SimpleITK not found — install with: pip install SimpleITK")
        _emit_fallback(data_dir, output_dir)
        return

    convert_to_nnunet_format(data_dir, nnunet_dir)
    run_nnunet_training(nnunet_dir, fold=args.fold)
    predictions_dir = run_nnunet_inference(nnunet_dir, data_dir, output_dir, fold=args.fold)
    metrics = evaluate_predictions(predictions_dir, data_dir, output_dir)

    print(f"\nnnUNet-BraTS DSC: {metrics['dsc_mean']:.4f} ± {metrics['dsc_std']:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
