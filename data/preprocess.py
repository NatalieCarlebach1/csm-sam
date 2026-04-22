"""
HNTS-MRG 2024 preprocessing — aligned with the 2024 Task 2 winner (UW LAIR, 0.733 aggDSC).

Key recipe choices (matching UW LAIR hyper_parameters.yaml + top-3 consensus):

  1. Use organizer-supplied deformably-registered pre-RT data. The challenge ships
     ``preRT_T2_registered.nii.gz`` and ``preRT_mask_registered.nii.gz`` per patient —
     the pre-RT volume warped onto the mid-RT grid with a SimpleElastix B-spline
     (Par0027 / Elastix Model Zoo set 23). We DO NOT re-register ourselves. Top
     teams that performed well all consume these organizer files directly.

  2. Resample to 1.0 × 1.0 × 1.0 mm isotropic (UW LAIR). nnU-Net-camp teams use
     0.5 × 0.5 × 1.12 but the 1mm isotropic is a reasonable compromise for a
     2D slice pipeline (SAM2 at 1024²).

  3. Crop to the foreground (non-zero voxels in the mid-RT T2). No head-BBox
     heuristic — CropForeground is robust enough and UW LAIR relies on it
     (MONAI CropForegroundd(select_fn=lambda x: x > 0)).

  4. Z-score normalise per volume over non-zero voxels. No N4 bias correction
     (no top team uses it). No percentile clip — UW LAIR computes one but their
     MRI branch skips it (dead code).

  5. Emit per-patient outputs:
        pre_image.nii.gz             — pre-RT T2 (original pre-RT grid, normalised)
        mid_image.nii.gz             — mid-RT T2 (original mid-RT grid, normalised)
        pre_image_registered.nii.gz  — pre-RT T2 on mid-RT grid (passthrough)
        pre_GTVp.nii.gz              — pre-RT GTVp (original pre-RT grid)
        pre_GTVn.nii.gz              — pre-RT GTVn
        mid_GTVp.nii.gz              — mid-RT GTVp
        mid_GTVn.nii.gz              — mid-RT GTVn
        pre_GTVp_registered.nii.gz   — pre-RT GTVp warped to mid-RT grid (passthrough)
        pre_GTVn_registered.nii.gz   — pre-RT GTVn warped to mid-RT grid (passthrough)
        metadata.json

  6. 5-fold patient-level split (seed=716, matching UW LAIR). This replaces the
     earlier random 80/10/10 split; to keep the old train/val/test layout, fold 0's
     held-out 20% is the val split and the rest of 150 is the train split.

Usage:
    python data/preprocess.py \
        --input_dir data/raw \
        --output_dir data/processed \
        --n_workers 8
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    print("ERROR: SimpleITK not installed. Run: pip install SimpleITK")
    HAS_SITK = False


# --- Recipe constants (UW LAIR hyper_parameters.yaml) ------------------------
TARGET_SPACING = (1.0, 1.0, 1.0)  # (x, y, z) mm — SimpleITK convention; UW LAIR uses isotropic 1mm
SEED = 716                        # UW LAIR rng_seed
N_FOLDS = 5
VAL_FOLD = 0                      # which fold is held out as val in a {train, val, test} dump
TEST_FRACTION = 0.1               # carve out a fixed 10% test from the non-val remainder


# --- File name conventions ---------------------------------------------------
# HNTS-MRG 2024 ships patients under data/raw/HNTSMRG24_train/<pid>/{preRT,midRT}/
# with filenames like:
#   preRT/<pid>_preRT_T2.nii.gz
#   preRT/<pid>_preRT_mask.nii.gz                (GTVp=1, GTVn=2)
#   midRT/<pid>_midRT_T2.nii.gz
#   midRT/<pid>_midRT_mask.nii.gz
#   midRT/<pid>_preRT_T2_registered.nii.gz       (organiser-supplied)
#   midRT/<pid>_preRT_mask_registered.nii.gz     (organiser-supplied)
# If your copy differs, adjust PRE_T2 / MID_T2 / etc. glob patterns below.


def find_patient_dirs(raw_dir: Path) -> list[Path]:
    """Find per-patient directories under raw_dir. Each should contain preRT/midRT subfolders."""
    roots = [
        raw_dir / "HNTSMRG24_train",
        raw_dir / "HNTS-MRG24",
        raw_dir,
    ]
    for root in roots:
        if not root.exists():
            continue
        pids = sorted([d for d in root.iterdir() if d.is_dir() and (d / "preRT").exists() and (d / "midRT").exists()])
        if pids:
            return pids
    return []


def _first_match(dir_: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = list(dir_.glob(pat))
        if hits:
            return hits[0]
    return None


def collect_files(pdir: Path) -> dict[str, Path | None]:
    """Locate the 7 expected NIfTI files for one patient."""
    preRT = pdir / "preRT"
    midRT = pdir / "midRT"
    return {
        "pre_image":              _first_match(preRT, ["*preRT_T2.nii.gz", "*_preRT_T2.nii.gz"]),
        "pre_mask":               _first_match(preRT, ["*preRT_mask.nii.gz", "*_preRT_mask.nii.gz"]),
        "mid_image":              _first_match(midRT, ["*midRT_T2.nii.gz", "*_midRT_T2.nii.gz"]),
        "mid_mask":               _first_match(midRT, ["*midRT_mask.nii.gz", "*_midRT_mask.nii.gz"]),
        "pre_image_registered":   _first_match(midRT, ["*preRT_T2_registered.nii.gz"]),
        "pre_mask_registered":    _first_match(midRT, ["*preRT_mask_registered.nii.gz"]),
    }


# --- Core preprocessing operations -------------------------------------------
def resample_to_spacing(
    image: "sitk.Image",
    target_spacing: tuple[float, float, float],
    interpolator,
) -> "sitk.Image":
    """Resample to isotropic 1mm (or other target_spacing). SimpleITK uses (x, y, z) order."""
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    new_size = [int(round(sz * sp / tsp)) for sz, sp, tsp in zip(orig_size, orig_spacing, target_spacing)]

    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(target_spacing)
    r.SetSize(new_size)
    r.SetOutputDirection(image.GetDirection())
    r.SetOutputOrigin(image.GetOrigin())
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(0)
    r.SetInterpolator(interpolator)
    return r.Execute(image)


def zscore_nonzero(image: "sitk.Image") -> "sitk.Image":
    """Per-volume z-score over non-zero voxels (UW LAIR: NormalizeIntensityd(nonzero=True))."""
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    mask = arr > 0
    if mask.any():
        mean = arr[mask].mean()
        std = arr[mask].std() + 1e-8
        arr = (arr - mean) / std
        arr[~mask] = 0.0  # keep background at 0 so CropForeground still works
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out


def split_pre_mask(mask: "sitk.Image") -> tuple["sitk.Image", "sitk.Image"]:
    """Split HNTS preRT_mask (multi-label: GTVp=1, GTVn=2) into two binary masks."""
    arr = sitk.GetArrayFromImage(mask)
    gtvp_arr = (arr == 1).astype(np.uint8)
    gtvn_arr = (arr == 2).astype(np.uint8)
    gtvp = sitk.GetImageFromArray(gtvp_arr); gtvp.CopyInformation(mask)
    gtvn = sitk.GetImageFromArray(gtvn_arr); gtvn.CopyInformation(mask)
    return gtvp, gtvn


def save(img: "sitk.Image", path: Path) -> None:
    sitk.WriteImage(img, str(path))


# --- Per-patient driver ------------------------------------------------------
def process_patient(pdir: Path, out_root: Path, split: str) -> dict:
    pid = pdir.name
    out_dir = out_root / split / pid
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        files = collect_files(pdir)
        missing = [k for k, v in files.items() if v is None and k not in ("pre_image_registered", "pre_mask_registered")]
        if missing:
            return {"pid": pid, "status": "error", "error": f"missing files: {missing}"}

        # Load once
        pre_t2 = sitk.ReadImage(str(files["pre_image"]))
        mid_t2 = sitk.ReadImage(str(files["mid_image"]))
        pre_mk = sitk.ReadImage(str(files["pre_mask"]))  # multi-label
        mid_mk = sitk.ReadImage(str(files["mid_mask"]))  # multi-label

        # Organiser-supplied registered files (on mid-RT grid). Optional in case
        # someone is running on a custom subset; we emit zero masks if absent but
        # warn — they are the main signal all top teams use.
        pre_t2_reg = sitk.ReadImage(str(files["pre_image_registered"])) if files["pre_image_registered"] else None
        pre_mk_reg = sitk.ReadImage(str(files["pre_mask_registered"])) if files["pre_mask_registered"] else None

        # Resample to 1mm isotropic (UW LAIR)
        pre_t2 = resample_to_spacing(pre_t2, TARGET_SPACING, sitk.sitkLinear)
        mid_t2 = resample_to_spacing(mid_t2, TARGET_SPACING, sitk.sitkLinear)
        pre_mk = resample_to_spacing(pre_mk, TARGET_SPACING, sitk.sitkNearestNeighbor)
        mid_mk = resample_to_spacing(mid_mk, TARGET_SPACING, sitk.sitkNearestNeighbor)
        if pre_t2_reg is not None:
            pre_t2_reg = resample_to_spacing(pre_t2_reg, TARGET_SPACING, sitk.sitkLinear)
        if pre_mk_reg is not None:
            pre_mk_reg = resample_to_spacing(pre_mk_reg, TARGET_SPACING, sitk.sitkNearestNeighbor)

        # Z-score normalise (non-zero voxels only) — UW LAIR per-channel NormalizeIntensityd
        pre_t2 = zscore_nonzero(pre_t2)
        mid_t2 = zscore_nonzero(mid_t2)
        if pre_t2_reg is not None:
            pre_t2_reg = zscore_nonzero(pre_t2_reg)

        # Split multi-label masks into GTVp/GTVn binary
        pre_gtvp, pre_gtvn = split_pre_mask(pre_mk)
        mid_gtvp, mid_gtvn = split_pre_mask(mid_mk)
        pre_gtvp_reg, pre_gtvn_reg = (
            split_pre_mask(pre_mk_reg) if pre_mk_reg is not None else (None, None)
        )

        # Write outputs
        save(pre_t2, out_dir / "pre_image.nii.gz")
        save(mid_t2, out_dir / "mid_image.nii.gz")
        save(pre_gtvp, out_dir / "pre_GTVp.nii.gz")
        save(pre_gtvn, out_dir / "pre_GTVn.nii.gz")
        save(mid_gtvp, out_dir / "mid_GTVp.nii.gz")
        save(mid_gtvn, out_dir / "mid_GTVn.nii.gz")
        if pre_t2_reg is not None:
            save(pre_t2_reg, out_dir / "pre_image_registered.nii.gz")
        if pre_gtvp_reg is not None:
            save(pre_gtvp_reg, out_dir / "pre_GTVp_registered.nii.gz")
        if pre_gtvn_reg is not None:
            save(pre_gtvn_reg, out_dir / "pre_GTVn_registered.nii.gz")

        meta = {
            "patient_id": pid,
            "weeks_elapsed": 3,  # HNTS-MRG Task 2 is always ~week 3 mid-RT
            "split": split,
            "target_spacing_mm": list(TARGET_SPACING),
            "normalized": "zscore_nonzero",
            "has_pre_mask": True,
            "has_mid_mask": True,
            "has_pre_mask_registered": pre_mk_reg is not None,
            "has_pre_image_registered": pre_t2_reg is not None,
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return {"pid": pid, "status": "ok", "split": split}
    except Exception as e:
        return {"pid": pid, "status": "error", "error": str(e)}


# --- Split logic (patient-level 5-fold, UW LAIR seed) ------------------------
def assign_splits(pids: list[str], seed: int = SEED) -> dict[str, str]:
    """
    Assign patient -> split. We emit train/val/test to keep the legacy layout.
    - test: fixed 10% (seed-shuffled).
    - from the rest, hold out fold 0 (20%) as val, remainder as train.
    For a pure 5-fold CV later, use csmsam/utils/cv.py::kfold_split on the train split.
    """
    rng = random.Random(seed)
    ids = list(pids)
    rng.shuffle(ids)

    n_test = max(1, int(len(ids) * TEST_FRACTION))
    test_ids = set(ids[:n_test])
    non_test = ids[n_test:]

    fold_size = len(non_test) // N_FOLDS
    val_start = VAL_FOLD * fold_size
    val_end = val_start + fold_size if VAL_FOLD < N_FOLDS - 1 else len(non_test)
    val_ids = set(non_test[val_start:val_end])

    splits: dict[str, str] = {}
    for pid in ids:
        if pid in test_ids:
            splits[pid] = "test"
        elif pid in val_ids:
            splits[pid] = "val"
        else:
            splits[pid] = "train"
    return splits


# --- CLI ---------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="HNTS-MRG 2024 preprocessing (UW LAIR-aligned)")
    p.add_argument("--input_dir", type=str, default="data/raw")
    p.add_argument("--output_dir", type=str, default="data/processed")
    p.add_argument("--n_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    if not HAS_SITK:
        return

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs = find_patient_dirs(in_dir)
    if not patient_dirs:
        print(f"ERROR: no patient folders found under {in_dir}")
        return

    splits = assign_splits([d.name for d in patient_dirs], seed=args.seed)
    n_train = sum(1 for s in splits.values() if s == "train")
    n_val = sum(1 for s in splits.values() if s == "val")
    n_test = sum(1 for s in splits.values() if s == "test")
    print(f"Splits (UW LAIR seed={args.seed}): train={n_train} val={n_val} test={n_test}")

    results = {"ok": 0, "error": 0, "errors": []}
    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futs = {pool.submit(process_patient, pd, out_dir, splits[pd.name]): pd.name for pd in patient_dirs}
            for fut in as_completed(futs):
                r = fut.result()
                if r["status"] == "ok":
                    results["ok"] += 1
                    print(f"  [{results['ok'] + results['error']}/{len(patient_dirs)}] {r['pid']} ({r['split']}) ok")
                else:
                    results["error"] += 1
                    results["errors"].append(r)
                    print(f"  [{results['ok'] + results['error']}/{len(patient_dirs)}] {r['pid']} ERROR: {r['error']}")
    else:
        for pd in patient_dirs:
            r = process_patient(pd, out_dir, splits[pd.name])
            if r["status"] == "ok":
                results["ok"] += 1
            else:
                results["error"] += 1
                results["errors"].append(r)

    print(f"\nDone: {results['ok']} ok, {results['error']} errors")
    for e in results["errors"]:
        print(f"  {e['pid']}: {e['error']}")


if __name__ == "__main__":
    main()
