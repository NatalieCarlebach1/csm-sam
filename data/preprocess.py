"""
HNTS-MRG 2024 preprocessing pipeline.

Reads the raw Zenodo zip (HNTSMRG24_train.zip) or its extracted directory and
produces the per-patient processed layout expected by csmsam/datasets/hnts_mrg.py.

Raw layout (inside the zip):
    HNTSMRG24_train/{patient_id}/preRT/{id}_preRT_T2.nii.gz
    HNTSMRG24_train/{patient_id}/preRT/{id}_preRT_mask.nii.gz
    HNTSMRG24_train/{patient_id}/midRT/{id}_midRT_T2.nii.gz
    HNTSMRG24_train/{patient_id}/midRT/{id}_midRT_mask.nii.gz
    HNTSMRG24_train/{patient_id}/midRT/{id}_preRT_T2_registered.nii.gz
    HNTSMRG24_train/{patient_id}/midRT/{id}_preRT_mask_registered.nii.gz

Raw mask labels:
    0 = background
    1 = GTVp (primary tumor)
    2 = GTVn (nodal metastases)

Output layout:
    data/processed/{train,val,test}/{patient_id}/
        pre_image.nii.gz
        mid_image.nii.gz
        pre_GTVp.nii.gz            (binary: raw label == 1)
        pre_GTVn.nii.gz            (binary: raw label == 2)
        mid_GTVp.nii.gz
        mid_GTVn.nii.gz
        pre_GTVp_registered.nii.gz (binary: raw registered label == 1)
        pre_GTVn_registered.nii.gz (binary: raw registered label == 2)
        pre_image_registered.nii.gz (optional, pre-RT T2 warped to mid-RT grid)
        metadata.json

Usage:
    python data/preprocess.py \\
        --input_dir data/raw \\
        --output_dir data/processed \\
        --n_workers 8
"""

import argparse
import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    print("ERROR: SimpleITK not installed. Run: pip install SimpleITK")
    HAS_SITK = False


WEEKS_ELAPSED_DEFAULT = 3  # typical mid-RT timing for HNTS-MRG 2024
ZIP_NAME = "HNTSMRG24_train.zip"
EXTRACTED_DIR_NAME = "HNTSMRG24_train"

# Label mapping in the raw combined masks
LABEL_GTVP = 1
LABEL_GTVN = 2


def extract_zip_if_needed(raw_dir: Path) -> Path:
    """Extract the Zenodo zip if the extracted directory does not exist.

    Returns the path to the extracted dataset root directory.
    """
    extracted = raw_dir / EXTRACTED_DIR_NAME
    if extracted.is_dir():
        print(f"[extract] Already extracted: {extracted}")
        return extracted

    zip_path = raw_dir / ZIP_NAME
    if not zip_path.is_file():
        raise FileNotFoundError(
            f"Neither extracted dir ({extracted}) nor zip ({zip_path}) found. "
            "Download the HNTS-MRG 2024 dataset first: "
            "python data/download_hnts_mrg.py --output_dir data/raw"
        )

    print(f"[extract] Extracting {zip_path} -> {raw_dir}/ ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    print(f"[extract] Done. Extracted to {extracted}")
    return extracted


def discover_patients(dataset_root: Path) -> list[dict]:
    """Discover all patient directories and map raw filenames.

    Returns a list of dicts with keys:
        patient_id, pre_image, mid_image, pre_mask, mid_mask,
        pre_mask_registered, pre_image_registered
    """
    patients = []
    # Patient dirs are numeric subdirectories of the dataset root
    for d in sorted(dataset_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        if not d.is_dir() or not d.name.isdigit():
            continue
        pid = d.name
        pre_dir = d / "preRT"
        mid_dir = d / "midRT"

        if not pre_dir.is_dir() or not mid_dir.is_dir():
            print(f"  Warning: skipping {pid} (missing preRT or midRT subdir)")
            continue

        pre_image = pre_dir / f"{pid}_preRT_T2.nii.gz"
        pre_mask = pre_dir / f"{pid}_preRT_mask.nii.gz"
        mid_image = mid_dir / f"{pid}_midRT_T2.nii.gz"
        mid_mask = mid_dir / f"{pid}_midRT_mask.nii.gz"
        pre_mask_registered = mid_dir / f"{pid}_preRT_mask_registered.nii.gz"
        pre_image_registered = mid_dir / f"{pid}_preRT_T2_registered.nii.gz"

        # pre_image and mid_image are required
        if not pre_image.is_file():
            print(f"  Warning: skipping {pid} (missing {pre_image.name})")
            continue
        if not mid_image.is_file():
            print(f"  Warning: skipping {pid} (missing {mid_image.name})")
            continue

        patients.append({
            "patient_id": pid,
            "pre_image": str(pre_image),
            "mid_image": str(mid_image),
            "pre_mask": str(pre_mask) if pre_mask.is_file() else None,
            "mid_mask": str(mid_mask) if mid_mask.is_file() else None,
            "pre_mask_registered": str(pre_mask_registered) if pre_mask_registered.is_file() else None,
            "pre_image_registered": str(pre_image_registered) if pre_image_registered.is_file() else None,
        })

    print(f"[discover] Found {len(patients)} patients with paired pre/mid-RT images")
    return patients


def split_mask(mask_img: "sitk.Image", label: int) -> "sitk.Image":
    """Extract a binary mask for a single label from a combined label image.

    Returns a UInt8 SimpleITK image with 1 where mask == label, 0 elsewhere.
    """
    arr = sitk.GetArrayFromImage(mask_img)
    binary = (arr == label).astype(np.uint8)
    out = sitk.GetImageFromArray(binary)
    out.CopyInformation(mask_img)
    return out


def process_patient(
    patient: dict,
    output_dir: Path,
    split: str,
) -> dict:
    """Preprocess one patient: copy images, split masks, write metadata.

    No resampling, cropping, or registration is performed here -- the raw
    data already includes registered files from the challenge organizers, and
    the dataset loader handles normalization and resizing on-the-fly.

    Returns a status dict.
    """
    pid = patient["patient_id"]
    patient_out = output_dir / split / pid

    # Idempotency: skip if metadata.json already exists
    if (patient_out / "metadata.json").is_file():
        return {"patient_id": pid, "status": "skipped", "split": split}

    patient_out.mkdir(parents=True, exist_ok=True)

    try:
        # --- Images: just copy (symlink would be fragile across machines) ---
        _copy_nifti(patient["pre_image"], patient_out / "pre_image.nii.gz")
        _copy_nifti(patient["mid_image"], patient_out / "mid_image.nii.gz")

        # --- Pre-RT mask -> split into GTVp and GTVn ---
        if patient["pre_mask"]:
            pre_mask_img = sitk.ReadImage(patient["pre_mask"])
            sitk.WriteImage(
                split_mask(pre_mask_img, LABEL_GTVP),
                str(patient_out / "pre_GTVp.nii.gz"),
            )
            sitk.WriteImage(
                split_mask(pre_mask_img, LABEL_GTVN),
                str(patient_out / "pre_GTVn.nii.gz"),
            )

        # --- Mid-RT mask -> split into GTVp and GTVn ---
        if patient["mid_mask"]:
            mid_mask_img = sitk.ReadImage(patient["mid_mask"])
            sitk.WriteImage(
                split_mask(mid_mask_img, LABEL_GTVP),
                str(patient_out / "mid_GTVp.nii.gz"),
            )
            sitk.WriteImage(
                split_mask(mid_mask_img, LABEL_GTVN),
                str(patient_out / "mid_GTVn.nii.gz"),
            )

        # --- Pre-RT mask registered (on mid-RT grid) -> split ---
        if patient["pre_mask_registered"]:
            reg_mask_img = sitk.ReadImage(patient["pre_mask_registered"])
            sitk.WriteImage(
                split_mask(reg_mask_img, LABEL_GTVP),
                str(patient_out / "pre_GTVp_registered.nii.gz"),
            )
            sitk.WriteImage(
                split_mask(reg_mask_img, LABEL_GTVN),
                str(patient_out / "pre_GTVn_registered.nii.gz"),
            )

        # --- Pre-RT image registered (optional) ---
        if patient["pre_image_registered"]:
            _copy_nifti(
                patient["pre_image_registered"],
                patient_out / "pre_image_registered.nii.gz",
            )

        # --- Metadata ---
        metadata = {
            "patient_id": pid,
            "weeks_elapsed": WEEKS_ELAPSED_DEFAULT,
            "split": split,
            "has_pre_mask": patient["pre_mask"] is not None,
            "has_mid_mask": patient["mid_mask"] is not None,
            "has_pre_mask_registered": patient["pre_mask_registered"] is not None,
            "has_pre_image_registered": patient["pre_image_registered"] is not None,
        }
        with open(patient_out / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return {"patient_id": pid, "status": "ok", "split": split}

    except Exception as e:
        return {"patient_id": pid, "status": "error", "error": str(e), "split": split}


def _copy_nifti(src: str, dst: Path) -> None:
    """Copy a NIfTI file via SimpleITK (read + write) to ensure consistent format."""
    import shutil
    shutil.copy2(src, str(dst))


def assign_splits(
    patient_ids: list[str],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, str]:
    """Deterministic train/val/test split by sorted patient ID.

    Returns a dict mapping patient_id -> split name.
    """
    sorted_ids = sorted(patient_ids, key=lambda x: int(x))
    n = len(sorted_ids)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val - n_test

    # Deterministic shuffle
    import random
    rng = random.Random(seed)
    shuffled = sorted_ids[:]
    rng.shuffle(shuffled)

    splits = {}
    for i, pid in enumerate(shuffled):
        if i < n_train:
            splits[pid] = "train"
        elif i < n_train + n_val:
            splits[pid] = "val"
        else:
            splits[pid] = "test"

    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess HNTS-MRG 2024 dataset from raw Zenodo zip"
    )
    parser.add_argument(
        "--input_dir", type=str, default="data/raw",
        help="Directory containing HNTSMRG24_train.zip or extracted HNTSMRG24_train/",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--n_workers", type=int, default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.1,
        help="Fraction of patients for validation",
    )
    parser.add_argument(
        "--test_fraction", type=float, default=0.1,
        help="Fraction of patients for test",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val/test split",
    )
    args = parser.parse_args()

    if not HAS_SITK:
        print("SimpleITK is required. Install with: pip install SimpleITK")
        return 1

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract zip if needed
    dataset_root = extract_zip_if_needed(input_dir)

    # Step 2: Discover patients
    patients = discover_patients(dataset_root)
    if not patients:
        print("No patients found. Check your download.")
        return 1

    # Step 3: Assign splits
    patient_ids = [p["patient_id"] for p in patients]
    split_map = assign_splits(
        patient_ids,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    n_train = sum(1 for s in split_map.values() if s == "train")
    n_val = sum(1 for s in split_map.values() if s == "val")
    n_test = sum(1 for s in split_map.values() if s == "test")
    print(f"\n[split] train={n_train}, val={n_val}, test={n_test}")
    print(f"[config] workers={args.n_workers}, seed={args.seed}")
    print(f"[config] output={output_dir}\n")

    # Step 4: Process patients
    results = {"ok": 0, "skipped": 0, "error": 0, "errors": []}
    n = len(patients)

    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {}
            for p in patients:
                split = split_map[p["patient_id"]]
                fut = executor.submit(process_patient, p, output_dir, split)
                futures[fut] = p["patient_id"]

            for future in as_completed(futures):
                r = future.result()
                results[r["status"]] = results.get(r["status"], 0) + 1
                done = results["ok"] + results["skipped"] + results["error"]
                if r["status"] == "error":
                    results["errors"].append(r)
                    print(f"  [{done}/{n}] {r['patient_id']} ({r['split']}) ERROR: {r.get('error', '?')}")
                elif r["status"] == "skipped":
                    print(f"  [{done}/{n}] {r['patient_id']} ({r['split']}) skipped (already exists)")
                else:
                    print(f"  [{done}/{n}] {r['patient_id']} ({r['split']}) ok")
    else:
        for i, p in enumerate(patients):
            split = split_map[p["patient_id"]]
            r = process_patient(p, output_dir, split)
            results[r["status"]] = results.get(r["status"], 0) + 1
            if r["status"] == "error":
                results["errors"].append(r)
                print(f"  [{i + 1}/{n}] {r['patient_id']} ({split}) ERROR: {r.get('error', '?')}")
            elif r["status"] == "skipped":
                print(f"  [{i + 1}/{n}] {r['patient_id']} ({split}) skipped (already exists)")
            else:
                print(f"  [{i + 1}/{n}] {r['patient_id']} ({split}) ok")

    # Summary
    print(f"\n{'='*60}")
    print(f"Preprocessing complete")
    print(f"  OK:      {results['ok']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Errors:  {results['error']}")
    if results["errors"]:
        print("\nErrors:")
        for e in results["errors"]:
            print(f"  {e['patient_id']}: {e.get('error', '?')}")

    # Print output structure summary
    for split_name in ["train", "val", "test"]:
        split_dir = output_dir / split_name
        if split_dir.is_dir():
            count = sum(1 for d in split_dir.iterdir() if d.is_dir())
            print(f"  {split_dir}: {count} patients")

    print(f"\nNext step: python train.py --config configs/default.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
