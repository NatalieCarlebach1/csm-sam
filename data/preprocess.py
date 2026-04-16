"""
HNTS-MRG 2024 preprocessing pipeline.

Steps:
  1. Parse dataset structure (detect naming conventions)
  2. Resample to common spacing (1×1×3 mm)
  3. Crop to head & neck region of interest
  4. Register mid-RT to pre-RT (rigid registration)
  5. Normalize intensities (percentile-based)
  6. Split into train/val/test (80/10/10 or use provided splits)
  7. Save as .nii.gz with metadata.json per patient

TODO: emit ``pre_GTVp_registered.nii.gz`` / ``pre_GTVn_registered.nii.gz``
(pre-RT masks warped to the mid-RT image grid). All HNTS-MRG 2024 top-5
teams (UW LAIR 0.733, mic-dkfz 0.727, HiLab 0.725, ...) use this signal as
an input channel or mask-aware attention prior; without it we cannot
reproduce their numbers. The current pipeline registers mid-RT to pre-RT
(wrong direction for this use) — we need an extra pass that inverts the
transform (or re-registers pre→mid with a deformable transform) and
resamples pre-RT GTVp/GTVn with nearest-neighbour interpolation onto the
mid-RT reference grid. See ``csmsam/datasets/hnts_mrg.py`` for the
consumer contract (keys ``pre_mask_registered`` / ``*_gtvp`` / ``*_gtvn``).

Usage:
    python data/preprocess.py \
        --input_dir data/raw \
        --output_dir data/processed \
        --n_workers 8 \
        --val_fraction 0.1 \
        --test_fraction 0.1
"""

import argparse
import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    print("ERROR: SimpleITK not installed. Run: pip install SimpleITK")
    HAS_SITK = False


TARGET_SPACING = (3.0, 1.0, 1.0)   # (z, y, x) in mm — 3mm slice thickness typical for H&N
TARGET_SIZE = (64, 256, 256)        # (D, H, W) after resampling
WEEKS_ELAPSED_DEFAULT = 3           # typical mid-RT timing


def find_dataset_root(raw_dir: Path) -> Path:
    """Auto-detect dataset root directory."""
    candidates = [
        raw_dir / "HaN_GTV",
        raw_dir / "HNTS-MRG24",
        raw_dir,
    ]
    for c in candidates:
        if (c / "imagesTr").exists() or (c / "labelsTr").exists():
            return c
    # Look recursively
    for d in raw_dir.rglob("imagesTr"):
        return d.parent
    return raw_dir


def parse_patient_list(dataset_root: Path) -> list[dict]:
    """
    Parse the dataset directory to find all patient file paths.

    Handles multiple naming conventions used in HNTS-MRG.
    """
    patients = []
    images_dir = dataset_root / "imagesTr"
    labels_dir = dataset_root / "labelsTr"

    if not images_dir.exists():
        print(f"Warning: imagesTr not found at {images_dir}")
        return patients

    # Find all pre-RT images (convention: *_0000.nii.gz)
    pre_rt_files = sorted(images_dir.glob("*_0000.nii.gz"))

    for pre_rt_path in pre_rt_files:
        # Extract patient ID
        stem = pre_rt_path.name.replace("_0000.nii.gz", "")
        mid_rt_path = images_dir / f"{stem}_0001.nii.gz"

        if not mid_rt_path.exists():
            print(f"  Warning: mid-RT image not found for {stem}, skipping")
            continue

        # Find labels (try multiple naming patterns)
        def find_label(pattern_list: list[str]) -> Path | None:
            for pattern in pattern_list:
                matches = list(labels_dir.glob(pattern))
                if matches:
                    return matches[0]
            return None

        pre_gtvp = find_label([
            f"{stem}_GTVp_pre.nii.gz",
            f"{stem}_pre_GTVp.nii.gz",
            f"{stem}_GTVp.nii.gz",
        ])
        pre_gtvn = find_label([
            f"{stem}_GTVn_pre.nii.gz",
            f"{stem}_pre_GTVn.nii.gz",
            f"{stem}_GTVn.nii.gz",
        ])
        mid_gtvp = find_label([
            f"{stem}_GTVp_mid.nii.gz",
            f"{stem}_mid_GTVp.nii.gz",
        ])
        mid_gtvn = find_label([
            f"{stem}_GTVn_mid.nii.gz",
            f"{stem}_mid_GTVn.nii.gz",
        ])

        patients.append({
            "patient_id": stem,
            "pre_image": str(pre_rt_path),
            "mid_image": str(mid_rt_path),
            "pre_gtvp": str(pre_gtvp) if pre_gtvp else None,
            "pre_gtvn": str(pre_gtvn) if pre_gtvn else None,
            "mid_gtvp": str(mid_gtvp) if mid_gtvp else None,
            "mid_gtvn": str(mid_gtvn) if mid_gtvn else None,
        })

    print(f"Found {len(patients)} patients with paired pre/mid-RT images")
    return patients


def resample_to_spacing(
    image: "sitk.Image",
    target_spacing: tuple[float, float, float],
    interpolator=None,
) -> "sitk.Image":
    """Resample a SimpleITK image to target voxel spacing."""
    if interpolator is None:
        interpolator = sitk.sitkLinear

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(orig_sz * orig_sp / tgt_sp))
        for orig_sz, orig_sp, tgt_sp in zip(original_size, original_spacing, target_spacing)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    resample.SetInterpolator(interpolator)
    return resample.Execute(image)


def crop_to_neck_region(
    image: "sitk.Image",
    mask: "sitk.Image | None" = None,
    padding: int = 20,
) -> tuple["sitk.Image", tuple]:
    """
    Crop to a bounding box around the neck/head region.

    If mask is provided, uses mask bounding box + padding.
    Otherwise crops to non-zero region of the image.
    """
    arr = sitk.GetArrayFromImage(image)  # (D, H, W)

    if mask is not None:
        mask_arr = sitk.GetArrayFromImage(mask)
        nonzero = np.where(mask_arr > 0)
    else:
        nonzero = np.where(arr > arr.mean() * 0.1)

    if len(nonzero[0]) == 0:
        return image, (0, arr.shape[0], 0, arr.shape[1], 0, arr.shape[2])

    z_min, z_max = max(0, nonzero[0].min() - padding), min(arr.shape[0], nonzero[0].max() + padding)
    y_min, y_max = max(0, nonzero[1].min() - padding), min(arr.shape[1], nonzero[1].max() + padding)
    x_min, x_max = max(0, nonzero[2].min() - padding), min(arr.shape[2], nonzero[2].max() + padding)

    cropped_arr = arr[z_min:z_max, y_min:y_max, x_min:x_max]
    cropped = sitk.GetImageFromArray(cropped_arr)
    cropped.SetSpacing(image.GetSpacing())

    return cropped, (z_min, z_max, y_min, y_max, x_min, x_max)


def apply_crop(image: "sitk.Image", crop_bounds: tuple) -> "sitk.Image":
    """Apply crop bounds (from crop_to_neck_region) to another image."""
    z_min, z_max, y_min, y_max, x_min, x_max = crop_bounds
    arr = sitk.GetArrayFromImage(image)
    cropped = arr[z_min:z_max, y_min:y_max, x_min:x_max]
    out = sitk.GetImageFromArray(cropped)
    out.SetSpacing(image.GetSpacing())
    return out


def rigid_register(
    fixed: "sitk.Image",
    moving: "sitk.Image",
) -> tuple["sitk.Image", "sitk.Transform"]:
    """
    Rigid registration: align mid-RT to pre-RT.

    Registration is important because patient position can differ between scans.
    We use a fast rigid registration (translation + rotation) via SimpleITK.
    """
    # Convert to Float32
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    registration = sitk.ImageRegistrationMethod()

    # Metric: mutual information (works well for MRI)
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.1)

    registration.SetInterpolator(sitk.sitkLinear)

    # Optimizer: gradient descent
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Initial transform: rigid (Euler 3D)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_f,
        moving_f,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration.Execute(fixed_f, moving_f)

    # Apply transform to moving image
    registered = sitk.Resample(
        moving_f,
        fixed_f,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_f.GetPixelID(),
    )

    return registered, final_transform


def apply_transform_to_mask(
    mask: "sitk.Image",
    reference: "sitk.Image",
    transform: "sitk.Transform",
) -> "sitk.Image":
    """Apply registration transform to a segmentation mask (nearest neighbor)."""
    return sitk.Resample(
        sitk.Cast(mask, sitk.sitkFloat32),
        reference,
        transform,
        sitk.sitkNearestNeighbor,
        0.0,
        sitk.sitkFloat32,
    )


def normalize_percentile(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Percentile-based intensity normalization for T2w MRI."""
    lo = np.percentile(arr[arr > 0], p_low) if arr.any() else 0
    hi = np.percentile(arr[arr > 0], p_high) if arr.any() else 1
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)
    return arr.astype(np.float32)


def pad_or_crop_to_size(arr: np.ndarray, target_size: tuple[int, int, int]) -> np.ndarray:
    """Pad or center-crop a 3D array to target_size (D, H, W)."""
    D, H, W = arr.shape
    tD, tH, tW = target_size

    # Pad if smaller
    pad_d = max(0, tD - D)
    pad_h = max(0, tH - H)
    pad_w = max(0, tW - W)
    arr = np.pad(arr, [
        (pad_d // 2, pad_d - pad_d // 2),
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2),
    ])

    # Center crop if larger
    D, H, W = arr.shape
    start_d = (D - tD) // 2
    start_h = (H - tH) // 2
    start_w = (W - tW) // 2
    return arr[start_d:start_d + tD, start_h:start_h + tH, start_w:start_w + tW]


def process_patient(
    patient: dict,
    output_dir: Path,
    split: str,
    register: bool = True,
) -> dict:
    """
    Preprocess one patient and save to output_dir/split/patient_id/.

    Returns a status dict.
    """
    pid = patient["patient_id"]
    patient_out = output_dir / split / pid
    patient_out.mkdir(parents=True, exist_ok=True)

    try:
        # Load images
        pre_img = sitk.ReadImage(patient["pre_image"])
        mid_img = sitk.ReadImage(patient["mid_image"])

        # Resample to common spacing
        pre_img = resample_to_spacing(pre_img, TARGET_SPACING, sitk.sitkLinear)
        mid_img = resample_to_spacing(mid_img, TARGET_SPACING, sitk.sitkLinear)

        # Load and resample masks
        def load_mask(path):
            if path and Path(path).exists():
                m = sitk.ReadImage(path)
                return resample_to_spacing(m, TARGET_SPACING, sitk.sitkNearestNeighbor)
            return None

        pre_gtvp = load_mask(patient["pre_gtvp"])
        pre_gtvn = load_mask(patient["pre_gtvn"])
        mid_gtvp = load_mask(patient["mid_gtvp"])
        mid_gtvn = load_mask(patient["mid_gtvn"])

        # Optional: rigid registration of mid-RT to pre-RT
        transform = None
        if register:
            try:
                mid_img_reg, transform = rigid_register(pre_img, mid_img)
                mid_img = mid_img_reg
                if mid_gtvp:
                    mid_gtvp = apply_transform_to_mask(mid_gtvp, pre_img, transform)
                if mid_gtvn:
                    mid_gtvn = apply_transform_to_mask(mid_gtvn, pre_img, transform)
            except Exception as e:
                print(f"  Warning: registration failed for {pid}: {e}. Skipping registration.")

        # Crop to ROI (use pre-RT mask as reference for crop bounds)
        ref_mask = pre_gtvp if pre_gtvp else None
        _, crop_bounds = crop_to_neck_region(pre_img, ref_mask, padding=30)
        pre_img = apply_crop(pre_img, crop_bounds)
        mid_img = apply_crop(mid_img, crop_bounds)

        def crop_mask(m):
            if m is not None:
                return apply_crop(m, crop_bounds)
            return None

        pre_gtvp = crop_mask(pre_gtvp)
        pre_gtvn = crop_mask(pre_gtvn)
        mid_gtvp = crop_mask(mid_gtvp)
        mid_gtvn = crop_mask(mid_gtvn)

        # Convert to numpy, normalize, pad/crop to target size
        pre_arr = normalize_percentile(sitk.GetArrayFromImage(pre_img).astype(np.float32))
        mid_arr = normalize_percentile(sitk.GetArrayFromImage(mid_img).astype(np.float32))
        pre_arr = pad_or_crop_to_size(pre_arr, TARGET_SIZE)
        mid_arr = pad_or_crop_to_size(mid_arr, TARGET_SIZE)

        def get_mask_arr(m):
            if m is not None:
                arr = (sitk.GetArrayFromImage(m) > 0.5).astype(np.float32)
                return pad_or_crop_to_size(arr, TARGET_SIZE)
            return np.zeros(TARGET_SIZE, dtype=np.float32)

        pre_gtvp_arr = get_mask_arr(pre_gtvp)
        pre_gtvn_arr = get_mask_arr(pre_gtvn)
        mid_gtvp_arr = get_mask_arr(mid_gtvp)
        mid_gtvn_arr = get_mask_arr(mid_gtvn)

        # Save all as NIfTI
        def save(arr, path, spacing=(3.0, 1.0, 1.0)):
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing(spacing)
            sitk.WriteImage(img, str(path))

        save(pre_arr, patient_out / "pre_image.nii.gz")
        save(mid_arr, patient_out / "mid_image.nii.gz")
        save(pre_gtvp_arr, patient_out / "pre_GTVp.nii.gz")
        save(pre_gtvn_arr, patient_out / "pre_GTVn.nii.gz")
        save(mid_gtvp_arr, patient_out / "mid_GTVp.nii.gz")
        save(mid_gtvn_arr, patient_out / "mid_GTVn.nii.gz")

        # Save metadata
        metadata = {
            "patient_id": pid,
            "weeks_elapsed": patient.get("weeks_elapsed", WEEKS_ELAPSED_DEFAULT),
            "target_size": list(TARGET_SIZE),
            "target_spacing": list(TARGET_SPACING),
            "registration_applied": register and transform is not None,
            "has_mid_gtvp": mid_gtvp is not None,
            "has_mid_gtvn": mid_gtvn is not None,
        }
        with open(patient_out / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return {"patient_id": pid, "status": "ok", "split": split}

    except Exception as e:
        return {"patient_id": pid, "status": "error", "error": str(e), "split": split}


def main():
    parser = argparse.ArgumentParser(description="Preprocess HNTS-MRG 2024 dataset")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Raw data directory")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--n_workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--test_fraction", type=float, default=0.1, help="Test fraction")
    parser.add_argument("--no_register", action="store_true", help="Skip rigid registration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    args = parser.parse_args()

    if not HAS_SITK:
        print("SimpleITK required: pip install SimpleITK")
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find dataset root
    dataset_root = find_dataset_root(input_dir)
    print(f"Dataset root: {dataset_root}")

    # Parse patient list
    patients = parse_patient_list(dataset_root)
    if not patients:
        print("No patients found. Check your download.")
        return

    # Split patients
    random.seed(args.seed)
    random.shuffle(patients)
    n = len(patients)
    n_val = max(1, int(n * args.val_fraction))
    n_test = max(1, int(n * args.test_fraction))
    n_train = n - n_val - n_test

    splits = (
        [(p, "train") for p in patients[:n_train]]
        + [(p, "val") for p in patients[n_train:n_train + n_val]]
        + [(p, "test") for p in patients[n_train + n_val:]]
    )

    print(f"\nSplit: train={n_train}, val={n_val}, test={n_test}")
    print(f"Registration: {'disabled' if args.no_register else 'enabled'}")
    print(f"Workers: {args.n_workers}")
    print(f"Output: {output_dir}\n")

    # Process patients
    results = {"ok": 0, "error": 0, "errors": []}

    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {
                executor.submit(process_patient, p, output_dir, split, not args.no_register): p["patient_id"]
                for p, split in splits
            }
            for future in as_completed(futures):
                r = future.result()
                if r["status"] == "ok":
                    results["ok"] += 1
                    print(f"  [{results['ok'] + results['error']}/{n}] {r['patient_id']} ({r['split']}) ✓")
                else:
                    results["error"] += 1
                    results["errors"].append(r)
                    print(f"  [{results['ok'] + results['error']}/{n}] {r['patient_id']} ERROR: {r.get('error', '?')}")
    else:
        for i, (patient, split) in enumerate(splits):
            r = process_patient(patient, output_dir, split, not args.no_register)
            if r["status"] == "ok":
                results["ok"] += 1
                print(f"  [{i + 1}/{n}] {r['patient_id']} ({split}) ✓")
            else:
                results["error"] += 1
                results["errors"].append(r)
                print(f"  [{i + 1}/{n}] {r['patient_id']} ERROR: {r.get('error', '?')}")

    print(f"\nPreprocessing complete: {results['ok']} OK, {results['error']} errors")
    if results["errors"]:
        print("Errors:")
        for e in results["errors"]:
            print(f"  {e['patient_id']}: {e.get('error', '?')}")

    print(f"\nNext step: python train.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
