"""
Download HNTS-MRG 2024 dataset from Zenodo.

HNTS-MRG 2024 (Head and Neck Tumor Segmentation for MR-Guided Radiotherapy):
  - 150 patients, pre-RT and mid-RT T2w MRI + manual GTVp/GTVn contours
  - Zenodo DOI: 10.5281/zenodo.11829006
  - License: CC BY 4.0
  - Access: Registration required at https://hnts-mrg.grand-challenge.org/

Usage:
    # Option 1: Direct Zenodo download (requires access token after registration)
    python data/download_hnts_mrg.py --output_dir data/raw --zenodo_token YOUR_TOKEN

    # Option 2: Manual download
    # 1. Register at https://hnts-mrg.grand-challenge.org/
    # 2. Download from https://zenodo.org/record/11829006
    # 3. Extract to data/raw/
    python data/download_hnts_mrg.py --output_dir data/raw --verify_only

    # Option 3: Use zenodo_get (pip install zenodo-get)
    zenodo_get 11829006 -o data/raw/

Expected structure after download:
    data/raw/
        HaN_GTV/
            ├── imagesTr/
            │   ├── HaN_001_0000.nii.gz   (pre-RT MRI, _0000 suffix)
            │   └── HaN_001_0001.nii.gz   (mid-RT MRI, _0001 suffix)
            ├── labelsTr/
            │   ├── HaN_001_GTVp_pre.nii.gz
            │   ├── HaN_001_GTVn_pre.nii.gz
            │   ├── HaN_001_GTVp_mid.nii.gz
            │   └── HaN_001_GTVn_mid.nii.gz
            └── dataset.json
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ZENODO_RECORD_ID = "11829006"
ZENODO_BASE_URL = f"https://zenodo.org/record/{ZENODO_RECORD_ID}"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Expected files (approximate — check Zenodo page for actual filenames)
EXPECTED_FILES = [
    "HNTS-MRG24_train.zip",
    "HNTS-MRG24_test.zip",
]

TOTAL_PATIENTS = 150
TRAIN_PATIENTS = 120  # approximate
TEST_PATIENTS = 30    # approximate (labels withheld)


def check_zenodo_get():
    """Check if zenodo_get is installed."""
    try:
        result = subprocess.run(["zenodo_get", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_with_zenodo_get(record_id: str, output_dir: Path):
    """Download all files from a Zenodo record using zenodo_get."""
    print(f"Downloading Zenodo record {record_id} to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["zenodo_get", record_id, "-o", str(output_dir)]
    result = subprocess.run(cmd, check=True)
    print("Download complete.")


def download_with_requests(url: str, output_path: Path, token: str | None = None):
    """Download a single file with progress reporting."""
    try:
        import requests
    except ImportError:
        print("pip install requests")
        sys.exit(1)

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"Downloading {url} → {output_path}")
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}% ({downloaded // 1024 // 1024} MB)", end="", flush=True)
    print()


def extract_zip(zip_path: Path, output_dir: Path):
    """Extract a zip file."""
    import zipfile
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    print(f"Extracted to {output_dir}")


def verify_download(raw_dir: Path) -> bool:
    """Verify the downloaded dataset structure."""
    print("\nVerifying dataset structure...")

    # Look for expected directories
    possible_roots = [
        raw_dir / "HaN_GTV",
        raw_dir / "HNTS-MRG24",
        raw_dir,
    ]

    found_root = None
    for root in possible_roots:
        if (root / "imagesTr").exists() or (root / "labelsTr").exists():
            found_root = root
            break

    if found_root is None:
        print("ERROR: Could not find dataset directory structure.")
        print("Expected one of: HaN_GTV/, HNTS-MRG24/, or imagesTr/ in:", raw_dir)
        return False

    print(f"Found dataset root: {found_root}")

    # Count patients
    images_dir = found_root / "imagesTr"
    labels_dir = found_root / "labelsTr"

    if images_dir.exists():
        pre_rt_files = list(images_dir.glob("*_0000.nii.gz"))
        mid_rt_files = list(images_dir.glob("*_0001.nii.gz"))
        print(f"  Pre-RT scans: {len(pre_rt_files)}")
        print(f"  Mid-RT scans: {len(mid_rt_files)}")

    if labels_dir.exists():
        gtvp_pre = list(labels_dir.glob("*GTVp*pre*.nii.gz")) + list(labels_dir.glob("*pre*GTVp*.nii.gz"))
        gtvn_pre = list(labels_dir.glob("*GTVn*pre*.nii.gz")) + list(labels_dir.glob("*pre*GTVn*.nii.gz"))
        gtvp_mid = list(labels_dir.glob("*GTVp*mid*.nii.gz")) + list(labels_dir.glob("*mid*GTVp*.nii.gz"))
        print(f"  GTVp pre-RT labels: {len(gtvp_pre)}")
        print(f"  GTVn pre-RT labels: {len(gtvn_pre)}")
        print(f"  GTVp mid-RT labels: {len(gtvp_mid)}")

    print("Verification complete.")
    return True


def print_manual_download_instructions():
    """Print instructions for manual download."""
    print("\n" + "=" * 60)
    print("HNTS-MRG 2024 Dataset — Manual Download Instructions")
    print("=" * 60)
    print()
    print("1. Register at the challenge page:")
    print(f"   https://hnts-mrg.grand-challenge.org/")
    print()
    print("2. Download from Zenodo (DOI: 10.5281/zenodo.11829006):")
    print(f"   {ZENODO_BASE_URL}")
    print()
    print("3. Alternative — use zenodo_get:")
    print(f"   pip install zenodo-get")
    print(f"   zenodo_get {ZENODO_RECORD_ID} -o data/raw/")
    print()
    print("4. After download, extract to data/raw/ and run:")
    print(f"   python data/download_hnts_mrg.py --output_dir data/raw --verify_only")
    print()
    print("5. Then preprocess:")
    print(f"   python data/preprocess.py --input_dir data/raw --output_dir data/processed")
    print("=" * 60)
    print()


def create_synthetic_dataset(output_dir: Path, n_patients: int = 10, n_slices: int = 30):
    """
    Create a small synthetic dataset for testing the pipeline without real data.

    Generates random MRI-like images with random tumor masks.
    Only use for code testing — NOT for actual experiments.
    """
    import numpy as np
    try:
        import SimpleITK as sitk
    except ImportError:
        print("pip install SimpleITK")
        return

    print(f"\nCreating SYNTHETIC dataset with {n_patients} patients for testing...")
    print("WARNING: Use only for pipeline testing, NOT for real experiments!\n")

    for split in ["train", "val", "test"]:
        n = max(1, n_patients // 3)
        for i in range(n):
            patient_id = f"synthetic_{split}_{i + 1:03d}"
            patient_dir = output_dir / split / patient_id
            patient_dir.mkdir(parents=True, exist_ok=True)

            H, W = 256, 256
            D = n_slices

            # Generate MRI-like volumes (random Gaussian + tumor bump)
            def make_mri():
                vol = np.random.normal(0.5, 0.15, (D, H, W)).clip(0, 1).astype(np.float32)
                # Add a bright tumor region
                cy, cx = H // 2 + np.random.randint(-30, 30), W // 2 + np.random.randint(-30, 30)
                cz = D // 2 + np.random.randint(-5, 5)
                rz, ry, rx = np.random.randint(3, 8), np.random.randint(15, 35), np.random.randint(15, 35)
                z, y, x = np.ogrid[:D, :H, :W]
                ellipsoid = ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2
                vol[ellipsoid < 1] += 0.3
                return vol.clip(0, 1), ellipsoid, cy, cx, cz, ry, rx, rz

            pre_vol, pre_ell, cy, cx, cz, ry, rx, rz = make_mri()
            # Mid-RT: tumor slightly shrunk (treatment response)
            shrink = np.random.uniform(0.6, 0.9)
            mid_vol = pre_vol.copy()
            mid_ell_scale = pre_ell / shrink ** 2
            mid_vol[mid_ell_scale < 1] = pre_vol[mid_ell_scale < 1] * 0.7

            # Masks
            pre_gtvp = (pre_ell < 1).astype(np.float32)
            mid_gtvp = (mid_ell_scale < 1).astype(np.float32)
            pre_gtvn = np.zeros_like(pre_gtvp)  # simplified — no nodal metastases
            mid_gtvn = np.zeros_like(mid_gtvp)

            def save_nii(arr, path):
                img = sitk.GetImageFromArray(arr)
                img.SetSpacing((1.0, 1.0, 3.0))
                sitk.WriteImage(img, str(path))

            save_nii(pre_vol, patient_dir / "pre_image.nii.gz")
            save_nii(mid_vol, patient_dir / "mid_image.nii.gz")
            save_nii(pre_gtvp, patient_dir / "pre_GTVp.nii.gz")
            save_nii(pre_gtvn, patient_dir / "pre_GTVn.nii.gz")
            save_nii(mid_gtvp, patient_dir / "mid_GTVp.nii.gz")
            save_nii(mid_gtvn, patient_dir / "mid_GTVn.nii.gz")

            metadata = {
                "patient_id": patient_id,
                "weeks_elapsed": int(np.random.choice([2, 3, 4])),
                "synthetic": True,
            }
            with open(patient_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

    print(f"Synthetic dataset created at {output_dir}")
    print(f"  Splits: train={n_patients // 3}, val={n_patients // 3}, test={n_patients // 3}")


def main():
    parser = argparse.ArgumentParser(description="Download HNTS-MRG 2024 dataset")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Download destination")
    parser.add_argument("--zenodo_token", type=str, default=None, help="Zenodo access token (if private)")
    parser.add_argument("--verify_only", action="store_true", help="Only verify existing download")
    parser.add_argument("--synthetic", action="store_true", help="Create synthetic dataset for testing")
    parser.add_argument("--n_synthetic", type=int, default=15, help="Number of synthetic patients")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.synthetic:
        create_synthetic_dataset(output_dir.parent / "processed", n_patients=args.n_synthetic)
        return

    if args.verify_only:
        ok = verify_download(output_dir)
        sys.exit(0 if ok else 1)

    print("=" * 60)
    print("HNTS-MRG 2024 Dataset Downloader")
    print("=" * 60)

    # Try zenodo_get first (easiest)
    if check_zenodo_get():
        try:
            download_with_zenodo_get(ZENODO_RECORD_ID, output_dir)
        except subprocess.CalledProcessError:
            print("zenodo_get failed. Falling back to manual download instructions.")
            print_manual_download_instructions()
            sys.exit(1)
    else:
        print_manual_download_instructions()
        print("\nTo auto-download, install zenodo-get:")
        print("  pip install zenodo-get")
        print("  python data/download_hnts_mrg.py --output_dir data/raw")
        sys.exit(0)

    # Verify and extract
    for zip_file in output_dir.glob("*.zip"):
        extract_zip(zip_file, output_dir)

    verify_download(output_dir)
    print("\nNext step: python data/preprocess.py --input_dir data/raw --output_dir data/processed")


if __name__ == "__main__":
    main()
