"""
Pre-extract all HNTS-MRG slices into a memory-mapped numpy cache per split.

Memory-efficient: writes slice-by-slice via np.memmap — only one patient's
volumes are in RAM at a time. Training reads via memmap (near-zero overhead).

Layout per split:
    data/processed/cache/<split>/
        pre_images.npy    memmap float16, (N_slices, H, W)
        mid_images.npy    memmap float16
        pre_masks.npy     memmap uint8
        mid_masks.npy     memmap uint8
        meta.json         {n_slices, H, W, dtype_img, dtype_mask,
                           patient_ids:[], slice_indices:[], weeks_elapsed:[]}

Usage:
    python scripts/cache_slices.py --data_dir data/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import SimpleITK as sitk
    def load_nifti(p):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
except ImportError:
    import nibabel as nib
    def load_nifti(p):
        return nib.load(str(p)).get_fdata().astype(np.float32).transpose(2, 0, 1)


def normalize_mri(vol, p_lo=1.0, p_hi=99.0):
    lo, hi = np.percentile(vol, p_lo), np.percentile(vol, p_hi)
    return np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def count_slices(data_dir: Path, split: str) -> tuple[int, int, int]:
    """Return (n_slices, H, W) by reading only volume headers."""
    split_dir = data_dir / split
    patients = sorted([d for d in split_dir.iterdir()
                       if d.is_dir() and (d / "mid_image.nii.gz").exists()])
    if not patients:
        return 0, 0, 0

    n = 0
    H = W = 0
    for pdir in patients:
        img = sitk.ReadImage(str(pdir / "mid_image.nii.gz")) if "sitk" in globals() else None
        if img is not None:
            D, hh, ww = img.GetSize()[2], img.GetSize()[1], img.GetSize()[0]
        else:
            arr = load_nifti(pdir / "mid_image.nii.gz")
            D, hh, ww = arr.shape
        n += D
        H, W = hh, ww
    return n, H, W


def cache_split(data_dir: Path, split: str, out_root: Path):
    split_dir = data_dir / split
    patients = sorted([d for d in split_dir.iterdir()
                       if d.is_dir() and (d / "mid_image.nii.gz").exists()])
    if not patients:
        print(f"  {split}: no patients, skipping")
        return

    # First pass — count slices and find max H, W (in case they vary per patient)
    print(f"  {split}: scanning {len(patients)} patients for shape info...")
    sizes = []
    for pdir in tqdm(patients, desc=f"    scan", leave=False):
        arr = load_nifti(pdir / "mid_image.nii.gz")
        sizes.append((pdir, arr.shape))
    total_slices = sum(s[0] for _, s in sizes)
    max_H = max(s[1] for _, s in sizes)
    max_W = max(s[2] for _, s in sizes)
    print(f"  {split}: {total_slices} slices, max HxW={max_H}x{max_W}")

    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preallocate memmaps on disk (no RAM)
    pre_mm = np.memmap(out_dir / "pre_images.npy", dtype=np.float16, mode="w+",
                       shape=(total_slices, max_H, max_W))
    mid_mm = np.memmap(out_dir / "mid_images.npy", dtype=np.float16, mode="w+",
                       shape=(total_slices, max_H, max_W))
    pre_m_mm = np.memmap(out_dir / "pre_masks.npy", dtype=np.uint8, mode="w+",
                         shape=(total_slices, max_H, max_W))
    mid_m_mm = np.memmap(out_dir / "mid_masks.npy", dtype=np.uint8, mode="w+",
                         shape=(total_slices, max_H, max_W))

    pids, sidx, weeks = [], [], []
    cursor = 0

    for pdir, _mid_shape in tqdm(sizes, desc=f"  {split}"):
        pre_vol = normalize_mri(load_nifti(pdir / "pre_image.nii.gz"))
        mid_vol = normalize_mri(load_nifti(pdir / "mid_image.nii.gz"))

        def load_mask(name):
            p = pdir / name
            return load_nifti(p) if p.exists() else np.zeros_like(pre_vol[:1])

        pre_gtvp = load_mask("pre_GTVp.nii.gz")
        pre_gtvn = load_mask("pre_GTVn.nii.gz")
        mid_gtvp = load_mask("mid_GTVp.nii.gz")
        mid_gtvn = load_mask("mid_GTVn.nii.gz")

        if pre_gtvp.shape[0] == 1:
            pre_gtvp = np.zeros_like(pre_vol)
        if pre_gtvn.shape[0] == 1:
            pre_gtvn = np.zeros_like(pre_vol)
        if mid_gtvp.shape[0] == 1:
            mid_gtvp = np.zeros_like(mid_vol)
        if mid_gtvn.shape[0] == 1:
            mid_gtvn = np.zeros_like(mid_vol)

        pre_mask_vol = ((pre_gtvp + pre_gtvn) > 0).astype(np.uint8)
        mid_mask_vol = ((mid_gtvp + mid_gtvn) > 0).astype(np.uint8)

        N = min(pre_vol.shape[0], mid_vol.shape[0], pre_mask_vol.shape[0], mid_mask_vol.shape[0])

        meta = pdir / "metadata.json"
        wk = json.load(open(meta)).get("weeks_elapsed", 3) if meta.exists() else 3

        for i in range(N):
            h, w = pre_vol[i].shape
            # Pad to max if smaller
            def pad(a, fill=0):
                out = np.full((max_H, max_W), fill, dtype=a.dtype)
                out[:a.shape[0], :a.shape[1]] = a
                return out
            pre_mm[cursor] = pad(pre_vol[i]).astype(np.float16)
            mid_mm[cursor] = pad(mid_vol[i]).astype(np.float16)
            pre_m_mm[cursor] = pad(pre_mask_vol[i]).astype(np.uint8)
            mid_m_mm[cursor] = pad(mid_mask_vol[i]).astype(np.uint8)
            pids.append(pdir.name)
            sidx.append(i)
            weeks.append(wk)
            cursor += 1

        del pre_vol, mid_vol, pre_gtvp, pre_gtvn, mid_gtvp, mid_gtvn, pre_mask_vol, mid_mask_vol

    pre_mm.flush(); mid_mm.flush(); pre_m_mm.flush(); mid_m_mm.flush()
    del pre_mm, mid_mm, pre_m_mm, mid_m_mm

    meta = {
        "n_slices": cursor,
        "H": max_H,
        "W": max_W,
        "dtype_img": "float16",
        "dtype_mask": "uint8",
        "patient_ids": pids,
        "slice_indices": sidx,
        "weeks_elapsed": weeks,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f)
    total_mb = sum((out_dir / n).stat().st_size for n in
                   ["pre_images.npy", "mid_images.npy", "pre_masks.npy", "mid_masks.npy"]) / 1e6
    print(f"  {split}: wrote {cursor} slices to {out_dir}  ({total_mb:.1f} MB on disk)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.output) if args.output else data_dir / "cache"

    print(f"Caching HNTS-MRG slices {data_dir} -> {out_root} (memmap mode)")
    for split in ["train", "val", "test"]:
        cache_split(data_dir, split, out_root)
    print("Done.")


if __name__ == "__main__":
    main()
