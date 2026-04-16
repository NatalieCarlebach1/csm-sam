"""
HNTS-MRG 2024 Dataset for CSM-SAM.

Dataset structure (after preprocessing):
    data/processed/
        train/
            patient_001/
                pre_image.nii.gz    — pre-RT T2-weighted MRI
                pre_GTVp.nii.gz     — pre-RT primary tumor mask
                pre_GTVn.nii.gz     — pre-RT nodal metastases mask
                mid_image.nii.gz    — mid-RT T2-weighted MRI
                mid_GTVp.nii.gz     — mid-RT primary tumor mask
                mid_GTVn.nii.gz     — mid-RT nodal metastases mask
                metadata.json       — weeks_elapsed, patient_id, etc.
            ...
        val/
            ...
        test/
            ...

HNTSMRGDataset:
    Returns one full 3D volume pair (pre + mid) per patient.
    Used for validation and testing (sequential 3D inference).

HNTSMRGSliceDataset:
    Returns individual 2D slice pairs extracted from volumes.
    Used for training (random access, augmentation-friendly).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False
    print("Warning: SimpleITK not installed. Using nibabel fallback.")
    import nibabel as nib


# SAM2 expected image statistics
SAM2_MEAN = torch.tensor([0.485, 0.456, 0.406])
SAM2_STD = torch.tensor([0.229, 0.224, 0.225])

# Default image size for SAM2-L (1024×1024)
SAM2_IMAGE_SIZE = 1024


def load_nifti(path: str | Path) -> np.ndarray:
    """Load a NIfTI volume to a float32 numpy array."""
    path = str(path)
    if HAS_SITK:
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
    else:
        arr = nib.load(path).get_fdata().astype(np.float32)
        arr = arr.transpose(2, 0, 1)  # (H, W, D) → (D, H, W)
    return arr


def normalize_mri(volume: np.ndarray, percentile_low: float = 1.0, percentile_high: float = 99.0) -> np.ndarray:
    """
    Percentile-based normalization for T2-weighted MRI.
    Clips to [p1, p99] then scales to [0, 1].
    """
    p_low = np.percentile(volume, percentile_low)
    p_high = np.percentile(volume, percentile_high)
    volume = np.clip(volume, p_low, p_high)
    volume = (volume - p_low) / (p_high - p_low + 1e-8)
    return volume.astype(np.float32)


def to_rgb_tensor(slice_2d: np.ndarray, size: int = SAM2_IMAGE_SIZE) -> torch.Tensor:
    """
    Convert a 2D grayscale slice to a SAM2-compatible RGB tensor.

    Args:
        slice_2d : (H, W) float32 in [0, 1]
        size     : target spatial size (SAM2 default: 1024)

    Returns:
        (3, size, size) float32 tensor, normalized with SAM2 mean/std
    """
    t = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    t = t.squeeze(0).repeat(3, 1, 1)  # (3, size, size)
    # SAM2 normalization
    t = (t - SAM2_MEAN[:, None, None]) / SAM2_STD[:, None, None]
    return t


def to_mask_tensor(mask_2d: np.ndarray, size: int = SAM2_IMAGE_SIZE) -> torch.Tensor:
    """
    Convert a 2D binary mask to a resized float tensor.

    Returns (1, size, size) float32 mask.
    """
    t = torch.from_numpy((mask_2d > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="nearest")
    return t.squeeze(0)  # (1, size, size)


class HNTSMRGDataset(Dataset):
    """
    Volume-level dataset for HNTS-MRG 2024.

    Returns one patient at a time with all slices.
    Used for sequential 3D inference (validation / test).

    __getitem__ returns:
        pre_images   : (N_slices, 3, H, W) — pre-RT slices as SAM2 tensors
        pre_masks    : (N_slices, 1, H, W) — pre-RT combined GTVp+GTVn masks
        mid_images   : (N_slices, 3, H, W)
        mid_masks    : (N_slices, 1, H, W)
        weeks_elapsed: int
        patient_id   : str
        pre_masks_gtvp: (N_slices, 1, H, W) — GTVp only
        pre_masks_gtvn: (N_slices, 1, H, W) — GTVn only
        mid_masks_gtvp: (N_slices, 1, H, W)
        mid_masks_gtvn: (N_slices, 1, H, W)
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = SAM2_IMAGE_SIZE,
        min_mid_tumor_slices: int = 1,
    ):
        """
        Args:
            data_dir   : root processed data directory
            split      : "train", "val", or "test"
            image_size : SAM2 input resolution
            min_mid_tumor_slices: filter patients with fewer than N mid-RT tumor slices
        """
        self.data_dir = Path(data_dir) / split
        self.image_size = image_size
        self.split = split

        self.patient_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and (d / "mid_image.nii.gz").exists()
        ])

        if not self.patient_dirs:
            raise FileNotFoundError(
                f"No patient directories found in {self.data_dir}. "
                "Run data/preprocess.py first."
            )

        print(f"[HNTSMRGDataset] {split}: {len(self.patient_dirs)} patients")

    def __len__(self) -> int:
        return len(self.patient_dirs)

    def _load_patient(self, patient_dir: Path) -> dict:
        """Load all volumes for one patient."""
        pre_vol = load_nifti(patient_dir / "pre_image.nii.gz")
        mid_vol = load_nifti(patient_dir / "mid_image.nii.gz")

        # Load GTVp and GTVn masks separately and combined
        pre_gtvp = load_nifti(patient_dir / "pre_GTVp.nii.gz") if (patient_dir / "pre_GTVp.nii.gz").exists() else np.zeros_like(pre_vol)
        pre_gtvn = load_nifti(patient_dir / "pre_GTVn.nii.gz") if (patient_dir / "pre_GTVn.nii.gz").exists() else np.zeros_like(pre_vol)
        mid_gtvp = load_nifti(patient_dir / "mid_GTVp.nii.gz") if (patient_dir / "mid_GTVp.nii.gz").exists() else np.zeros_like(mid_vol)
        mid_gtvn = load_nifti(patient_dir / "mid_GTVn.nii.gz") if (patient_dir / "mid_GTVn.nii.gz").exists() else np.zeros_like(mid_vol)

        # Normalize MRI
        pre_vol = normalize_mri(pre_vol)
        mid_vol = normalize_mri(mid_vol)

        # Load metadata
        meta_path = patient_dir / "metadata.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        weeks_elapsed = metadata.get("weeks_elapsed", 3)

        return {
            "pre_vol": pre_vol,
            "mid_vol": mid_vol,
            "pre_gtvp": pre_gtvp,
            "pre_gtvn": pre_gtvn,
            "mid_gtvp": mid_gtvp,
            "mid_gtvn": mid_gtvn,
            "weeks_elapsed": weeks_elapsed,
            "patient_id": patient_dir.name,
        }

    def __getitem__(self, idx: int) -> dict:
        patient_dir = self.patient_dirs[idx]
        data = self._load_patient(patient_dir)

        pre_vol = data["pre_vol"]  # (D, H, W)
        mid_vol = data["mid_vol"]
        N = min(pre_vol.shape[0], mid_vol.shape[0])

        # Align number of slices (truncate to shorter)
        pre_vol = pre_vol[:N]
        mid_vol = mid_vol[:N]
        pre_gtvp = data["pre_gtvp"][:N]
        pre_gtvn = data["pre_gtvn"][:N]
        mid_gtvp = data["mid_gtvp"][:N]
        mid_gtvn = data["mid_gtvn"][:N]

        # Convert each slice to SAM2 tensor
        pre_images = torch.stack([to_rgb_tensor(pre_vol[i], self.image_size) for i in range(N)])
        mid_images = torch.stack([to_rgb_tensor(mid_vol[i], self.image_size) for i in range(N)])
        pre_masks_gtvp = torch.stack([to_mask_tensor(pre_gtvp[i], self.image_size) for i in range(N)])
        pre_masks_gtvn = torch.stack([to_mask_tensor(pre_gtvn[i], self.image_size) for i in range(N)])
        mid_masks_gtvp = torch.stack([to_mask_tensor(mid_gtvp[i], self.image_size) for i in range(N)])
        mid_masks_gtvn = torch.stack([to_mask_tensor(mid_gtvn[i], self.image_size) for i in range(N)])

        return {
            "pre_images": pre_images,                              # (N, 3, H, W)
            "mid_images": mid_images,
            "pre_masks": (pre_masks_gtvp + pre_masks_gtvn).clamp(0, 1),  # combined
            "mid_masks": (mid_masks_gtvp + mid_masks_gtvn).clamp(0, 1),
            "pre_masks_gtvp": pre_masks_gtvp,
            "pre_masks_gtvn": pre_masks_gtvn,
            "mid_masks_gtvp": mid_masks_gtvp,
            "mid_masks_gtvn": mid_masks_gtvn,
            "weeks_elapsed": data["weeks_elapsed"],
            "patient_id": data["patient_id"],
        }


class HNTSMRGSliceDataset(Dataset):
    """
    Slice-level dataset for training CSM-SAM.

    Extracts 2D slice pairs from all patients. Each item is one
    (pre_slice, mid_slice, pre_mask_slice, mid_mask_slice) quadruple.

    Only slices with at least 1 tumor pixel in the mid-RT mask are returned
    (plus an equal number of tumor-free slices for balance).

    Augmentation applied:
        - Random horizontal + vertical flip
        - Random rotation ±15°
        - Gaussian noise (σ ~ U[0, 0.02])
        - Intensity jitter (brightness ±10%)
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = SAM2_IMAGE_SIZE,
        augment: bool = True,
        tumor_ratio: float = 0.7,
        cache_metadata: bool = True,
    ):
        """
        Args:
            tumor_ratio: fraction of returned slices that contain tumor
        """
        self.data_dir = Path(data_dir) / split
        self.image_size = image_size
        self.augment = augment and (split == "train")
        self.tumor_ratio = tumor_ratio

        # Build index of all valid (patient, slice_idx) pairs
        self.tumor_slices: list[tuple[Path, int]] = []
        self.bg_slices: list[tuple[Path, int]] = []

        patient_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and (d / "mid_GTVp.nii.gz").exists()
        ])

        print(f"[HNTSMRGSliceDataset] Building index from {len(patient_dirs)} {split} patients...")

        for pdir in patient_dirs:
            try:
                mid_gtvp = load_nifti(pdir / "mid_GTVp.nii.gz")
                mid_gtvn_path = pdir / "mid_GTVn.nii.gz"
                mid_gtvn = load_nifti(mid_gtvn_path) if mid_gtvn_path.exists() else np.zeros_like(mid_gtvp)
                mid_combined = (mid_gtvp + mid_gtvn) > 0  # (D, H, W) bool

                for i in range(mid_combined.shape[0]):
                    if mid_combined[i].any():
                        self.tumor_slices.append((pdir, i))
                    else:
                        self.bg_slices.append((pdir, i))
            except Exception as e:
                print(f"  Warning: skipping {pdir.name}: {e}")

        print(f"  Tumor slices: {len(self.tumor_slices)}, BG slices: {len(self.bg_slices)}")

    def __len__(self) -> int:
        # Balance tumor vs. background
        n_tumor = len(self.tumor_slices)
        n_bg = int(n_tumor * (1 - self.tumor_ratio) / self.tumor_ratio)
        return n_tumor + min(n_bg, len(self.bg_slices))

    def __getitem__(self, idx: int) -> dict:
        n_tumor = len(self.tumor_slices)

        # Decide whether to return a tumor slice or BG slice
        if idx < n_tumor:
            patient_dir, slice_idx = self.tumor_slices[idx]
        else:
            bg_idx = (idx - n_tumor) % len(self.bg_slices)
            patient_dir, slice_idx = self.bg_slices[bg_idx]

        return self._load_slice(patient_dir, slice_idx)

    def _load_slice(self, patient_dir: Path, slice_idx: int) -> dict:
        """Load one pre/mid slice pair."""
        # Load volumes
        pre_vol = normalize_mri(load_nifti(patient_dir / "pre_image.nii.gz"))
        mid_vol = normalize_mri(load_nifti(patient_dir / "mid_image.nii.gz"))
        pre_gtvp = load_nifti(patient_dir / "pre_GTVp.nii.gz") if (patient_dir / "pre_GTVp.nii.gz").exists() else np.zeros_like(pre_vol)
        pre_gtvn = load_nifti(patient_dir / "pre_GTVn.nii.gz") if (patient_dir / "pre_GTVn.nii.gz").exists() else np.zeros_like(pre_vol)
        mid_gtvp = load_nifti(patient_dir / "mid_GTVp.nii.gz") if (patient_dir / "mid_GTVp.nii.gz").exists() else np.zeros_like(mid_vol)
        mid_gtvn = load_nifti(patient_dir / "mid_GTVn.nii.gz") if (patient_dir / "mid_GTVn.nii.gz").exists() else np.zeros_like(mid_vol)

        # Clamp slice index to valid range
        N = min(pre_vol.shape[0], mid_vol.shape[0])
        slice_idx = min(slice_idx, N - 1)

        pre_slice = pre_vol[slice_idx]   # (H, W)
        mid_slice = mid_vol[slice_idx]
        pre_mask = (pre_gtvp[slice_idx] + pre_gtvn[slice_idx]) > 0
        mid_mask = (mid_gtvp[slice_idx] + mid_gtvn[slice_idx]) > 0
        pre_gtvp_sl = pre_gtvp[slice_idx] > 0
        pre_gtvn_sl = pre_gtvn[slice_idx] > 0
        mid_gtvp_sl = mid_gtvp[slice_idx] > 0
        mid_gtvn_sl = mid_gtvn[slice_idx] > 0

        # Augmentation
        if self.augment:
            pre_slice, mid_slice, pre_mask, mid_mask = self._augment(
                pre_slice, mid_slice, pre_mask.astype(np.float32), mid_mask.astype(np.float32)
            )

        # Load metadata
        meta_path = patient_dir / "metadata.json"
        weeks_elapsed = 3
        if meta_path.exists():
            with open(meta_path) as f:
                weeks_elapsed = json.load(f).get("weeks_elapsed", 3)

        return {
            "pre_image": to_rgb_tensor(pre_slice.astype(np.float32), self.image_size),
            "mid_image": to_rgb_tensor(mid_slice.astype(np.float32), self.image_size),
            "pre_mask": to_mask_tensor(pre_mask.astype(np.float32), self.image_size),
            "mid_mask": to_mask_tensor(mid_mask.astype(np.float32), self.image_size),
            "pre_mask_gtvp": to_mask_tensor(pre_gtvp_sl.astype(np.float32), self.image_size),
            "pre_mask_gtvn": to_mask_tensor(pre_gtvn_sl.astype(np.float32), self.image_size),
            "mid_mask_gtvp": to_mask_tensor(mid_gtvp_sl.astype(np.float32), self.image_size),
            "mid_mask_gtvn": to_mask_tensor(mid_gtvn_sl.astype(np.float32), self.image_size),
            "weeks_elapsed": weeks_elapsed,
            "patient_id": patient_dir.name,
            "slice_idx": slice_idx,
        }

    def _augment(
        self,
        pre: np.ndarray,
        mid: np.ndarray,
        pre_mask: np.ndarray,
        mid_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply consistent spatial augmentations to both pre and mid slices."""
        # Horizontal flip
        if random.random() < 0.5:
            pre = np.fliplr(pre).copy()
            mid = np.fliplr(mid).copy()
            pre_mask = np.fliplr(pre_mask).copy()
            mid_mask = np.fliplr(mid_mask).copy()

        # Vertical flip
        if random.random() < 0.3:
            pre = np.flipud(pre).copy()
            mid = np.flipud(mid).copy()
            pre_mask = np.flipud(pre_mask).copy()
            mid_mask = np.flipud(mid_mask).copy()

        # Gaussian noise (intensity only, not mask)
        if random.random() < 0.5:
            sigma = random.uniform(0, 0.02)
            pre = (pre + np.random.normal(0, sigma, pre.shape)).clip(0, 1)
            mid = (mid + np.random.normal(0, sigma, mid.shape)).clip(0, 1)

        # Brightness jitter
        if random.random() < 0.4:
            delta = random.uniform(-0.1, 0.1)
            pre = (pre + delta).clip(0, 1)
            mid = (mid + delta).clip(0, 1)

        return pre, mid, pre_mask, mid_mask


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = SAM2_IMAGE_SIZE,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """
    Build train (slice-level) and val/test (volume-level) DataLoaders.

    Returns:
        {
            "train": DataLoader[HNTSMRGSliceDataset],
            "val":   DataLoader[HNTSMRGDataset],
            "test":  DataLoader[HNTSMRGDataset],
        }
    """
    train_ds = HNTSMRGSliceDataset(data_dir, split="train", image_size=image_size, augment=True)
    val_ds = HNTSMRGDataset(data_dir, split="val", image_size=image_size)
    test_ds = HNTSMRGDataset(data_dir, split="test", image_size=image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
