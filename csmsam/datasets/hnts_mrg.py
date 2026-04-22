"""
HNTS-MRG 2024 Dataset for CSM-SAM.

Dataset structure (after preprocessing):
    data/processed/
        train/
            patient_001/
                pre_image.nii.gz           — pre-RT T2-weighted MRI
                pre_GTVp.nii.gz            — pre-RT primary tumor mask
                pre_GTVn.nii.gz            — pre-RT nodal metastases mask
                mid_image.nii.gz           — mid-RT T2-weighted MRI
                mid_GTVp.nii.gz            — mid-RT primary tumor mask
                mid_GTVn.nii.gz            — mid-RT nodal metastases mask
                pre_GTVp_registered.nii.gz — (optional) pre-RT GTVp warped to mid-RT grid
                pre_GTVn_registered.nii.gz — (optional) pre-RT GTVn warped to mid-RT grid
                metadata.json              — weeks_elapsed, patient_id, etc.
            ...

    NOTE: HNTS-MRG 2024 SOTA (UW LAIR 0.733, mic-dkfz/"BAMF" 0.727, HiLab 0.725)
    all consume ``preRT_mask_registered`` — the pre-RT mask warped onto the
    mid-RT grid — as a primary input signal. The challenge organizers ship this
    field. If ``pre_{GTVp,GTVn}_registered.nii.gz`` are present in a patient
    dir, the loaders below expose them on the mid-RT image grid. See
    ``data/preprocess.py`` (TODO there) for producing them locally.
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

HNTSMRGSequenceDataset:
    Returns K consecutive mid-RT slices plus matching pre-RT slices and
    the full pre-RT volume (for memory encoding). Used for sequence training.
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

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
    Z-score normalization over foreground (non-zero) voxels, then clip to [0, 1].
    Matches SOTA BraTS/HNTS-MRG preprocessing (nnU-Net, MedNeXt, SwinUNETR all use
    per-modality z-score within the brain mask, not global percentile scaling).
    """
    foreground = volume[volume > 0]
    if foreground.size == 0:
        return volume.astype(np.float32)
    mean = foreground.mean()
    std = foreground.std() + 1e-8
    volume = (volume - mean) / std
    # Clip outliers (covers >99.9% of brain signal) then scale to [0, 1]
    volume = np.clip(volume, -5.0, 5.0)
    volume = (volume + 5.0) / 10.0
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


class _VolumeCache:
    """
    LRU cache of normalized patient volumes.

    Cache is per instance (i.e. per-dataset, which in a DataLoader means
    per-worker because workers get their own copy of the dataset). Stores
    fully-normalized numpy volumes keyed by patient directory path.
    """

    def __init__(self, maxsize: int = 8):
        self.maxsize = maxsize
        self._cache: "OrderedDict[str, dict]" = OrderedDict()

    def get(self, patient_dir: Path) -> dict:
        key = str(patient_dir)
        if key in self._cache:
            # Mark as recently used
            self._cache.move_to_end(key)
            return self._cache[key]

        data = self._load(patient_dir)
        self._cache[key] = data
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)  # evict LRU
        return data

    def _load(self, patient_dir: Path) -> dict:
        """Load and normalize all volumes for a patient."""
        pre_vol = normalize_mri(load_nifti(patient_dir / "pre_image.nii.gz"))
        mid_vol = normalize_mri(load_nifti(patient_dir / "mid_image.nii.gz"))

        def _load_mask(name: str, ref: np.ndarray) -> np.ndarray:
            p = patient_dir / name
            return load_nifti(p) if p.exists() else np.zeros_like(ref)

        pre_gtvp = _load_mask("pre_GTVp.nii.gz", pre_vol)
        pre_gtvn = _load_mask("pre_GTVn.nii.gz", pre_vol)
        mid_gtvp = _load_mask("mid_GTVp.nii.gz", mid_vol)
        mid_gtvn = _load_mask("mid_GTVn.nii.gz", mid_vol)

        # Registered pre-RT masks + image on the mid-RT grid — this is the
        # single biggest input signal that all HNTS-MRG 2024 top-5 teams use.
        # UW LAIR (0.733, #1) feeds the registered masks as auxiliary channels
        # into a mask-aware attention module; mic-dkfz (0.727, #2) and HiLab
        # (0.725, #3) concat them as extra nnU-Net input channels. We expose
        # them here so the SAM2 mask-prompt path can consume them too.
        # Shape follows mid_vol. Zero when file is absent.
        pre_gtvp_reg = _load_mask("pre_GTVp_registered.nii.gz", mid_vol)
        pre_gtvn_reg = _load_mask("pre_GTVn_registered.nii.gz", mid_vol)
        pre_vol_reg_path = patient_dir / "pre_image_registered.nii.gz"
        if pre_vol_reg_path.exists():
            pre_vol_reg = normalize_mri(load_nifti(pre_vol_reg_path))
        else:
            pre_vol_reg = np.zeros_like(mid_vol)

        meta_path = patient_dir / "metadata.json"
        weeks_elapsed = 3
        if meta_path.exists():
            with open(meta_path) as f:
                weeks_elapsed = json.load(f).get("weeks_elapsed", 3)

        return {
            "pre_vol": pre_vol,
            "mid_vol": mid_vol,
            "pre_gtvp": pre_gtvp,
            "pre_gtvn": pre_gtvn,
            "mid_gtvp": mid_gtvp,
            "mid_gtvn": mid_gtvn,
            "pre_vol_registered": pre_vol_reg,
            "pre_gtvp_registered": pre_gtvp_reg,
            "pre_gtvn_registered": pre_gtvn_reg,
            "weeks_elapsed": weeks_elapsed,
            "patient_id": patient_dir.name,
        }


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
        pre_mask_registered:      (N_slices, 1, H, W) — pre-RT mask warped to mid-RT grid
        pre_mask_registered_gtvp: (N_slices, 1, H, W)
        pre_mask_registered_gtvn: (N_slices, 1, H, W)
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = SAM2_IMAGE_SIZE,
        min_mid_tumor_slices: int = 1,
        volume_cache: Optional[_VolumeCache] = None,
    ):
        """
        Args:
            data_dir   : root processed data directory
            split      : "train", "val", or "test"
            image_size : SAM2 input resolution
            min_mid_tumor_slices: filter patients with fewer than N mid-RT tumor slices
            volume_cache: optional shared volume cache
        """
        self.data_dir = Path(data_dir) / split
        self.image_size = image_size
        self.split = split
        self.volume_cache = volume_cache

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
        """Load all volumes for one patient (optionally via cache)."""
        if self.volume_cache is not None:
            return self.volume_cache.get(patient_dir)

        pre_vol = load_nifti(patient_dir / "pre_image.nii.gz")
        mid_vol = load_nifti(patient_dir / "mid_image.nii.gz")

        pre_gtvp = load_nifti(patient_dir / "pre_GTVp.nii.gz") if (patient_dir / "pre_GTVp.nii.gz").exists() else np.zeros_like(pre_vol)
        pre_gtvn = load_nifti(patient_dir / "pre_GTVn.nii.gz") if (patient_dir / "pre_GTVn.nii.gz").exists() else np.zeros_like(pre_vol)
        mid_gtvp = load_nifti(patient_dir / "mid_GTVp.nii.gz") if (patient_dir / "mid_GTVp.nii.gz").exists() else np.zeros_like(mid_vol)
        mid_gtvn = load_nifti(patient_dir / "mid_GTVn.nii.gz") if (patient_dir / "mid_GTVn.nii.gz").exists() else np.zeros_like(mid_vol)
        # Registered pre-RT image + masks (on mid-RT grid). SOTA input.
        pre_gtvp_reg = load_nifti(patient_dir / "pre_GTVp_registered.nii.gz") if (patient_dir / "pre_GTVp_registered.nii.gz").exists() else np.zeros_like(mid_vol)
        pre_gtvn_reg = load_nifti(patient_dir / "pre_GTVn_registered.nii.gz") if (patient_dir / "pre_GTVn_registered.nii.gz").exists() else np.zeros_like(mid_vol)
        pre_vol_reg_path = patient_dir / "pre_image_registered.nii.gz"
        pre_vol_reg = (
            normalize_mri(load_nifti(pre_vol_reg_path))
            if pre_vol_reg_path.exists() else np.zeros_like(mid_vol)
        )

        pre_vol = normalize_mri(pre_vol)
        mid_vol = normalize_mri(mid_vol)

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
            "pre_vol_registered": pre_vol_reg,
            "pre_gtvp_registered": pre_gtvp_reg,
            "pre_gtvn_registered": pre_gtvn_reg,
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
        pre_gtvp_reg = data["pre_gtvp_registered"][:N]
        pre_gtvn_reg = data["pre_gtvn_registered"][:N]

        # Convert each slice to SAM2 tensor
        pre_images = torch.stack([to_rgb_tensor(pre_vol[i], self.image_size) for i in range(N)])
        mid_images = torch.stack([to_rgb_tensor(mid_vol[i], self.image_size) for i in range(N)])
        pre_masks_gtvp = torch.stack([to_mask_tensor(pre_gtvp[i], self.image_size) for i in range(N)])
        pre_masks_gtvn = torch.stack([to_mask_tensor(pre_gtvn[i], self.image_size) for i in range(N)])
        mid_masks_gtvp = torch.stack([to_mask_tensor(mid_gtvp[i], self.image_size) for i in range(N)])
        mid_masks_gtvn = torch.stack([to_mask_tensor(mid_gtvn[i], self.image_size) for i in range(N)])
        pre_mask_reg_gtvp = torch.stack([to_mask_tensor(pre_gtvp_reg[i], self.image_size) for i in range(N)])
        pre_mask_reg_gtvn = torch.stack([to_mask_tensor(pre_gtvn_reg[i], self.image_size) for i in range(N)])

        return {
            "pre_images": pre_images,                              # (N, 3, H, W)
            "mid_images": mid_images,
            "pre_masks": (pre_masks_gtvp + pre_masks_gtvn).clamp(0, 1),  # combined
            "mid_masks": (mid_masks_gtvp + mid_masks_gtvn).clamp(0, 1),
            "pre_masks_gtvp": pre_masks_gtvp,
            "pre_masks_gtvn": pre_masks_gtvn,
            "mid_masks_gtvp": mid_masks_gtvp,
            "mid_masks_gtvn": mid_masks_gtvn,
            # Pre-RT masks warped onto the mid-RT grid (HNTS-MRG SOTA input).
            # All zero when *_registered.nii.gz files are absent — TODO:
            # data/preprocess.py currently does not produce these files, so
            # these tensors will be zero on the shipping data and must be
            # filled in before attempting to reproduce UW-LAIR / BAMF numbers.
            "pre_mask_registered": (pre_mask_reg_gtvp + pre_mask_reg_gtvn).clamp(0, 1),
            "pre_mask_registered_gtvp": pre_mask_reg_gtvp,
            "pre_mask_registered_gtvn": pre_mask_reg_gtvn,
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

    Uses an LRU volume cache (per-dataloader-worker) to avoid reloading the
    patient's NIfTI volumes on every __getitem__.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = SAM2_IMAGE_SIZE,
        augment: bool = True,
        tumor_ratio: float = 0.7,
        cache_metadata: bool = True,
        volume_cache_size: int = 8,
        volume_cache: Optional[_VolumeCache] = None,
    ):
        """
        Args:
            tumor_ratio: fraction of returned slices that contain tumor
            volume_cache_size: number of patient volumes to cache in memory
            volume_cache: optional externally-provided cache (shared across datasets)
        """
        self.data_dir = Path(data_dir) / split
        self.image_size = image_size
        self.augment = augment and (split == "train")
        self.tumor_ratio = tumor_ratio
        self.volume_cache = volume_cache if volume_cache is not None else _VolumeCache(maxsize=volume_cache_size)

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
        """Load one pre/mid slice pair (via volume cache)."""
        data = self.volume_cache.get(patient_dir)

        pre_vol = data["pre_vol"]
        mid_vol = data["mid_vol"]
        pre_gtvp_vol = data["pre_gtvp"]
        pre_gtvn_vol = data["pre_gtvn"]
        mid_gtvp_vol = data["mid_gtvp"]
        mid_gtvn_vol = data["mid_gtvn"]
        # Registered pre-RT volume + masks on the mid-RT grid (SOTA signal).
        pre_vol_reg = data["pre_vol_registered"]
        pre_gtvp_reg_vol = data["pre_gtvp_registered"]
        pre_gtvn_reg_vol = data["pre_gtvn_registered"]

        # Clamp slice index to valid range. slice_idx comes from mid-RT
        # enumeration (see tumor_slices / bg_slices build loop), so it is
        # already a mid-RT index. Clamp pre_vol separately — unregistered
        # pre-RT may have a different slice count.
        N_mid = mid_vol.shape[0]
        slice_idx = min(slice_idx, N_mid - 1)
        pre_idx_legacy = min(slice_idx, pre_vol.shape[0] - 1)

        # Unregistered pre-RT slice (legacy — kept for backward compat).
        pre_slice = pre_vol[pre_idx_legacy]
        pre_gtvp_sl = pre_gtvp_vol[pre_idx_legacy] > 0
        pre_gtvn_sl = pre_gtvn_vol[pre_idx_legacy] > 0

        # Mid-RT slice + per-structure targets.
        mid_slice = mid_vol[slice_idx]
        mid_gtvp_sl = mid_gtvp_vol[slice_idx] > 0
        mid_gtvn_sl = mid_gtvn_vol[slice_idx] > 0

        # Registered pre-RT image + masks, same slice index as mid (same grid).
        pre_slice_reg = pre_vol_reg[slice_idx]
        pre_gtvp_reg_sl = pre_gtvp_reg_vol[slice_idx] > 0
        pre_gtvn_reg_sl = pre_gtvn_reg_vol[slice_idx] > 0

        # Combined masks
        pre_mask = (pre_gtvp_vol[pre_idx_legacy] + pre_gtvn_vol[pre_idx_legacy]) > 0
        mid_mask = (mid_gtvp_vol[slice_idx] + mid_gtvn_vol[slice_idx]) > 0

        # Augmentation — flip/noise applied consistently to ALL spatially-
        # aligned slices. Mid and pre_registered share the mid-RT grid, so
        # they receive the same transform. Legacy pre-RT (on its own grid)
        # gets a separate flip of its own since it is not spatially aligned
        # with the mid frame anyway.
        if self.augment:
            (mid_slice, pre_slice_reg,
             pre_gtvp_reg_sl_f, pre_gtvn_reg_sl_f,
             mid_gtvp_sl_f, mid_gtvn_sl_f,
             mid_mask_f) = self._augment_aligned(
                mid_slice, pre_slice_reg,
                pre_gtvp_reg_sl.astype(np.float32), pre_gtvn_reg_sl.astype(np.float32),
                mid_gtvp_sl.astype(np.float32), mid_gtvn_sl.astype(np.float32),
                mid_mask.astype(np.float32),
            )
            pre_gtvp_reg_sl = pre_gtvp_reg_sl_f > 0.5
            pre_gtvn_reg_sl = pre_gtvn_reg_sl_f > 0.5
            mid_gtvp_sl = mid_gtvp_sl_f > 0.5
            mid_gtvn_sl = mid_gtvn_sl_f > 0.5
            mid_mask = mid_mask_f > 0.5
            # legacy unregistered pre-RT (separate transform — different grid)
            pre_slice, pre_mask = self._augment_pair(
                pre_slice, pre_mask.astype(np.float32),
            )

        weeks_elapsed = data.get("weeks_elapsed", 3)

        return {
            # Legacy unregistered pre-RT (kept for backward compatibility
            # with the full-volume cross-session memory encoder path).
            "pre_image": to_rgb_tensor(pre_slice.astype(np.float32), self.image_size),
            "pre_mask": to_mask_tensor(pre_mask.astype(np.float32), self.image_size),
            "pre_mask_gtvp": to_mask_tensor(pre_gtvp_sl.astype(np.float32), self.image_size),
            "pre_mask_gtvn": to_mask_tensor(pre_gtvn_sl.astype(np.float32), self.image_size),

            # Registered pre-RT (SOTA input — all HNTS-MRG top teams use this).
            # Same spatial grid as mid_image, so SAM2's dense mask prompt
            # actually aligns with tumor locations in mid-RT space.
            "pre_image_registered":     to_rgb_tensor(pre_slice_reg.astype(np.float32), self.image_size),
            "pre_mask_registered":      to_mask_tensor(((pre_gtvp_reg_sl | pre_gtvn_reg_sl)).astype(np.float32), self.image_size),
            "pre_mask_gtvp_registered": to_mask_tensor(pre_gtvp_reg_sl.astype(np.float32), self.image_size),
            "pre_mask_gtvn_registered": to_mask_tensor(pre_gtvn_reg_sl.astype(np.float32), self.image_size),

            # Mid-RT
            "mid_image":     to_rgb_tensor(mid_slice.astype(np.float32), self.image_size),
            "mid_mask":      to_mask_tensor(mid_mask.astype(np.float32), self.image_size),
            "mid_mask_gtvp": to_mask_tensor(mid_gtvp_sl.astype(np.float32), self.image_size),
            "mid_mask_gtvn": to_mask_tensor(mid_gtvn_sl.astype(np.float32), self.image_size),

            "weeks_elapsed": weeks_elapsed,
            "patient_id": patient_dir.name,
            "slice_idx": slice_idx,
        }

    def _augment_aligned(self, *arrs):
        """Apply one set of random flips/intensity aug to N spatially-aligned (H,W) arrays.
        First array is the "reference" intensity image for noise/brightness;
        masks (float32 0/1) only receive the geometric transform.
        """
        # Decide flips once
        hflip = random.random() < 0.5
        vflip = random.random() < 0.3
        add_noise = random.random() < 0.5
        noise_sigma = random.uniform(0, 0.02) if add_noise else 0.0
        add_bright = random.random() < 0.4
        delta = random.uniform(-0.1, 0.1) if add_bright else 0.0

        out = []
        for i, a in enumerate(arrs):
            a = a.copy()
            if hflip: a = np.fliplr(a).copy()
            if vflip: a = np.flipud(a).copy()
            # Intensity aug only for images (first 2 entries: mid_slice, pre_slice_reg).
            if i < 2:
                if noise_sigma > 0:
                    a = (a + np.random.normal(0, noise_sigma, a.shape)).clip(0, 1)
                if delta != 0.0:
                    a = (a + delta).clip(0, 1)
            out.append(a)
        return tuple(out)

    def _augment_pair(self, img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flip/noise aug for a single (image, mask) pair on its own grid."""
        if random.random() < 0.5:
            img = np.fliplr(img).copy(); mask = np.fliplr(mask).copy()
        if random.random() < 0.3:
            img = np.flipud(img).copy(); mask = np.flipud(mask).copy()
        if random.random() < 0.5:
            sigma = random.uniform(0, 0.02)
            img = (img + np.random.normal(0, sigma, img.shape)).clip(0, 1)
        if random.random() < 0.4:
            delta = random.uniform(-0.1, 0.1)
            img = (img + delta).clip(0, 1)
        return img, mask

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


class HNTSMRGSequenceDataset(Dataset):
    """
    Sequence-level dataset for training CSM-SAM with K consecutive slices.

    Each __getitem__ returns a dict containing:
        pre_images:        (N_pre, 3, H, W) — full pre-RT volume (for memory encoding)
        pre_masks:         (N_pre, 1, H, W) — combined pre-RT mask for whole volume
        mid_images:        (K, 3, H, W)     — K consecutive mid-RT slices
        pre_images_seq:    (K, 3, H, W)     — same-index pre-RT slices (for change head)
        pre_mask_gtvp:     (K, 1, H, W)
        pre_mask_gtvn:     (K, 1, H, W)
        mid_mask_gtvp:     (K, 1, H, W)
        mid_mask_gtvn:     (K, 1, H, W)
        weeks_elapsed:     int
        patient_id:        str
        start_slice:       int

    Sampling: with probability `tumor_window_ratio`, the K-slice window is
    drawn uniformly from windows that contain at least one tumor pixel in
    the mid-RT mask; otherwise uniformly across all valid starts.
    Augmentations (flip / noise / brightness) are applied consistently to
    every slice in the sampled window.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = SAM2_IMAGE_SIZE,
        augment: bool = True,
        sequence_length: int = 4,
        tumor_window_ratio: float = 0.8,
        volume_cache_size: int = 8,
        volume_cache: Optional[_VolumeCache] = None,
    ):
        self.data_dir = Path(data_dir) / split
        self.image_size = image_size
        self.augment = augment and (split == "train")
        self.sequence_length = int(sequence_length)
        self.tumor_window_ratio = float(tumor_window_ratio)
        self.volume_cache = volume_cache if volume_cache is not None else _VolumeCache(maxsize=volume_cache_size)

        self.patient_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and (d / "mid_image.nii.gz").exists()
        ])
        if not self.patient_dirs:
            raise FileNotFoundError(
                f"No patient directories found in {self.data_dir}. "
                "Run data/preprocess.py first."
            )

        # Pre-compute, per patient, the list of valid K-windows and the subset
        # of windows that contain tumor. This scan only reads the mid masks
        # (small) — the expensive volume load stays in the cache on first use.
        self.patient_meta: list[dict] = []
        K = self.sequence_length
        print(
            f"[HNTSMRGSequenceDataset] Indexing {len(self.patient_dirs)} {split} "
            f"patients (K={K})..."
        )
        for pdir in self.patient_dirs:
            try:
                mid_gtvp = load_nifti(pdir / "mid_GTVp.nii.gz") if (pdir / "mid_GTVp.nii.gz").exists() else None
                mid_gtvn = load_nifti(pdir / "mid_GTVn.nii.gz") if (pdir / "mid_GTVn.nii.gz").exists() else None
                if mid_gtvp is None and mid_gtvn is None:
                    continue
                ref = mid_gtvp if mid_gtvp is not None else mid_gtvn
                if mid_gtvp is None:
                    mid_gtvp = np.zeros_like(ref)
                if mid_gtvn is None:
                    mid_gtvn = np.zeros_like(ref)
                tumor_per_slice = ((mid_gtvp + mid_gtvn) > 0).reshape(
                    mid_gtvp.shape[0], -1
                ).any(axis=1)  # (D,)
                D = int(tumor_per_slice.shape[0])
                if D < K:
                    continue
                max_start = D - K
                valid_starts = list(range(0, max_start + 1))
                tumor_starts = [
                    s for s in valid_starts if tumor_per_slice[s:s + K].any()
                ]
                self.patient_meta.append({
                    "dir": pdir,
                    "n_slices": D,
                    "valid_starts": valid_starts,
                    "tumor_starts": tumor_starts,
                })
            except Exception as e:
                print(f"  Warning: skipping {pdir.name}: {e}")

        n_tumor_windows = sum(len(m["tumor_starts"]) for m in self.patient_meta)
        n_windows = sum(len(m["valid_starts"]) for m in self.patient_meta)
        print(
            f"  {len(self.patient_meta)} patients, {n_tumor_windows} tumor windows "
            f"/ {n_windows} total windows"
        )

    def __len__(self) -> int:
        # One sequence per patient per epoch; shuffling re-samples windows each epoch.
        return len(self.patient_meta)

    def _pick_start(self, meta: dict) -> int:
        tumor_starts = meta["tumor_starts"]
        valid_starts = meta["valid_starts"]
        if tumor_starts and random.random() < self.tumor_window_ratio:
            return random.choice(tumor_starts)
        return random.choice(valid_starts)

    def __getitem__(self, idx: int) -> dict:
        meta = self.patient_meta[idx]
        patient_dir: Path = meta["dir"]
        data = self.volume_cache.get(patient_dir)

        pre_vol = data["pre_vol"]
        mid_vol = data["mid_vol"]
        pre_gtvp_vol = data["pre_gtvp"]
        pre_gtvn_vol = data["pre_gtvn"]
        mid_gtvp_vol = data["mid_gtvp"]
        mid_gtvn_vol = data["mid_gtvn"]

        D_pre = int(pre_vol.shape[0])
        D_mid = int(mid_vol.shape[0])
        K = self.sequence_length

        # Mid-RT window
        start = self._pick_start(meta)
        start = min(max(0, start), max(0, D_mid - K))
        end = start + K

        mid_slices = mid_vol[start:end]
        mid_gtvp_sl = mid_gtvp_vol[start:end]
        mid_gtvn_sl = mid_gtvn_vol[start:end]

        # Pre-RT: use the SAME slice indices for the aligned per-slice signal
        # (change head). Clamp to pre volume length if shorter.
        pre_start = min(start, max(0, D_pre - K))
        pre_end = pre_start + K
        pre_seq_slices = pre_vol[pre_start:pre_end]
        pre_seq_gtvp = pre_gtvp_vol[pre_start:pre_end]
        pre_seq_gtvn = pre_gtvn_vol[pre_start:pre_end]

        # Consistent augmentation decisions across the whole window
        aug_decisions = self._decide_augmentations()

        # Build mid-seq tensors
        mid_images_list = []
        mid_mask_gtvp_list = []
        mid_mask_gtvn_list = []
        for k in range(K):
            im = mid_slices[k].astype(np.float32)
            gp = (mid_gtvp_sl[k] > 0).astype(np.float32)
            gn = (mid_gtvn_sl[k] > 0).astype(np.float32)
            if self.augment:
                im, gp, gn = self._apply_augmentations(im, [gp, gn], aug_decisions)
            mid_images_list.append(to_rgb_tensor(im, self.image_size))
            mid_mask_gtvp_list.append(to_mask_tensor(gp, self.image_size))
            mid_mask_gtvn_list.append(to_mask_tensor(gn, self.image_size))
        mid_images = torch.stack(mid_images_list)
        mid_mask_gtvp = torch.stack(mid_mask_gtvp_list)
        mid_mask_gtvn = torch.stack(mid_mask_gtvn_list)

        # Build per-slice pre-RT sequence tensors (same aug decisions)
        pre_seq_images_list = []
        pre_seq_mask_gtvp_list = []
        pre_seq_mask_gtvn_list = []
        for k in range(K):
            im = pre_seq_slices[k].astype(np.float32)
            gp = (pre_seq_gtvp[k] > 0).astype(np.float32)
            gn = (pre_seq_gtvn[k] > 0).astype(np.float32)
            if self.augment:
                im, gp, gn = self._apply_augmentations(im, [gp, gn], aug_decisions)
            pre_seq_images_list.append(to_rgb_tensor(im, self.image_size))
            pre_seq_mask_gtvp_list.append(to_mask_tensor(gp, self.image_size))
            pre_seq_mask_gtvn_list.append(to_mask_tensor(gn, self.image_size))
        pre_images_seq = torch.stack(pre_seq_images_list)
        pre_mask_gtvp = torch.stack(pre_seq_mask_gtvp_list)
        pre_mask_gtvn = torch.stack(pre_seq_mask_gtvn_list)

        # Full pre-RT volume for memory encoding (applies only flips from aug
        # decisions — noise/brightness aren't geometry and we keep the pre
        # memory stream consistent-but-not-noisy).
        full_pre_images_list = []
        full_pre_masks_list = []
        for k in range(D_pre):
            im = pre_vol[k].astype(np.float32)
            m = ((pre_gtvp_vol[k] + pre_gtvn_vol[k]) > 0).astype(np.float32)
            if self.augment:
                im, m = self._apply_geometry_only(im, m, aug_decisions)
            full_pre_images_list.append(to_rgb_tensor(im, self.image_size))
            full_pre_masks_list.append(to_mask_tensor(m, self.image_size))
        pre_images = torch.stack(full_pre_images_list)
        pre_masks = torch.stack(full_pre_masks_list)

        return {
            "pre_images": pre_images,               # (N_pre, 3, H, W)
            "pre_masks": pre_masks,                 # (N_pre, 1, H, W)
            "mid_images": mid_images,               # (K, 3, H, W)
            "pre_images_seq": pre_images_seq,       # (K, 3, H, W)
            "pre_mask_gtvp": pre_mask_gtvp,         # (K, 1, H, W)
            "pre_mask_gtvn": pre_mask_gtvn,         # (K, 1, H, W)
            "mid_mask_gtvp": mid_mask_gtvp,         # (K, 1, H, W)
            "mid_mask_gtvn": mid_mask_gtvn,         # (K, 1, H, W)
            "weeks_elapsed": int(data.get("weeks_elapsed", 3)),
            "patient_id": patient_dir.name,
            "start_slice": int(start),
        }

    # --- Augmentation helpers -------------------------------------------------

    def _decide_augmentations(self) -> dict:
        """Sample a single set of augmentation decisions for one K-window."""
        return {
            "hflip": random.random() < 0.5,
            "vflip": random.random() < 0.3,
            "noise_sigma": random.uniform(0, 0.02) if random.random() < 0.5 else 0.0,
            "brightness": random.uniform(-0.1, 0.1) if random.random() < 0.4 else 0.0,
        }

    def _apply_augmentations(
        self,
        image: np.ndarray,
        masks: list[np.ndarray],
        d: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the sampled augmentations to one image + two masks."""
        if d["hflip"]:
            image = np.fliplr(image).copy()
            masks = [np.fliplr(m).copy() for m in masks]
        if d["vflip"]:
            image = np.flipud(image).copy()
            masks = [np.flipud(m).copy() for m in masks]
        if d["noise_sigma"] > 0:
            image = (image + np.random.normal(0, d["noise_sigma"], image.shape)).clip(0, 1)
        if d["brightness"] != 0.0:
            image = (image + d["brightness"]).clip(0, 1)
        image = image.astype(np.float32, copy=False)
        return image, masks[0], masks[1]

    def _apply_geometry_only(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        d: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply only geometric aug (flips) — used for full pre-RT volume."""
        if d["hflip"]:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if d["vflip"]:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        return image, mask


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = SAM2_IMAGE_SIZE,
    pin_memory: bool = True,
    sequence_length: int = 0,
    tumor_window_ratio: float = 0.8,
) -> dict[str, DataLoader]:
    """
    Build train (slice- or sequence-level) and val/test (volume-level) DataLoaders.

    Args:
        sequence_length: if > 1, use HNTSMRGSequenceDataset for training; else
                         fall back to the slice-level dataset.
        tumor_window_ratio: ratio of training windows biased to contain tumor
                            (only used when sequence_length > 1).

    Returns:
        {
            "train": DataLoader[HNTSMRGSliceDataset | HNTSMRGSequenceDataset],
            "val":   DataLoader[HNTSMRGDataset],
            "test":  DataLoader[HNTSMRGDataset],
        }
    """
    if sequence_length and sequence_length > 1:
        train_ds: Dataset = HNTSMRGSequenceDataset(
            data_dir,
            split="train",
            image_size=image_size,
            augment=True,
            sequence_length=sequence_length,
            tumor_window_ratio=tumor_window_ratio,
        )
    else:
        train_ds = HNTSMRGSliceDataset(
            data_dir, split="train", image_size=image_size, augment=True
        )

    val_ds = HNTSMRGDataset(data_dir, split="val", image_size=image_size)
    test_ds = HNTSMRGDataset(data_dir, split="test", image_size=image_size)

    # When DDP is active, shard the train split across ranks so each GPU sees
    # a different 1/world_size of the data — otherwise every rank iterates the
    # full dataset (wasted compute) and wall-time per epoch doesn't scale with
    # GPU count.
    ddp_active = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if ddp_active else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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
