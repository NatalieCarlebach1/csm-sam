"""
BraTS-GLI 2024 Post-Treatment Glioma Longitudinal MRI Dataset for CSM-SAM.

Dataset layout (after zip extraction):
    data/raw/BraTS_GLI/
        Training.zip   (~14 GB)
        Testing.zip    (~2.5 GB)
        Training/
            BraTS-GLI-{ID:05d}-{TP:03d}/
                BraTS-GLI-{ID:05d}-{TP:03d}-t1c.nii.gz
                BraTS-GLI-{ID:05d}-{TP:03d}-t1n.nii.gz
                BraTS-GLI-{ID:05d}-{TP:03d}-t2f.nii.gz
                BraTS-GLI-{ID:05d}-{TP:03d}-t2w.nii.gz
                BraTS-GLI-{ID:05d}-{TP:03d}-seg.nii.gz
            ...
        Testing/
            ...

Timepoints:
    TP=000  → baseline    (analog to HNTS pre-RT)
    TP=001  → follow-up   (analog to HNTS mid-RT)  — ~118 patients have this

Segmentation labels:
    0 = background
    1 = necrotic tumor core
    2 = peritumoral edema
    3 = enhancing tumor
    4 = resection cavity       (excluded from the binary tumor mask)

Modalities:
    t1c = T1 post-contrast,  t1n = T1 native,
    t2f = T2 FLAIR (default), t2w = T2 weighted
    'all' → composite 3-channel image from (t1c, t2f, t2w)

    TODO: BraTS-GLI 2024 SOTA (nnU-Net / MedNeXt / SwinUNETR ensembles, all
    published top entries) consumes **all 4 modalities** (t1c, t1n, t2w, t2f)
    as a 4-channel 3D input. Our 'all' composite currently stacks only 3
    channels (t1c, t2f, t2w) because SAM2's ViT-H expects a 3-channel image.
    When wiring CSM-SAM's image encoder we will need either (a) a 4→3 input
    projection at the stem, or (b) a 4-channel variant that bypasses SAM2's
    stem. Dropping t1n vs SOTA is a known ~few-DSC-point handicap on BraTS.

Classes:
    BraTSGLIDataset        — volume-level pairs for validation / inference
    BraTSGLISliceDataset   — 2D slice pairs for training (tumor-biased sampler)
    build_dataloaders(...) — train/val/test DataLoader factory
"""

from __future__ import annotations

import random
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Reuse HNTS helpers — BraTS is also NIfTI MRI so these apply unchanged.
from .hnts_mrg import (
    SAM2_IMAGE_SIZE,
    SAM2_MEAN,
    SAM2_STD,
    load_nifti,
    normalize_mri,
    to_mask_tensor,
    to_rgb_tensor,
)


VALID_MODALITIES = ("t1c", "t1n", "t2f", "t2w", "all", "all4")
RESECTION_LABEL = 4  # excluded from binary tumor mask
BRATS_FOLLOWUP_WEEKS_DEFAULT = 12  # BraTS follow-up ~= 3 months


# ---------------------------------------------------------------------------
# Extraction + patient discovery
# ---------------------------------------------------------------------------

def _extract_zip_if_needed(data_dir: Path, split_name: str) -> Path:
    """
    Ensure ``data_dir/<split_name>`` exists on disk by extracting the
    ``data_dir/<split_name>.zip`` archive once on first use.

    Returns the path to the extracted split directory.
    """
    split_dir = data_dir / split_name
    zip_path = data_dir / f"{split_name}.zip"

    # Already extracted and non-empty → nothing to do.
    if split_dir.exists() and any(split_dir.iterdir()):
        return split_dir

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Neither extracted directory {split_dir} nor archive {zip_path} exists."
        )

    print(f"[BraTS-GLI] Extracting {zip_path} → {data_dir} (one-time, may take a while)...")
    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    # Some BraTS archives unpack to a wrapper dir with a different name
    # (e.g. "BraTS2024-...-Training"). If our expected split_dir doesn't
    # exist but a single sibling directory does, point split_dir at it.
    if not split_dir.exists():
        candidates = [
            d for d in data_dir.iterdir()
            if d.is_dir() and split_name.lower() in d.name.lower()
        ]
        if len(candidates) == 1:
            split_dir = candidates[0]
        else:
            raise FileNotFoundError(
                f"Expected {split_dir} after extraction; found candidates: {candidates}"
            )
    return split_dir


def _parse_patient_folder(name: str) -> Optional[tuple[int, int]]:
    """Parse 'BraTS-GLI-{ID:05d}-{TP:03d}' → (patient_id_int, timepoint_int)."""
    parts = name.split("-")
    # Expect at least ["BraTS", "GLI", "<id>", "<tp>"]
    if len(parts) < 4 or parts[0] != "BraTS" or parts[1] != "GLI":
        return None
    try:
        return int(parts[2]), int(parts[3])
    except ValueError:
        return None


def _discover_patients(split_dir: Path) -> dict[int, dict[int, Path]]:
    """
    Walk ``split_dir`` and return a mapping::

        {patient_id:int -> {timepoint:int -> Path(patient_folder)}}
    """
    patients: dict[int, dict[int, Path]] = {}
    for d in sorted(split_dir.iterdir()):
        if not d.is_dir():
            continue
        parsed = _parse_patient_folder(d.name)
        if parsed is None:
            continue
        pid, tp = parsed
        patients.setdefault(pid, {})[tp] = d
    return patients


def _modality_files_exist(pdir: Path, modality: str) -> bool:
    """Check whether the NIfTI files required for ``modality`` exist in pdir."""
    base = pdir.name
    required: list[str] = []
    if modality in ("all", "all4"):
        mods = ("t1c", "t2f", "t2w") if modality == "all" else ("t1c", "t1n", "t2f", "t2w")
        required = [f"{base}-{m}.nii.gz" for m in mods]
    else:
        required = [f"{base}-{modality}.nii.gz"]
    required.append(f"{base}-seg.nii.gz")
    return all((pdir / r).exists() for r in required)


# ---------------------------------------------------------------------------
# Volume loading + multi-modal compositing
# ---------------------------------------------------------------------------

def _load_modality_volume(pdir: Path, modality: str) -> np.ndarray:
    """
    Load normalized intensity volume(s) for a patient folder.

    Single-modality returns (D, H, W); all returns (3, D, H, W);
    all4 returns (4, D, H, W) with t1c+t1n+t2f+t2w.
    """
    base = pdir.name
    if modality in ("all", "all4"):
        mods = ("t1c", "t2f", "t2w") if modality == "all" else ("t1c", "t1n", "t2f", "t2w")
        channels = []
        for m in mods:
            vol = load_nifti(pdir / f"{base}-{m}.nii.gz")
            channels.append(normalize_mri(vol))
        return np.stack(channels, axis=0)  # (3 or 4, D, H, W)
    vol = load_nifti(pdir / f"{base}-{modality}.nii.gz")
    return normalize_mri(vol)


def _load_seg_volume(pdir: Path) -> np.ndarray:
    """Load raw segmentation label volume (int-valued, float32)."""
    base = pdir.name
    return load_nifti(pdir / f"{base}-seg.nii.gz")


def _to_rgb_multimodal(volume_4d: np.ndarray, slice_idx: int, size: int) -> torch.Tensor:
    """
    Build a normalized N-channel tensor from a (N, D, H, W) multi-modal volume.
    """
    chans = volume_4d[:, slice_idx]  # (N, H, W)
    N = chans.shape[0]
    t = torch.from_numpy(chans.astype(np.float32)).unsqueeze(0)  # (1, N, H, W)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    t = t.squeeze(0)  # (N, H, W)
    mean = SAM2_MEAN if N == 3 else torch.cat([SAM2_MEAN, SAM2_MEAN[-1:]])
    std  = SAM2_STD  if N == 3 else torch.cat([SAM2_STD,  SAM2_STD[-1:]])
    t = (t - mean[:, None, None]) / std[:, None, None]
    return t


def _slice_to_rgb(volume: np.ndarray, slice_idx: int, modality: str, size: int) -> torch.Tensor:
    """Dispatch to grayscale-replicated or multi-modal RGB tensor construction."""
    if modality in ("all", "all4"):
        return _to_rgb_multimodal(volume, slice_idx, size)
    return to_rgb_tensor(volume[slice_idx].astype(np.float32), size)


def _binary_tumor_mask(seg_vol: np.ndarray) -> np.ndarray:
    """Binary tumor mask: any non-zero label EXCEPT resection cavity (4)."""
    return ((seg_vol > 0) & (seg_vol != RESECTION_LABEL)).astype(np.float32)


# ---------------------------------------------------------------------------
# Pair / singleton index building with train / val split
# ---------------------------------------------------------------------------

def _build_split_indices(
    data_dir: Path,
    split: str,
    modality: str,
    include_singletons: bool,
    val_fraction: float = 0.10,
    seed: int = 1234,
) -> tuple[list[dict], list[dict]]:
    """
    Return (pair_records, singleton_records) for the requested dataset split.

    Pair record: {'patient_id', 'pre_dir', 'mid_dir'}
    Singleton record: {'patient_id', 'pre_dir'}

    For split='train' / 'val' we read the Training/ folder and carve off the
    last val_fraction of patient IDs (sorted) as validation. split='test'
    reads Testing/.
    """
    split = split.lower()
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be one of train/val/test, got {split!r}")

    source_split = "Training" if split in ("train", "val") else "Testing"
    split_dir = _extract_zip_if_needed(data_dir, source_split)
    patients = _discover_patients(split_dir)

    pair_pids: list[int] = []
    single_pids: list[int] = []
    for pid, tps in patients.items():
        has0 = 0 in tps and _modality_files_exist(tps[0], modality)
        has1 = 1 in tps and _modality_files_exist(tps[1], modality)
        if has0 and has1:
            pair_pids.append(pid)
        elif has0:
            single_pids.append(pid)

    pair_pids.sort()
    single_pids.sort()

    if split in ("train", "val"):
        # Deterministic train/val partition on the sorted pair list.
        n_pairs = len(pair_pids)
        n_val = max(1, int(round(n_pairs * val_fraction))) if n_pairs > 0 else 0
        rng = random.Random(seed)
        shuffled = pair_pids.copy()
        rng.shuffle(shuffled)
        val_pids = set(shuffled[:n_val])
        if split == "train":
            selected_pairs = [p for p in pair_pids if p not in val_pids]
            selected_singles = single_pids if include_singletons else []
        else:  # val
            selected_pairs = [p for p in pair_pids if p in val_pids]
            selected_singles = []  # val never uses singletons
    else:  # test
        selected_pairs = pair_pids
        selected_singles = single_pids if include_singletons else []

    pair_records = [
        {
            "patient_id": f"BraTS-GLI-{pid:05d}",
            "pre_dir": patients[pid][0],
            "mid_dir": patients[pid][1],
        }
        for pid in selected_pairs
    ]
    singleton_records = [
        {
            "patient_id": f"BraTS-GLI-{pid:05d}",
            "pre_dir": patients[pid][0],
        }
        for pid in selected_singles
    ]
    return pair_records, singleton_records


# ---------------------------------------------------------------------------
# Volume-level dataset
# ---------------------------------------------------------------------------

class BraTSGLIDataset(Dataset):
    """
    Volume-level longitudinal BraTS-GLI dataset.

    One sample = one patient's (TP000 baseline, TP001 follow-up) pair. Slices
    are truncated to ``min(pre_D, mid_D)`` to align depth.

    __getitem__ returns::

        {
            "pre_image":           (N, 3, H, W)  float32 SAM2-normalized
            "mid_image":           (N, 3, H, W)
            "pre_mask":            (N, 1, H, W)  float32 binary (tumor vs bg, no cavity)
            "mid_mask":            (N, 1, H, W)
            "pre_mask_semantic":   (N, H, W)     long    raw labels {0..4}
            "mid_mask_semantic":   (N, H, W)     long
            "weeks_elapsed":       int
            "patient_id":          str
        }
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        modality: str = "t2f",
        image_size: int = SAM2_IMAGE_SIZE,
        sequence_length: int = 0,  # reserved; 0 ⇒ full 3D slice stack
        include_singletons: bool = False,
        weeks_elapsed: int = BRATS_FOLLOWUP_WEEKS_DEFAULT,
        val_fraction: float = 0.10,
        n_folds: int = 1,
        fold: int = 0,
    ):
        if modality not in VALID_MODALITIES:
            raise ValueError(f"modality must be one of {VALID_MODALITIES}, got {modality!r}")

        self.data_dir = Path(data_dir)
        self.split = split
        self.modality = modality
        self.image_size = image_size
        self.sequence_length = int(sequence_length)
        self.include_singletons = bool(include_singletons)
        self.weeks_elapsed = int(weeks_elapsed)

        self.pair_records, self.singleton_records = _build_split_indices(
            self.data_dir,
            split=split,
            modality=modality,
            include_singletons=include_singletons,
            val_fraction=val_fraction,
        )

        if not self.pair_records:
            raise FileNotFoundError(
                f"[BraTSGLIDataset] No longitudinal pairs found for split={split!r} "
                f"in {self.data_dir}. Extracted correctly?"
            )

        print(
            f"[BraTSGLIDataset] {split}: {len(self.pair_records)} longitudinal pairs"
            + (f", {len(self.singleton_records)} singletons" if include_singletons else "")
        )

    def __len__(self) -> int:
        return len(self.pair_records)

    def _load_patient_pair(self, rec: dict) -> dict:
        pre_vol = _load_modality_volume(rec["pre_dir"], self.modality)
        mid_vol = _load_modality_volume(rec["mid_dir"], self.modality)
        pre_seg = _load_seg_volume(rec["pre_dir"])
        mid_seg = _load_seg_volume(rec["mid_dir"])
        return {
            "pre_vol": pre_vol,
            "mid_vol": mid_vol,
            "pre_seg": pre_seg,
            "mid_seg": mid_seg,
        }

    def _depth(self, vol: np.ndarray) -> int:
        # (D, H, W) or (3, D, H, W) → depth axis
        return int(vol.shape[1]) if vol.ndim == 4 else int(vol.shape[0])

    def _slice_vol(self, vol: np.ndarray, end: int) -> np.ndarray:
        """Truncate depth dimension to [0:end]."""
        return vol[:, :end] if vol.ndim == 4 else vol[:end]

    def __getitem__(self, idx: int) -> dict:
        rec = self.pair_records[idx]
        data = self._load_patient_pair(rec)

        D_pre = self._depth(data["pre_vol"])
        D_mid = self._depth(data["mid_vol"])
        D_seg_pre = int(data["pre_seg"].shape[0])
        D_seg_mid = int(data["mid_seg"].shape[0])
        N = min(D_pre, D_mid, D_seg_pre, D_seg_mid)

        pre_vol = self._slice_vol(data["pre_vol"], N)
        mid_vol = self._slice_vol(data["mid_vol"], N)
        pre_seg = data["pre_seg"][:N]
        mid_seg = data["mid_seg"][:N]

        pre_bin = _binary_tumor_mask(pre_seg)
        mid_bin = _binary_tumor_mask(mid_seg)

        pre_images = torch.stack([
            _slice_to_rgb(pre_vol, i, self.modality, self.image_size) for i in range(N)
        ])
        mid_images = torch.stack([
            _slice_to_rgb(mid_vol, i, self.modality, self.image_size) for i in range(N)
        ])
        pre_masks = torch.stack([to_mask_tensor(pre_bin[i], self.image_size) for i in range(N)])
        mid_masks = torch.stack([to_mask_tensor(mid_bin[i], self.image_size) for i in range(N)])

        # Semantic labels: nearest-neighbour resize as long tensor (N, H, W)
        pre_sem = self._resize_semantic(pre_seg)
        mid_sem = self._resize_semantic(mid_seg)

        return {
            "pre_image": pre_images,
            "mid_image": mid_images,
            "pre_mask": pre_masks,
            "mid_mask": mid_masks,
            "pre_mask_semantic": pre_sem,
            "mid_mask_semantic": mid_sem,
            "weeks_elapsed": self.weeks_elapsed,
            "patient_id": rec["patient_id"],
        }

    def _resize_semantic(self, seg_vol: np.ndarray) -> torch.Tensor:
        """Resize raw (N, H, W) int labels to (N, image_size, image_size) long tensor."""
        t = torch.from_numpy(seg_vol.astype(np.int64)).unsqueeze(1).float()  # (N, 1, H, W)
        t = F.interpolate(t, size=(self.image_size, self.image_size), mode="nearest")
        return t.squeeze(1).long()


# ---------------------------------------------------------------------------
# Slice-level dataset (training)
# ---------------------------------------------------------------------------

class BraTSGLISliceDataset(Dataset):
    """
    Slice-level BraTS-GLI training dataset.

    Mirrors HNTSMRGSliceDataset: random 2D slice pairs from longitudinal
    (TP000, TP001) pairs, with tumor-biased sampling controlled by
    ``tumor_ratio``. Includes flip / noise / brightness augmentation.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        modality: str = "t2f",
        image_size: int = SAM2_IMAGE_SIZE,
        augment: bool = True,
        tumor_ratio: float = 0.7,
        sequence_length: int = 0,  # reserved
        weeks_elapsed: int = BRATS_FOLLOWUP_WEEKS_DEFAULT,
        val_fraction: float = 0.10,
        volume_cache_size: int = 4,
        n_folds: int = 1,
        fold: int = 0,
    ):
        if modality not in VALID_MODALITIES:
            raise ValueError(f"modality must be one of {VALID_MODALITIES}, got {modality!r}")

        self.data_dir = Path(data_dir)
        self.split = split
        self.modality = modality
        self.image_size = image_size
        self.augment = augment and (split == "train")
        self.tumor_ratio = float(tumor_ratio)
        self.sequence_length = int(sequence_length)
        self.weeks_elapsed = int(weeks_elapsed)
        self.volume_cache_size = int(volume_cache_size)
        self._vol_cache: dict[str, dict] = {}

        self.pair_records, _ = _build_split_indices(
            self.data_dir,
            split=split,
            modality=modality,
            include_singletons=False,
            val_fraction=val_fraction,
        )
        if not self.pair_records:
            raise FileNotFoundError(
                f"[BraTSGLISliceDataset] No pairs for split={split!r} in {self.data_dir}"
            )

        # Build tumor / bg slice indices by scanning the mid-timepoint seg only.
        self.tumor_slices: list[tuple[int, int]] = []  # (rec_idx, slice_idx)
        self.bg_slices: list[tuple[int, int]] = []

        print(f"[BraTSGLISliceDataset] Indexing {len(self.pair_records)} {split} pairs...")
        for r_idx, rec in enumerate(self.pair_records):
            try:
                mid_seg = _load_seg_volume(rec["mid_dir"])
            except Exception as e:
                print(f"  Warning: skipping {rec['patient_id']}: {e}")
                continue
            tumor_map = _binary_tumor_mask(mid_seg) > 0  # (D, H, W)
            D = tumor_map.shape[0]
            per_slice = tumor_map.reshape(D, -1).any(axis=1)
            for s in range(D):
                if per_slice[s]:
                    self.tumor_slices.append((r_idx, s))
                else:
                    self.bg_slices.append((r_idx, s))

        print(
            f"  Tumor slices: {len(self.tumor_slices)}, BG slices: {len(self.bg_slices)}"
        )

    # ---- Caching ---------------------------------------------------------

    def _get_patient_cache(self, rec_idx: int) -> dict:
        rec = self.pair_records[rec_idx]
        key = rec["patient_id"]
        if key in self._vol_cache:
            return self._vol_cache[key]

        entry = {
            "pre_vol": _load_modality_volume(rec["pre_dir"], self.modality),
            "mid_vol": _load_modality_volume(rec["mid_dir"], self.modality),
            "pre_seg": _load_seg_volume(rec["pre_dir"]),
            "mid_seg": _load_seg_volume(rec["mid_dir"]),
        }

        # Simple FIFO-ish cap (cheap alternative to an LRU; per-worker anyway).
        if len(self._vol_cache) >= self.volume_cache_size:
            self._vol_cache.pop(next(iter(self._vol_cache)))
        self._vol_cache[key] = entry
        return entry

    # ---- Sampling --------------------------------------------------------

    def __len__(self) -> int:
        n_tumor = len(self.tumor_slices)
        if n_tumor == 0:
            return len(self.bg_slices)
        n_bg = int(n_tumor * (1 - self.tumor_ratio) / max(self.tumor_ratio, 1e-6))
        return n_tumor + min(n_bg, len(self.bg_slices))

    def __getitem__(self, idx: int) -> dict:
        n_tumor = len(self.tumor_slices)
        if n_tumor > 0 and idx < n_tumor:
            rec_idx, slice_idx = self.tumor_slices[idx]
        elif self.bg_slices:
            bg_idx = (idx - n_tumor) % len(self.bg_slices)
            rec_idx, slice_idx = self.bg_slices[bg_idx]
        else:
            rec_idx, slice_idx = self.tumor_slices[idx % max(n_tumor, 1)]
        return self._load_slice(rec_idx, slice_idx)

    def _load_slice(self, rec_idx: int, slice_idx: int) -> dict:
        rec = self.pair_records[rec_idx]
        cache = self._get_patient_cache(rec_idx)

        pre_vol = cache["pre_vol"]
        mid_vol = cache["mid_vol"]
        pre_seg = cache["pre_seg"]
        mid_seg = cache["mid_seg"]

        D_pre = pre_vol.shape[1] if pre_vol.ndim == 4 else pre_vol.shape[0]
        D_mid = mid_vol.shape[1] if mid_vol.ndim == 4 else mid_vol.shape[0]
        N = min(D_pre, D_mid, pre_seg.shape[0], mid_seg.shape[0])
        slice_idx = min(max(0, slice_idx), N - 1)

        # Grab slice arrays for augmentation (needs float grids).
        if self.modality in ("all", "all4"):
            pre_slice = pre_vol[:, slice_idx]  # (3, H, W)
            mid_slice = mid_vol[:, slice_idx]
        else:
            pre_slice = pre_vol[slice_idx]     # (H, W)
            mid_slice = mid_vol[slice_idx]

        pre_bin = _binary_tumor_mask(pre_seg[slice_idx:slice_idx + 1])[0]
        mid_bin = _binary_tumor_mask(mid_seg[slice_idx:slice_idx + 1])[0]
        pre_sem = pre_seg[slice_idx].astype(np.int64)
        mid_sem = mid_seg[slice_idx].astype(np.int64)

        if self.augment:
            pre_slice, mid_slice, pre_bin, mid_bin, pre_sem, mid_sem = self._augment(
                pre_slice, mid_slice, pre_bin, mid_bin, pre_sem, mid_sem
            )

        # Build tensors
        if self.modality in ("all", "all4"):
            pre_t = self._multimodal_slice_to_tensor(pre_slice)
            mid_t = self._multimodal_slice_to_tensor(mid_slice)
        else:
            pre_t = to_rgb_tensor(pre_slice.astype(np.float32), self.image_size)
            mid_t = to_rgb_tensor(mid_slice.astype(np.float32), self.image_size)

        pre_mask_t = to_mask_tensor(pre_bin.astype(np.float32), self.image_size)
        mid_mask_t = to_mask_tensor(mid_bin.astype(np.float32), self.image_size)
        pre_sem_t = self._resize_semantic_slice(pre_sem)
        mid_sem_t = self._resize_semantic_slice(mid_sem)

        return {
            "pre_image": pre_t,
            "mid_image": mid_t,
            "pre_mask": pre_mask_t,
            "mid_mask": mid_mask_t,
            "pre_mask_semantic": pre_sem_t,
            "mid_mask_semantic": mid_sem_t,
            "weeks_elapsed": self.weeks_elapsed,
            "patient_id": rec["patient_id"],
            "slice_idx": int(slice_idx),
        }

    # ---- Helpers ---------------------------------------------------------

    def _multimodal_slice_to_tensor(self, chans: np.ndarray) -> torch.Tensor:
        """(3, H, W) float → SAM2-normalized (3, size, size) tensor."""
        t = torch.from_numpy(chans.astype(np.float32)).unsqueeze(0)  # (1, 3, H, W)
        t = F.interpolate(t, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        t = t.squeeze(0)
        t = (t - SAM2_MEAN[:, None, None]) / SAM2_STD[:, None, None]
        return t

    def _resize_semantic_slice(self, sem_2d: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(sem_2d.astype(np.int64)).unsqueeze(0).unsqueeze(0).float()
        t = F.interpolate(t, size=(self.image_size, self.image_size), mode="nearest")
        return t.squeeze(0).squeeze(0).long()

    def _augment(
        self,
        pre: np.ndarray,
        mid: np.ndarray,
        pre_mask: np.ndarray,
        mid_mask: np.ndarray,
        pre_sem: np.ndarray,
        mid_sem: np.ndarray,
    ):
        """Flip / noise / brightness, consistently on both timepoints."""
        # Multi-modal: flip on the last two axes only.
        def _flip_lr(a: np.ndarray) -> np.ndarray:
            return np.flip(a, axis=-1).copy()

        def _flip_ud(a: np.ndarray) -> np.ndarray:
            return np.flip(a, axis=-2).copy()

        if random.random() < 0.5:
            pre = _flip_lr(pre); mid = _flip_lr(mid)
            pre_mask = _flip_lr(pre_mask); mid_mask = _flip_lr(mid_mask)
            pre_sem = _flip_lr(pre_sem); mid_sem = _flip_lr(mid_sem)

        if random.random() < 0.3:
            pre = _flip_ud(pre); mid = _flip_ud(mid)
            pre_mask = _flip_ud(pre_mask); mid_mask = _flip_ud(mid_mask)
            pre_sem = _flip_ud(pre_sem); mid_sem = _flip_ud(mid_sem)

        if random.random() < 0.5:
            sigma = random.uniform(0, 0.02)
            pre = (pre + np.random.normal(0, sigma, pre.shape)).clip(0, 1)
            mid = (mid + np.random.normal(0, sigma, mid.shape)).clip(0, 1)

        if random.random() < 0.4:
            delta = random.uniform(-0.1, 0.1)
            pre = (pre + delta).clip(0, 1)
            mid = (mid + delta).clip(0, 1)

        return pre, mid, pre_mask, mid_mask, pre_sem, mid_sem


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = SAM2_IMAGE_SIZE,
    modality: str = "t2f",
    pin_memory: bool = True,
    sequence_length: int = 0,
    tumor_ratio: float = 0.7,
    include_singletons: bool = False,
    val_fraction: float = 0.10,
) -> dict[str, DataLoader]:
    """
    Build train (slice-level) + val/test (volume-level) DataLoaders for BraTS-GLI.

    Returns::

        {"train": DataLoader[BraTSGLISliceDataset],
         "val":   DataLoader[BraTSGLIDataset],
         "test":  DataLoader[BraTSGLIDataset]}
    """
    train_ds = BraTSGLISliceDataset(
        data_dir,
        split="train",
        modality=modality,
        image_size=image_size,
        augment=True,
        tumor_ratio=tumor_ratio,
        sequence_length=sequence_length,
        val_fraction=val_fraction,
    )
    val_ds = BraTSGLIDataset(
        data_dir,
        split="val",
        modality=modality,
        image_size=image_size,
        sequence_length=sequence_length,
        include_singletons=False,
        val_fraction=val_fraction,
    )
    test_ds = BraTSGLIDataset(
        data_dir,
        split="test",
        modality=modality,
        image_size=image_size,
        sequence_length=sequence_length,
        include_singletons=include_singletons,
        val_fraction=val_fraction,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BraTS-GLI dataset smoke test")
    parser.add_argument("--data_dir", type=str, default="data/raw/BraTS_GLI")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--modality", type=str, default="t2f", choices=list(VALID_MODALITIES))
    parser.add_argument("--image_size", type=int, default=SAM2_IMAGE_SIZE)
    parser.add_argument("--mode", type=str, default="volume", choices=["volume", "slice"])
    args = parser.parse_args()

    if args.mode == "volume":
        ds = BraTSGLIDataset(
            args.data_dir, split=args.split, modality=args.modality, image_size=args.image_size,
        )
        print(f"len={len(ds)}")
        sample = ds[0]
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {tuple(v.shape)} {v.dtype}")
            else:
                print(f"  {k}: {v}")
    else:
        ds = BraTSGLISliceDataset(
            args.data_dir, split=args.split, modality=args.modality, image_size=args.image_size,
        )
        print(f"len={len(ds)}")
        sample = ds[0]
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {tuple(v.shape)} {v.dtype}")
            else:
                print(f"  {k}: {v}")
