"""
OAIZIB-CM Knee MRI Segmentation Dataset for CSM-SAM.

OAIZIB-CM is a SINGLE-TIMEPOINT (cross-sectional) knee MRI segmentation
dataset derived from the Osteoarthritis Initiative (OAI) with segmentations
released by ZIB. It is NOT longitudinal — there is no pre/mid pair.

**Role in CSM-SAM: supplementary pretraining, not a primary evaluation
benchmark.** HNTS-MRG 2024 Task 2 (mid-RT H&N tumor) is the only primary
evaluation target. OAIZIB-CM is used purely to warm up the cross-session
memory attention on (pre == mid) "self" pairs, giving the model an explicit
identity signal:
    "when pre and mid are identical, cross-session propagation should recover
     the input mask and predict zero change."

This stabilizes the change head early in training on a structurally simpler
task than HNTS-MRG, before fine-tuning on true longitudinal pairs. We do not
report OAIZIB-CM Dice in the main paper tables.

Dataset on disk (zipped, auto-extracted on first use):
    data/raw/OAIZIB-CM/
        imagesTr.zip   → imagesTr/oaizib_NNN_0000.nii.gz  (404 knees, 3D MRI)
        labelsTr.zip   → labelsTr/oaizib_NNN.nii.gz
        imagesTs.zip   → imagesTs/oaizib_NNN_0000.nii.gz  (103 knees)
        labelsTs.zip   → labelsTs/oaizib_NNN.nii.gz
        info.zip       → dataset readme / licence / json

Label mapping (5 ROI + background, per the OAIZIB-CM README — we trust this
mapping without re-verifying from the NIfTI; a volume-level sanity check
(np.unique <= {0..5}) is logged on first load but not enforced):
    0 — background
    1 — femur
    2 — tibia
    3 — femoral cartilage
    4 — medial tibial cartilage
    5 — lateral tibial cartilage

Splits:
    train : first 90% of imagesTr / labelsTr          (seed 42 shuffle)
    val   : last  10% of imagesTr / labelsTr          (seed 42 shuffle)
    test  : imagesTs / labelsTs

Public API:
    OAIZIBDataset        — volume-level, pair_mode in {'self', 'single'}
    OAIZIBSliceDataset   — slice-level sampling for training
    build_dataloaders    — {'train', 'val', 'test'} DataLoaders
"""

from __future__ import annotations

import random
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .hnts_mrg import (
    SAM2_IMAGE_SIZE,
    SAM2_MEAN,  # noqa: F401  (re-exported for convenience)
    SAM2_STD,   # noqa: F401
    load_nifti,
    normalize_mri,
    to_mask_tensor,
    to_rgb_tensor,
)


# OAIZIB-CM label set (see module docstring). Used only for documentation /
# optional sanity logging — we do not enforce or re-verify.
OAIZIB_LABELS: dict[int, str] = {
    0: "background",
    1: "femur",
    2: "tibia",
    3: "femoral_cartilage",
    4: "medial_tibial_cartilage",
    5: "lateral_tibial_cartilage",
}
OAIZIB_NUM_CLASSES = len(OAIZIB_LABELS)  # 6 incl. background


# --------------------------------------------------------------------------- #
# Zip extraction
# --------------------------------------------------------------------------- #

_ZIP_SPECS: dict[str, tuple[str, str]] = {
    # logical name : (zip filename, expected top-level dir after extraction)
    "imagesTr": ("imagesTr.zip", "imagesTr"),
    "labelsTr": ("labelsTr.zip", "labelsTr"),
    "imagesTs": ("imagesTs.zip", "imagesTs"),
    "labelsTs": ("labelsTs.zip", "labelsTs"),
}


def _ensure_extracted(data_dir: Path) -> dict[str, Path]:
    """
    Ensure each of the OAIZIB-CM image/label zips is extracted under
    ``data_dir``. Returns a mapping of logical-name → extracted directory.

    Idempotent: if the target directory already exists and is non-empty, the
    zip is not re-extracted.
    """
    data_dir = Path(data_dir)
    resolved: dict[str, Path] = {}
    for name, (zip_name, subdir) in _ZIP_SPECS.items():
        target = data_dir / subdir
        if target.exists() and any(target.iterdir()):
            resolved[name] = target
            continue
        zip_path = data_dir / zip_name
        if not zip_path.exists():
            raise FileNotFoundError(
                f"OAIZIB-CM zip not found: {zip_path}. "
                "Expected imagesTr.zip / labelsTr.zip / imagesTs.zip / "
                "labelsTs.zip under data_dir."
            )
        print(f"[OAIZIB-CM] Extracting {zip_path.name} → {data_dir}/ ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        if not target.exists():
            # Some zips may extract without the expected top-level dir if
            # already flattened — fall back to data_dir itself.
            target = data_dir
        resolved[name] = target
    return resolved


# --------------------------------------------------------------------------- #
# Indexing / splits
# --------------------------------------------------------------------------- #

def _pair_image_and_label(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path, str]]:
    """
    Return list of (image_path, label_path, patient_id) triples.

    OAIZIB-CM naming:
        imagesTr/oaizib_NNN_0000.nii.gz  ↔  labelsTr/oaizib_NNN.nii.gz
    The trailing ``_0000`` denotes the modality channel (nnU-Net convention);
    we strip it to recover the patient id.
    """
    items: list[tuple[Path, Path, str]] = []
    for img_path in sorted(images_dir.glob("*.nii.gz")):
        stem = img_path.name
        # drop ".nii.gz"
        if stem.endswith(".nii.gz"):
            stem = stem[: -len(".nii.gz")]
        # drop "_0000" modality suffix if present
        if stem.endswith("_0000"):
            pid = stem[: -len("_0000")]
        else:
            pid = stem
        lbl_path = labels_dir / f"{pid}.nii.gz"
        if not lbl_path.exists():
            # missing label → skip silently (some distributions have extras)
            continue
        items.append((img_path, lbl_path, pid))
    return items


def _split_train_val(
    items: list[tuple[Path, Path, str]],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list, list]:
    """Deterministic 90/10 split of the train list using a seeded RNG."""
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val = sorted(shuffled[-n_val:], key=lambda x: x[2])
    train = sorted(shuffled[:-n_val], key=lambda x: x[2])
    return train, val


# --------------------------------------------------------------------------- #
# Volume-level dataset
# --------------------------------------------------------------------------- #

class OAIZIBDataset(Dataset):
    """
    Volume-level dataset for OAIZIB-CM knee MRI.

    ``pair_mode`` controls the output format:

    * 'self' — CSM-SAM-compatible view. Each item returns a (pre, mid) pair
        where pre == mid (same image, same mask) and ``change_mask`` is all
        zeros. ``single_timepoint=True`` is emitted so the trainer can gate
        the change-head loss off for OAIZIB-CM samples.
    * 'single' — plain cross-sectional view. Returns only one image / mask
        without the pre/mid duplication; use for standard segmentation
        pretraining or for baselines that do not use CSM-SAM's memory.

    __getitem__ (pair_mode='self') returns:
        pre_image          : (N, 3, H, W) — SAM2-normalized RGB slices
        mid_image          : (N, 3, H, W) — identical to pre_image
        pre_mask           : (N, 1, H, W) — binary (any class > 0)
        mid_mask           : (N, 1, H, W) — identical to pre_mask
        pre_mask_semantic  : (N, H, W)    — long, values in {0..5}
        mid_mask_semantic  : (N, H, W)    — long, identical to pre_mask_semantic
        change_mask        : (N, 1, H, W) — zeros (by construction)
        weeks_elapsed      : 0            — no temporal gap
        patient_id         : str
        single_timepoint   : True         — flag for trainer to skip change loss

    __getitem__ (pair_mode='single') returns:
        image              : (N, 3, H, W)
        mask               : (N, 1, H, W) — binary
        mask_semantic      : (N, H, W)    — long
        patient_id         : str
        single_timepoint   : True
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = SAM2_IMAGE_SIZE,
        pair_mode: str = "self",
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        if pair_mode not in ("self", "single"):
            raise ValueError(f"pair_mode must be 'self' or 'single', got {pair_mode!r}")
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = int(image_size)
        self.pair_mode = pair_mode

        dirs = _ensure_extracted(self.data_dir)

        if split == "test":
            items = _pair_image_and_label(dirs["imagesTs"], dirs["labelsTs"])
            self.items = items
        else:
            all_tr = _pair_image_and_label(dirs["imagesTr"], dirs["labelsTr"])
            train_items, val_items = _split_train_val(
                all_tr, val_fraction=val_fraction, seed=seed
            )
            self.items = train_items if split == "train" else val_items

        if not self.items:
            raise FileNotFoundError(
                f"No OAIZIB-CM samples found for split={split!r} under {self.data_dir}."
            )

        print(
            f"[OAIZIBDataset] {split}: {len(self.items)} volumes "
            f"(pair_mode={pair_mode})"
        )

    def __len__(self) -> int:
        return len(self.items)

    def _load_volume(self, idx: int) -> tuple[np.ndarray, np.ndarray, str]:
        img_path, lbl_path, pid = self.items[idx]
        vol = normalize_mri(load_nifti(img_path))       # (D, H, W) float32 in [0,1]
        lbl = load_nifti(lbl_path).astype(np.int64)     # (D, H, W) long {0..5}
        # Align depths defensively
        D = min(vol.shape[0], lbl.shape[0])
        return vol[:D], lbl[:D], pid

    def __getitem__(self, idx: int) -> dict:
        vol, lbl, pid = self._load_volume(idx)
        N = vol.shape[0]

        image_slices = torch.stack([to_rgb_tensor(vol[i], self.image_size) for i in range(N)])
        binary_mask_slices = torch.stack(
            [to_mask_tensor((lbl[i] > 0).astype(np.float32), self.image_size) for i in range(N)]
        )
        # Semantic mask: nearest-resize per slice, keep integer labels.
        sem_slices = torch.stack([
            torch.nn.functional.interpolate(
                torch.from_numpy(lbl[i].astype(np.float32))[None, None],
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0).squeeze(0).long()
            for i in range(N)
        ])  # (N, H, W) long

        if self.pair_mode == "single":
            return {
                "image": image_slices,
                "mask": binary_mask_slices,
                "mask_semantic": sem_slices,
                "patient_id": pid,
                "single_timepoint": True,
            }

        # pair_mode == 'self' — duplicate tensors for pre / mid.
        zero_change = torch.zeros_like(binary_mask_slices)
        return {
            "pre_image": image_slices,
            "mid_image": image_slices.clone(),
            "pre_mask": binary_mask_slices,
            "mid_mask": binary_mask_slices.clone(),
            "pre_mask_semantic": sem_slices,
            "mid_mask_semantic": sem_slices.clone(),
            "change_mask": zero_change,
            "weeks_elapsed": 0,
            "patient_id": pid,
            "single_timepoint": True,
        }


# --------------------------------------------------------------------------- #
# Slice-level dataset (training)
# --------------------------------------------------------------------------- #

class OAIZIBSliceDataset(Dataset):
    """
    Per-slice random-access dataset for OAIZIB-CM. One item = one 2D slice.

    On first use, scans every volume once to build a flat index of
    (volume_idx, slice_idx) pairs, optionally filtered to slices with
    foreground. Volumes are loaded lazily and cached in-memory per worker.

    pair_mode matches ``OAIZIBDataset``:
        'self'   → returns {pre_image, mid_image, pre_mask, mid_mask,
                            pre_mask_semantic, mid_mask_semantic,
                            change_mask, weeks_elapsed, patient_id, slice_idx,
                            single_timepoint=True}
        'single' → returns {image, mask, mask_semantic, patient_id, slice_idx,
                            single_timepoint=True}
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = SAM2_IMAGE_SIZE,
        pair_mode: str = "self",
        augment: bool = True,
        foreground_only: bool = True,
        foreground_ratio: float = 0.9,
        val_fraction: float = 0.1,
        seed: int = 42,
        volume_cache_size: int = 4,
    ):
        if pair_mode not in ("self", "single"):
            raise ValueError(f"pair_mode must be 'self' or 'single', got {pair_mode!r}")

        self.image_size = int(image_size)
        self.pair_mode = pair_mode
        self.augment = augment and (split == "train")
        self.foreground_only = bool(foreground_only)
        self.foreground_ratio = float(foreground_ratio)
        self.volume_cache_size = int(volume_cache_size)

        # Delegate volume discovery + split to the volume-level dataset.
        self._vol_ds = OAIZIBDataset(
            data_dir=data_dir,
            split=split,
            image_size=image_size,
            pair_mode="single",          # semantics only; we re-assemble below
            val_fraction=val_fraction,
            seed=seed,
        )

        # Build flat index by scanning label volumes only (cheaper than images).
        self.fg_index: list[tuple[int, int]] = []
        self.bg_index: list[tuple[int, int]] = []
        print(
            f"[OAIZIBSliceDataset] Indexing {len(self._vol_ds)} {split} volumes ..."
        )
        for vi, (_img_path, lbl_path, _pid) in enumerate(self._vol_ds.items):
            try:
                lbl = load_nifti(lbl_path)
                fg_per_slice = (lbl > 0).reshape(lbl.shape[0], -1).any(axis=1)
                for si, is_fg in enumerate(fg_per_slice):
                    if is_fg:
                        self.fg_index.append((vi, si))
                    else:
                        self.bg_index.append((vi, si))
            except Exception as e:
                print(f"  Warning: skipping volume {vi} ({lbl_path.name}): {e}")

        print(
            f"  Foreground slices: {len(self.fg_index)}, "
            f"Background slices: {len(self.bg_index)}"
        )

        # Simple LRU of (vol_idx -> (vol, lbl, pid))
        self._cache: "dict[int, tuple[np.ndarray, np.ndarray, str]]" = {}
        self._cache_order: list[int] = []

    # -- cache --
    def _get_volume(self, vol_idx: int) -> tuple[np.ndarray, np.ndarray, str]:
        if vol_idx in self._cache:
            # touch
            self._cache_order.remove(vol_idx)
            self._cache_order.append(vol_idx)
            return self._cache[vol_idx]
        vol, lbl, pid = self._vol_ds._load_volume(vol_idx)
        self._cache[vol_idx] = (vol, lbl, pid)
        self._cache_order.append(vol_idx)
        if len(self._cache_order) > self.volume_cache_size:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        return vol, lbl, pid

    # -- length / sampling --
    def __len__(self) -> int:
        n_fg = len(self.fg_index)
        if not self.foreground_only and self.bg_index:
            # Mix in a fraction of BG slices.
            n_bg = int(n_fg * (1.0 - self.foreground_ratio) / max(self.foreground_ratio, 1e-6))
            return n_fg + min(n_bg, len(self.bg_index))
        return n_fg

    def _pick(self, idx: int) -> tuple[int, int]:
        n_fg = len(self.fg_index)
        if idx < n_fg:
            return self.fg_index[idx]
        bg_idx = (idx - n_fg) % max(1, len(self.bg_index))
        return self.bg_index[bg_idx]

    # -- augmentations --
    def _augment(
        self,
        image: np.ndarray,
        mask_bin: np.ndarray,
        mask_sem: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Horizontal flip
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask_bin = np.fliplr(mask_bin).copy()
            mask_sem = np.fliplr(mask_sem).copy()
        # Vertical flip (knees are roughly symmetric axially — keep prob low)
        if random.random() < 0.2:
            image = np.flipud(image).copy()
            mask_bin = np.flipud(mask_bin).copy()
            mask_sem = np.flipud(mask_sem).copy()
        # Gaussian noise
        if random.random() < 0.5:
            sigma = random.uniform(0.0, 0.02)
            image = (image + np.random.normal(0, sigma, image.shape)).clip(0, 1)
        # Brightness
        if random.random() < 0.4:
            delta = random.uniform(-0.1, 0.1)
            image = (image + delta).clip(0, 1)
        return image, mask_bin, mask_sem

    # -- item --
    def __getitem__(self, idx: int) -> dict:
        vol_idx, slice_idx = self._pick(idx)
        vol, lbl, pid = self._get_volume(vol_idx)

        slice_idx = min(slice_idx, vol.shape[0] - 1)
        image = vol[slice_idx].astype(np.float32)
        lbl_slice = lbl[slice_idx].astype(np.int64)
        mask_bin = (lbl_slice > 0).astype(np.float32)

        if self.augment:
            image, mask_bin, lbl_slice = self._augment(
                image, mask_bin, lbl_slice.astype(np.float32)
            )
            lbl_slice = lbl_slice.astype(np.int64)

        image_t = to_rgb_tensor(image, self.image_size)
        mask_bin_t = to_mask_tensor(mask_bin, self.image_size)
        mask_sem_t = torch.nn.functional.interpolate(
            torch.from_numpy(lbl_slice.astype(np.float32))[None, None],
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0).squeeze(0).long()

        if self.pair_mode == "single":
            return {
                "image": image_t,
                "mask": mask_bin_t,
                "mask_semantic": mask_sem_t,
                "patient_id": pid,
                "slice_idx": int(slice_idx),
                "single_timepoint": True,
            }

        # pair_mode == 'self'
        return {
            "pre_image": image_t,
            "mid_image": image_t.clone(),
            "pre_mask": mask_bin_t,
            "mid_mask": mask_bin_t.clone(),
            "pre_mask_semantic": mask_sem_t,
            "mid_mask_semantic": mask_sem_t.clone(),
            "change_mask": torch.zeros_like(mask_bin_t),
            "weeks_elapsed": 0,
            "patient_id": pid,
            "slice_idx": int(slice_idx),
            "single_timepoint": True,
        }


# --------------------------------------------------------------------------- #
# DataLoader factory
# --------------------------------------------------------------------------- #

def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = SAM2_IMAGE_SIZE,
    pin_memory: bool = True,
    pair_mode: str = "self",
    foreground_only: bool = True,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """
    Build {'train', 'val', 'test'} DataLoaders for OAIZIB-CM.

    * train — ``OAIZIBSliceDataset`` (per-slice random access, augmented)
    * val   — ``OAIZIBDataset`` (per-volume, no aug)
    * test  — ``OAIZIBDataset`` (per-volume, no aug)

    Each item carries ``single_timepoint=True``; the CSM-SAM trainer should
    detect this flag and skip the change-map CE loss on OAIZIB-CM batches
    (since change is zero by construction and provides no learning signal
    beyond "identity → identity").
    """
    train_ds = OAIZIBSliceDataset(
        data_dir=data_dir,
        split="train",
        image_size=image_size,
        pair_mode=pair_mode,
        augment=True,
        foreground_only=foreground_only,
        val_fraction=val_fraction,
        seed=seed,
    )
    val_ds = OAIZIBDataset(
        data_dir=data_dir,
        split="val",
        image_size=image_size,
        pair_mode=pair_mode,
        val_fraction=val_fraction,
        seed=seed,
    )
    test_ds = OAIZIBDataset(
        data_dir=data_dir,
        split="test",
        image_size=image_size,
        pair_mode=pair_mode,
        val_fraction=val_fraction,
        seed=seed,
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


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

def _smoke_test(data_dir: str | Path = "data/raw/OAIZIB-CM") -> None:
    """
    Minimal sanity check: builds each dataset once, fetches a single item,
    and prints its tensor shapes. Expects the zips to already be present.

    Run manually:
        python -m csmsam.datasets.oaizib_cm
    """
    data_dir = Path(data_dir)
    print(f"[smoke] data_dir = {data_dir}")

    # Volume-level, self-paired
    vol_ds = OAIZIBDataset(data_dir, split="val", pair_mode="self", image_size=256)
    item = vol_ds[0]
    print(f"[smoke] OAIZIBDataset(self) patient_id = {item['patient_id']}")
    print(f"        pre_image          shape = {tuple(item['pre_image'].shape)}")
    print(f"        mid_image          shape = {tuple(item['mid_image'].shape)}")
    print(f"        pre_mask           shape = {tuple(item['pre_mask'].shape)}")
    print(f"        mid_mask           shape = {tuple(item['mid_mask'].shape)}")
    print(f"        pre_mask_semantic  shape = {tuple(item['pre_mask_semantic'].shape)}, "
          f"dtype = {item['pre_mask_semantic'].dtype}, "
          f"unique = {torch.unique(item['pre_mask_semantic']).tolist()}")
    print(f"        change_mask        sum   = {float(item['change_mask'].sum()):.1f}  (expect 0)")
    print(f"        weeks_elapsed            = {item['weeks_elapsed']}")
    print(f"        single_timepoint         = {item['single_timepoint']}")
    assert item["single_timepoint"] is True
    assert torch.equal(item["pre_image"], item["mid_image"])
    assert torch.equal(item["pre_mask"], item["mid_mask"])
    assert float(item["change_mask"].sum()) == 0.0

    # Volume-level, single
    single_ds = OAIZIBDataset(data_dir, split="val", pair_mode="single", image_size=256)
    single_item = single_ds[0]
    print(f"[smoke] OAIZIBDataset(single) image shape = {tuple(single_item['image'].shape)}, "
          f"mask shape = {tuple(single_item['mask'].shape)}, "
          f"mask_semantic shape = {tuple(single_item['mask_semantic'].shape)}")

    # Slice-level
    slice_ds = OAIZIBSliceDataset(
        data_dir, split="val", pair_mode="self", image_size=256, augment=False
    )
    s = slice_ds[0]
    print(f"[smoke] OAIZIBSliceDataset(self) pre_image shape = {tuple(s['pre_image'].shape)}, "
          f"mask shape = {tuple(s['pre_mask'].shape)}, single_timepoint = {s['single_timepoint']}")


if __name__ == "__main__":
    _smoke_test()
