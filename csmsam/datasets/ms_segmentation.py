"""
MS-Segmentation dataset for CSM-SAM.

**Role in CSM-SAM: supplementary pretraining, not a primary evaluation
benchmark.** Small brain-MRI MS lesion dataset (60 examples total across 6
parquet shards; each row is a 3D NIfTI volume, not a PNG). Exposed with a
self-paired interface (pre == mid) so the CSM-SAM trainer can ingest it
uniformly with `single_timepoint=True`. By default we collapse each volume to
the 2D slice with the largest lesion (fallback: middle slice); set
``return_3d=True`` on the dataset to keep the full (D, H, W) volume and emit
a per-slice stacked pre/mid pair instead.

HNTS-MRG 2024 Task 2 is the only primary evaluation target; MS results are not
reported in the main paper tables.
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .hnts_mrg import SAM2_IMAGE_SIZE, SAM2_MEAN, SAM2_STD, to_mask_tensor, to_rgb_tensor  # noqa: F401


def _decode_png(raw: bytes) -> np.ndarray:
    """Decode PNG bytes to a float32 numpy array in [0, 1]."""
    img = Image.open(io.BytesIO(raw))
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def _decode_nifti(raw: bytes) -> np.ndarray:
    """Decode NIfTI bytes to a float32 numpy array (no normalization)."""
    nii = nib.Nifti1Image.from_bytes(raw)
    return nii.get_fdata().astype(np.float32)


def _decode_any(raw: bytes) -> np.ndarray:
    """Decode raw bytes as NIfTI (preferred) or PNG."""
    # NIfTI-1 header starts with sizeof_hdr == 348 (little-endian: 5c 01 00 00)
    if len(raw) >= 4 and raw[:4] == b"\x5c\x01\x00\x00":
        return _decode_nifti(raw)
    return _decode_png(raw)


class MSSegmentationDataset(Dataset):
    """Self-paired MS lesion dataset. pre==mid, zero change map, weeks=0."""

    SEED = 42

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = SAM2_IMAGE_SIZE,
        augment: bool = True,
        return_3d: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == "train")
        # When True, return the full (D, H, W) volume stacked per slice
        # (pre_image shape = (D, 3, H, W)) instead of the single
        # largest-lesion slice. Augmentations are disabled in 3D mode because
        # fliplr/rot90 semantics are ambiguous across the volume axis here.
        self.return_3d = bool(return_3d)
        if self.return_3d and self.augment:
            self.augment = False

        # Locate shard files
        ms_root = self.data_dir
        if (ms_root / "MS-Segmentation").is_dir():
            ms_root = ms_root / "MS-Segmentation"
        if (ms_root / "data").is_dir():
            ms_root = ms_root / "data"

        self.shard_paths: list[Path] = sorted(ms_root.glob("train-*.parquet"))
        if not self.shard_paths:
            raise FileNotFoundError(
                f"No parquet shards found under {ms_root}. Expected "
                "train-00000-of-00006.parquet ... train-00005-of-00006.parquet."
            )

        # Index rows across shards: [(shard_idx, row_idx), ...]
        self._shard_cache: dict[int, pq.Table] = {}
        self.all_index: list[tuple[int, int]] = []
        for si, p in enumerate(self.shard_paths):
            md = pq.read_metadata(str(p))
            n = md.num_rows
            for ri in range(n):
                self.all_index.append((si, ri))

        # 70/10/20 seeded split
        rng = np.random.RandomState(self.SEED)
        perm = rng.permutation(len(self.all_index))
        n_total = len(self.all_index)
        n_train = int(round(n_total * 0.7))
        n_val = int(round(n_total * 0.1))
        train_ids = perm[:n_train]
        val_ids = perm[n_train:n_train + n_val]
        test_ids = perm[n_train + n_val:]
        if split == "train":
            sel = train_ids
        elif split == "val":
            sel = val_ids
        elif split == "test":
            sel = test_ids
        else:
            raise ValueError(f"Unknown split: {split}")
        self.index: list[tuple[int, int]] = [self.all_index[i] for i in sel]

        print(
            f"[MSSegmentationDataset] {split}: {len(self.index)} examples "
            f"(from {n_total} total across {len(self.shard_paths)} shards)"
        )

    def __len__(self) -> int:
        return len(self.index)

    def _get_table(self, shard_idx: int) -> pq.Table:
        t = self._shard_cache.get(shard_idx)
        if t is None:
            t = pq.read_table(str(self.shard_paths[shard_idx]))
            self._shard_cache[shard_idx] = t
        return t

    def _load_row(self, shard_idx: int, row_idx: int) -> tuple[np.ndarray, np.ndarray]:
        table = self._get_table(shard_idx)
        img_col = table.column("image")
        lbl_col = table.column("label")
        img_struct = img_col[row_idx].as_py()
        lbl_struct = lbl_col[row_idx].as_py()
        img = _decode_any(img_struct["bytes"])
        lbl = _decode_any(lbl_struct["bytes"])

        # If volumetric (H, W, D), either return the full volume (return_3d)
        # or pick the slice with the largest lesion (fallback: middle slice).
        if img.ndim == 3:
            lbl_bin = (lbl > 0.5).astype(np.float32)
            if self.return_3d:
                # Keep full volume; move depth axis to front → (D, H, W).
                img = np.moveaxis(img, -1, 0)
                lbl = np.moveaxis(lbl_bin, -1, 0)
            else:
                areas = lbl_bin.reshape(-1, lbl_bin.shape[-1]).sum(axis=0)
                z = int(areas.argmax()) if areas.max() > 0 else lbl_bin.shape[-1] // 2
                img = img[..., z]
                lbl = lbl_bin[..., z]
        else:
            lbl = (lbl > 0.5).astype(np.float32)

        # Per-image normalization to [0, 1] (NIfTI intensities are arbitrary)
        mn, mx = float(img.min()), float(img.max())
        if mx > mn:
            img = (img - mn) / (mx - mn)
        return img.astype(np.float32), lbl.astype(np.float32)

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        if random.random() < 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()
        return img, mask

    def __getitem__(self, idx: int) -> dict:
        shard_idx, row_idx = self.index[idx]
        img, mask = self._load_row(shard_idx, row_idx)

        flat_id = shard_idx * 10 + row_idx

        if self.return_3d:
            # img/mask shape (D, H, W). Stack per-slice into (D, C, H, W).
            D = img.shape[0]
            image_t = torch.stack(
                [to_rgb_tensor(img[z], self.image_size) for z in range(D)]
            )  # (D, 3, H, W)
            mask_t = torch.stack(
                [to_mask_tensor(mask[z], self.image_size) for z in range(D)]
            )  # (D, 1, H, W)
            change_t = torch.zeros_like(mask_t)
            return {
                "pre_image": image_t,
                "mid_image": image_t,
                "pre_mask": mask_t,
                "mid_mask": mask_t,
                "change_mask": change_t,
                "weeks_elapsed": 0,
                "patient_id": f"ms_{flat_id:04d}",
                "single_timepoint": True,
                "is_3d": True,
            }

        if self.augment:
            img, mask = self._augment(img, mask)

        image_t = to_rgb_tensor(img, self.image_size)       # (3, H, W)
        mask_t = to_mask_tensor(mask, self.image_size)      # (1, H, W)
        change_t = torch.zeros_like(mask_t)                 # (1, H, W)

        # Flat index across all shards, stable regardless of split
        return {
            "pre_image": image_t,
            "mid_image": image_t,
            "pre_mask": mask_t,
            "mid_mask": mask_t,
            "change_mask": change_t,
            "weeks_elapsed": 0,
            "patient_id": f"ms_{flat_id:04d}",
            "single_timepoint": True,
        }


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 4,
    num_workers: int = 2,
    image_size: int = SAM2_IMAGE_SIZE,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders for MS-Segmentation."""
    train_ds = MSSegmentationDataset(data_dir, split="train", image_size=image_size, augment=True)
    val_ds = MSSegmentationDataset(data_dir, split="val", image_size=image_size, augment=False)
    test_ds = MSSegmentationDataset(data_dir, split="test", image_size=image_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/MS-Segmentation"
    ds = MSSegmentationDataset(data_dir, split="train", image_size=256, augment=True)
    print(f"Train size: {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {v!r}")

    loaders = build_dataloaders(data_dir, batch_size=2, num_workers=0, image_size=256)
    for split, dl in loaders.items():
        print(f"{split}: {len(dl.dataset)} examples, {len(dl)} batches")
    batch = next(iter(loaders["train"]))
    print("Train batch:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {v}")
