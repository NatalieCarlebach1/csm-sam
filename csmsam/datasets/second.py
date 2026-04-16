"""
SECOND Semantic Change Detection Dataset for CSM-SAM.

SECOND provides paired bi-temporal remote-sensing images with full semantic
masks at each time point PLUS a change mask. It is the best non-medical
analog for CSM-SAM, supplying the exact supervision triple the model needs:

    (pre_mask, mid_mask, change_mask)

On-disk layout (parquet shards under data/raw/SECOND/data/):
    train-0000X-of-00005.parquet    (594 rows/shard)
    test-0000X-of-00003.parquet

Schema (per row):
    t1_image     : {bytes, path}  — RGB PNG, 512x512
    t2_image     : {bytes, path}  — RGB PNG, 512x512
    t1_mask      : {bytes, path}  — 1-channel PNG, values 0..6 (semantic)
    t2_mask      : {bytes, path}  — 1-channel PNG, values 0..6 (semantic)
    change_mask  : {bytes, path}  — 1-channel PNG, 0/1
    image_name   : str

Classes (approximate land-cover, 7 total):
    0 : background / no-data / no-change
    1 : non-vegetated ground surface
    2 : tree
    3 : low vegetation
    4 : water
    5 : buildings
    6 : playground
"""

from __future__ import annotations

import io
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from csmsam.datasets.hnts_mrg import (
    SAM2_IMAGE_SIZE,
    SAM2_MEAN,
    SAM2_STD,
    to_mask_tensor,
    to_rgb_tensor,
)


def _to_rgb_tensor_from_rgb(img_rgb: np.ndarray, size: int = SAM2_IMAGE_SIZE) -> torch.Tensor:
    """
    Convert an (H, W, 3) uint8 RGB array to a SAM2-compatible normalized tensor.

    The hnts_mrg `to_rgb_tensor` replicates a grayscale slice to 3 channels;
    here we keep the native RGB channels from SECOND.
    """
    import torch.nn.functional as F

    t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)  # (H, W, 3)
    t = t.permute(2, 0, 1).unsqueeze(0)                       # (1, 3, H, W)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    t = t.squeeze(0)                                          # (3, size, size)
    t = (t - SAM2_MEAN[:, None, None]) / SAM2_STD[:, None, None]
    return t


def _semantic_mask_tensor(mask_2d: np.ndarray, size: int = SAM2_IMAGE_SIZE) -> torch.Tensor:
    """
    Resize a (H, W) integer semantic mask via nearest neighbor to (size, size) long.
    """
    import torch.nn.functional as F

    t = torch.from_numpy(mask_2d.astype(np.int64)).unsqueeze(0).unsqueeze(0).float()
    t = F.interpolate(t, size=(size, size), mode="nearest")
    return t.squeeze(0).squeeze(0).long()  # (size, size)


class _ShardCache:
    """Tiny LRU cache of opened ParquetFile handles (one per shard)."""

    def __init__(self, maxsize: int = 4):
        self.maxsize = maxsize
        self._cache: "OrderedDict[str, pq.ParquetFile]" = OrderedDict()

    def get(self, path: Path) -> pq.ParquetFile:
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        pf = pq.ParquetFile(str(path))
        self._cache[key] = pf
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return pf


def _decode_bytes_field(field) -> bytes:
    """Extract raw bytes from a {bytes, path} struct cell (pyarrow scalar)."""
    if isinstance(field, dict):
        return field.get("bytes") or Path(field["path"]).read_bytes()
    # pyarrow scalar-like
    as_py = field.as_py() if hasattr(field, "as_py") else field
    if isinstance(as_py, dict):
        return as_py.get("bytes") or Path(as_py["path"]).read_bytes()
    if isinstance(as_py, (bytes, bytearray)):
        return bytes(as_py)
    raise ValueError(f"Unsupported field type: {type(as_py)}")


def _decode_image(raw: bytes, mode: str) -> np.ndarray:
    img = Image.open(io.BytesIO(raw)).convert(mode)
    return np.array(img)


class SECONDDataset(Dataset):
    """
    Parquet-backed SECOND dataset.

    __getitem__ returns:
        pre_image          : (3, H, W)  SAM2-normalized RGB
        mid_image          : (3, H, W)
        pre_mask           : (1, H, W)  binary foreground (any class > 0)
        mid_mask           : (1, H, W)  binary foreground
        pre_mask_semantic  : (H, W)     long tensor of class ids (0..6)
        mid_mask_semantic  : (H, W)     long
        change_mask        : (1, H, W)  binary change indicator
        image_name         : str
        weeks_elapsed      : int        default 1
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = SAM2_IMAGE_SIZE,
        augment: bool = True,
        weeks_elapsed: int = 1,
        shard_cache_size: int = 4,
    ):
        self.data_dir = Path(data_dir)
        # 'val' aliases to 'test' — SECOND has no separate validation shard set.
        split_eff = "test" if split == "val" else split
        if split_eff not in ("train", "test"):
            raise ValueError(f"Unknown split: {split!r}")
        self.split = split_eff
        self.image_size = image_size
        self.augment = augment and (split_eff == "train")
        self.weeks_elapsed = int(weeks_elapsed)

        prefix = f"{split_eff}-"
        shards = sorted(self.data_dir.glob(f"{prefix}*.parquet"))
        if not shards and (self.data_dir / "data").is_dir():
            # HF-style layout: parquet shards live under <data_dir>/data/
            self.data_dir = self.data_dir / "data"
            shards = sorted(self.data_dir.glob(f"{prefix}*.parquet"))
        if not shards:
            raise FileNotFoundError(
                f"No {split_eff} shards in {self.data_dir} (looked for {prefix}*.parquet)."
            )

        # Build (shard_path, row_idx) index by reading each shard's row count.
        self.index: list[tuple[Path, int]] = []
        self.shard_cache = _ShardCache(maxsize=shard_cache_size)
        for shard in shards:
            pf = self.shard_cache.get(shard)
            n = pf.metadata.num_rows
            self.index.extend((shard, i) for i in range(n))

        print(f"[SECONDDataset] {split_eff}: {len(self.index)} rows across {len(shards)} shards")

    def __len__(self) -> int:
        return len(self.index)

    # ------------------------------------------------------------------
    def _read_row(self, shard: Path, row_idx: int) -> dict:
        """Read a single row as a dict of bytes / str fields.

        Uses row-group-level reads so we only materialize the target row group.
        """
        pf = self.shard_cache.get(shard)
        # Locate row group containing row_idx
        cum = 0
        for rg_idx in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg_idx).num_rows
            if row_idx < cum + n:
                local = row_idx - cum
                table = pf.read_row_group(rg_idx)
                break
            cum += n
        else:
            raise IndexError(row_idx)

        row = {name: table.column(name)[local].as_py() for name in table.column_names}
        return row

    # ------------------------------------------------------------------
    def _augment(
        self,
        pre: np.ndarray,
        mid: np.ndarray,
        pre_sem: np.ndarray,
        mid_sem: np.ndarray,
        change: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Consistent hflip / vflip / rot90 applied to all tensors."""
        if random.random() < 0.5:  # hflip
            pre = np.ascontiguousarray(pre[:, ::-1])
            mid = np.ascontiguousarray(mid[:, ::-1])
            pre_sem = np.ascontiguousarray(pre_sem[:, ::-1])
            mid_sem = np.ascontiguousarray(mid_sem[:, ::-1])
            change = np.ascontiguousarray(change[:, ::-1])
        if random.random() < 0.5:  # vflip
            pre = np.ascontiguousarray(pre[::-1, :])
            mid = np.ascontiguousarray(mid[::-1, :])
            pre_sem = np.ascontiguousarray(pre_sem[::-1, :])
            mid_sem = np.ascontiguousarray(mid_sem[::-1, :])
            change = np.ascontiguousarray(change[::-1, :])
        k = random.randint(0, 3)
        if k:
            pre = np.ascontiguousarray(np.rot90(pre, k=k, axes=(0, 1)))
            mid = np.ascontiguousarray(np.rot90(mid, k=k, axes=(0, 1)))
            pre_sem = np.ascontiguousarray(np.rot90(pre_sem, k=k, axes=(0, 1)))
            mid_sem = np.ascontiguousarray(np.rot90(mid_sem, k=k, axes=(0, 1)))
            change = np.ascontiguousarray(np.rot90(change, k=k, axes=(0, 1)))
        return pre, mid, pre_sem, mid_sem, change

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        shard, row_idx = self.index[idx]
        row = self._read_row(shard, row_idx)

        pre_rgb = _decode_image(_decode_bytes_field(row["t1_image"]), mode="RGB")      # (H, W, 3)
        mid_rgb = _decode_image(_decode_bytes_field(row["t2_image"]), mode="RGB")
        pre_sem = _decode_image(_decode_bytes_field(row["t1_mask"]), mode="L")         # (H, W)
        mid_sem = _decode_image(_decode_bytes_field(row["t2_mask"]), mode="L")
        change = _decode_image(_decode_bytes_field(row["change_mask"]), mode="L")
        image_name = str(row.get("image_name", f"{shard.stem}_{row_idx}"))

        # Binarize change map defensively (some datasets store as 0/255).
        change = (change > 0).astype(np.uint8)

        if self.augment:
            pre_rgb, mid_rgb, pre_sem, mid_sem, change = self._augment(
                pre_rgb, mid_rgb, pre_sem, mid_sem, change
            )

        pre_image = _to_rgb_tensor_from_rgb(pre_rgb, self.image_size)
        mid_image = _to_rgb_tensor_from_rgb(mid_rgb, self.image_size)

        # Binary foreground masks: any semantic class > 0 is foreground.
        pre_mask_bin = (pre_sem > 0).astype(np.float32)
        mid_mask_bin = (mid_sem > 0).astype(np.float32)

        return {
            "pre_image": pre_image,
            "mid_image": mid_image,
            "pre_mask": to_mask_tensor(pre_mask_bin, self.image_size),       # (1, H, W) float
            "mid_mask": to_mask_tensor(mid_mask_bin, self.image_size),       # (1, H, W) float
            "pre_mask_semantic": _semantic_mask_tensor(pre_sem, self.image_size),  # (H, W) long
            "mid_mask_semantic": _semantic_mask_tensor(mid_sem, self.image_size),  # (H, W) long
            "change_mask": to_mask_tensor(change.astype(np.float32), self.image_size),  # (1, H, W)
            "image_name": image_name,
            "weeks_elapsed": self.weeks_elapsed,
        }


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = SAM2_IMAGE_SIZE,
    pin_memory: bool = True,
    weeks_elapsed: int = 1,
) -> dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders for SECOND. 'val' aliases to 'test'.
    """
    train_ds = SECONDDataset(
        data_dir, split="train", image_size=image_size, augment=True, weeks_elapsed=weeks_elapsed
    )
    val_ds = SECONDDataset(
        data_dir, split="val", image_size=image_size, augment=False, weeks_elapsed=weeks_elapsed
    )
    test_ds = SECONDDataset(
        data_dir, split="test", image_size=image_size, augment=False, weeks_elapsed=weeks_elapsed
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
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    data_dir = Path(
        sys.argv[1] if len(sys.argv) > 1 else "data/raw/SECOND/data"
    )

    ds = SECONDDataset(data_dir, split="train", image_size=512, augment=True)
    print(f"len(train) = {len(ds)}")

    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:<20} tensor {tuple(v.shape)} dtype={v.dtype} "
                  f"min={v.min().item():.3f} max={v.max().item():.3f}")
        else:
            print(f"  {k:<20} {type(v).__name__} = {v}")

    loaders = build_dataloaders(data_dir, batch_size=2, num_workers=0, image_size=512)
    batch = next(iter(loaders["train"]))
    print("batch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:<20} {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k:<20} {v}")
