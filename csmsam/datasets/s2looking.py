"""
S2Looking bitemporal change detection dataset for CSM-SAM.

Each parquet row has `t1_image`, `t2_image`, `change_mask` (all
{bytes, path} structs) and `image_name`. Images are 1024x1024 RGB;
change_mask is a binary single-channel mask.

S2Looking has no per-timepoint semantic masks, so `pre_mask` and
`mid_mask` are returned as zeros. `weeks_elapsed` is a placeholder (1).
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# SAM2 expected image statistics
SAM2_MEAN = torch.tensor([0.485, 0.456, 0.406])
SAM2_STD = torch.tensor([0.229, 0.224, 0.225])

# Default image size for SAM2-L (1024×1024)
SAM2_IMAGE_SIZE = 1024


def to_rgb_tensor(img_rgb: np.ndarray, size: int = SAM2_IMAGE_SIZE) -> torch.Tensor:
    """
    Convert an (H, W, 3) uint8/float RGB array to a SAM2-compatible tensor.

    Returns (3, size, size) float32 tensor, normalized with SAM2 mean/std.
    """
    arr = img_rgb.astype(np.float32) / 255.0 if img_rgb.dtype == np.uint8 else img_rgb.astype(np.float32)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    t = t.squeeze(0)  # (3, size, size)
    t = (t - SAM2_MEAN[:, None, None]) / SAM2_STD[:, None, None]
    return t


def to_mask_tensor(mask_2d: np.ndarray, size: int = SAM2_IMAGE_SIZE) -> torch.Tensor:
    """Convert a 2D binary mask to a (1, size, size) float32 tensor."""
    t = torch.from_numpy((mask_2d > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="nearest")
    return t.squeeze(0)


def _decode_image_bytes(b: bytes, mode: str = "RGB") -> np.ndarray:
    """Decode PNG/JPEG bytes via PIL to a numpy array."""
    with Image.open(io.BytesIO(b)) as im:
        im = im.convert(mode)
        return np.array(im)


class S2LookingDataset(Dataset):
    """
    Bitemporal change detection dataset reading sharded parquet files.

    __getitem__ returns dict with keys:
        pre_image, mid_image : (3, H, W) SAM2-normalized
        pre_mask, mid_mask   : (1, H, W) zeros (no semantic labels)
        change_mask          : (1, H, W) binary
        image_name           : str
        weeks_elapsed        : 1 (placeholder)
    """

    # TODO(csm-sam/csmsam/datasets/s2looking.py:80): expose a `crop_size`
    # training option (e.g. 256 or 512) matching ChangeFormer / BIT / SNUNet
    # S2Looking recipes. The native tile is 1024x1024; SOTA methods train on
    # random 256-crops and tile at test-time. Currently we always resize the
    # full 1024 tile to `image_size`, which makes direct F1 comparison with
    # published numbers unfair.
    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = SAM2_IMAGE_SIZE,
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        # S2Looking has no val split — mirror test.
        self.split = "test" if split == "val" else split
        self.augment = augment and (self.split == "train")

        shards = sorted(self.data_dir.glob(f"{self.split}-*.parquet"))
        if not shards:
            raise FileNotFoundError(
                f"No {self.split}-*.parquet shards found in {self.data_dir}"
            )
        self.shards = shards

        # Build global_idx -> (shard_path, row_idx) index once.
        self.index: list[tuple[Path, int]] = []
        for shard in shards:
            n = pq.ParquetFile(str(shard)).metadata.num_rows
            self.index.extend((shard, i) for i in range(n))

        # Lazy per-shard table cache (one open table per worker).
        self._table_cache: dict[str, "pq.Table"] = {}

        print(f"[S2LookingDataset] {self.split}: {len(self.index)} rows across {len(shards)} shards")

    def __len__(self) -> int:
        return len(self.index)

    def _get_table(self, shard: Path):
        key = str(shard)
        if key not in self._table_cache:
            self._table_cache[key] = pq.read_table(key)
        return self._table_cache[key]

    def _extract_bytes(self, cell) -> bytes:
        """Row cell for a struct column -> raw image bytes."""
        if isinstance(cell, dict):
            return cell["bytes"]
        # pyarrow scalar
        return cell["bytes"].as_py()

    def __getitem__(self, idx: int) -> dict:
        shard, row_idx = self.index[idx]
        table = self._get_table(shard)

        t1_cell = table.column("t1_image")[row_idx].as_py()
        t2_cell = table.column("t2_image")[row_idx].as_py()
        cm_cell = table.column("change_mask")[row_idx].as_py()
        image_name = table.column("image_name")[row_idx].as_py()

        pre = _decode_image_bytes(self._extract_bytes(t1_cell), "RGB")      # (H, W, 3)
        mid = _decode_image_bytes(self._extract_bytes(t2_cell), "RGB")
        change = _decode_image_bytes(self._extract_bytes(cm_cell), "L")      # (H, W)

        if self.augment:
            pre, mid, change = self._augment(pre, mid, change)

        H, W = change.shape[:2]
        pre_mask_zero = np.zeros((H, W), dtype=np.float32)
        mid_mask_zero = np.zeros((H, W), dtype=np.float32)

        return {
            "pre_image": to_rgb_tensor(pre, self.image_size),
            "mid_image": to_rgb_tensor(mid, self.image_size),
            "pre_mask": to_mask_tensor(pre_mask_zero, self.image_size),
            "mid_mask": to_mask_tensor(mid_mask_zero, self.image_size),
            "change_mask": to_mask_tensor(change, self.image_size),
            "image_name": image_name,
            "weeks_elapsed": 1,
        }

    def _augment(
        self,
        pre: np.ndarray,
        mid: np.ndarray,
        change: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Consistent hflip / vflip / 90° rotation for the triplet."""
        if random.random() < 0.5:
            pre = np.ascontiguousarray(np.fliplr(pre))
            mid = np.ascontiguousarray(np.fliplr(mid))
            change = np.ascontiguousarray(np.fliplr(change))
        if random.random() < 0.5:
            pre = np.ascontiguousarray(np.flipud(pre))
            mid = np.ascontiguousarray(np.flipud(mid))
            change = np.ascontiguousarray(np.flipud(change))
        k = random.randint(0, 3)
        if k:
            pre = np.ascontiguousarray(np.rot90(pre, k=k, axes=(0, 1)))
            mid = np.ascontiguousarray(np.rot90(mid, k=k, axes=(0, 1)))
            change = np.ascontiguousarray(np.rot90(change, k=k, axes=(0, 1)))
        return pre, mid, change


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = SAM2_IMAGE_SIZE,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """Build train / val / test DataLoaders (val mirrors test)."""
    train_ds = S2LookingDataset(data_dir, split="train", image_size=image_size, augment=True)
    val_ds = S2LookingDataset(data_dir, split="val", image_size=image_size, augment=False)
    test_ds = S2LookingDataset(data_dir, split="test", image_size=image_size, augment=False)

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


if __name__ == "__main__":
    ds = S2LookingDataset(
        "/home/tals/Documents/csm-sam/data/raw/S2Looking/data",
        split="train",
        augment=False,
    )
    print(f"len={len(ds)}")
    s = ds[0]
    for k, v in s.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {v!r}")
