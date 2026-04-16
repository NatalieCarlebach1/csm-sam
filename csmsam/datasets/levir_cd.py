"""
LEVIR-CD bitemporal change detection dataset for CSM-SAM.

Disk layout (after extraction):
    data/raw/LEVIR-CD/
        train/A/train_1.png, B/train_1.png, label/train_1.png, ...
        val/...
        test/...
"""

from __future__ import annotations

import random
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# SAM2 expected image statistics
SAM2_MEAN = torch.tensor([0.485, 0.456, 0.406])
SAM2_STD = torch.tensor([0.229, 0.224, 0.225])

# Default image size for SAM2-L (1024×1024)
SAM2_IMAGE_SIZE = 1024


def to_rgb_tensor(image_rgb: np.ndarray, size: int = SAM2_IMAGE_SIZE) -> torch.Tensor:
    """
    Convert an RGB image (H, W, 3) uint8 or float to a SAM2-compatible tensor.

    Returns (3, size, size) float32, normalized with SAM2 mean/std.
    """
    arr = image_rgb.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    t = t.squeeze(0)
    t = (t - SAM2_MEAN[:, None, None]) / SAM2_STD[:, None, None]
    return t


def to_mask_tensor(mask_2d: np.ndarray, size: int = SAM2_IMAGE_SIZE) -> torch.Tensor:
    """Binary (H, W) mask → (1, size, size) float32 tensor."""
    t = torch.from_numpy((mask_2d > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="nearest")
    return t.squeeze(0)


def _maybe_extract(data_root: Path, split: str) -> Path:
    """Extract {split}.zip into data_root/{split} if not already present."""
    split_dir = data_root / split
    if split_dir.exists() and any(split_dir.rglob("*.png")):
        return split_dir
    zip_path = data_root / f"{split}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Neither extracted folder {split_dir} nor archive {zip_path} exist."
        )
    split_dir.mkdir(parents=True, exist_ok=True)
    print(f"[LEVIR-CD] Extracting {zip_path} → {split_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(split_dir)
    return split_dir


def _find_split_dir(data_root: Path, split: str) -> Path:
    """Locate the split directory (A/, B/, label/) under data_root, extracting if needed."""
    split_dir = _maybe_extract(data_root, split)
    # Look for A/B/label either directly or one level deep
    if (split_dir / "A").exists() and (split_dir / "B").exists():
        return split_dir
    for sub in split_dir.rglob("A"):
        if sub.is_dir() and (sub.parent / "B").is_dir() and (sub.parent / "label").is_dir():
            return sub.parent
    raise FileNotFoundError(f"Could not find A/B/label under {split_dir}")


class LEVIRCDDataset(Dataset):
    """
    LEVIR-CD bitemporal change detection dataset.

    Each item returns a pre/post RGB pair and a binary change mask. LEVIR does
    not supply per-timepoint semantic masks, so pre_mask and mid_mask are
    zeros — supervision comes from change_mask.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = SAM2_IMAGE_SIZE,
        augment: bool = True,
        crop_size: Optional[int] = None,
    ):
        """
        Args:
            data_dir   : root containing {split}.zip or {split}/
            split      : 'train' | 'val' | 'test'
            image_size : output SAM2 resolution
            augment    : enable spatial augmentations (only when split='train')
            crop_size  : if set and training, take a random crop of this size
                         before resizing (e.g. 256 for 256×256 crops).
        """
        self.data_root = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == "train")
        self.crop_size = crop_size if (crop_size and split == "train") else None

        self.split_dir = _find_split_dir(self.data_root, split)
        self.a_dir = self.split_dir / "A"
        self.b_dir = self.split_dir / "B"
        self.label_dir = self.split_dir / "label"

        self.names: list[str] = sorted(
            p.name for p in self.a_dir.glob("*.png")
            if (self.b_dir / p.name).exists() and (self.label_dir / p.name).exists()
        )
        if not self.names:
            raise FileNotFoundError(f"No matching A/B/label PNGs under {self.split_dir}")

        print(f"[LEVIRCDDataset] {split}: {len(self.names)} pairs at {self.split_dir}")

    def __len__(self) -> int:
        return len(self.names)

    def _load_triplet(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pre = np.asarray(Image.open(self.a_dir / name).convert("RGB"))
        mid = np.asarray(Image.open(self.b_dir / name).convert("RGB"))
        change = np.asarray(Image.open(self.label_dir / name).convert("L"))
        return pre, mid, change

    def _random_crop(
        self,
        pre: np.ndarray,
        mid: np.ndarray,
        change: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        H, W = pre.shape[:2]
        cs = int(self.crop_size)
        if H <= cs or W <= cs:
            return pre, mid, change
        top = random.randint(0, H - cs)
        left = random.randint(0, W - cs)
        return (
            pre[top:top + cs, left:left + cs],
            mid[top:top + cs, left:left + cs],
            change[top:top + cs, left:left + cs],
        )

    def _augment(
        self,
        pre: np.ndarray,
        mid: np.ndarray,
        change: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            pre = np.fliplr(pre).copy()
            mid = np.fliplr(mid).copy()
            change = np.fliplr(change).copy()
        if random.random() < 0.5:
            pre = np.flipud(pre).copy()
            mid = np.flipud(mid).copy()
            change = np.flipud(change).copy()
        k = random.randint(0, 3)
        if k:
            pre = np.rot90(pre, k).copy()
            mid = np.rot90(mid, k).copy()
            change = np.rot90(change, k).copy()
        return pre, mid, change

    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx]
        pre, mid, change = self._load_triplet(name)

        if self.crop_size is not None:
            pre, mid, change = self._random_crop(pre, mid, change)

        if self.augment:
            pre, mid, change = self._augment(pre, mid, change)

        pre_image = to_rgb_tensor(pre, self.image_size)
        mid_image = to_rgb_tensor(mid, self.image_size)
        change_mask = to_mask_tensor(change, self.image_size)
        zero_mask = torch.zeros_like(change_mask)

        return {
            "pre_image": pre_image,
            "mid_image": mid_image,
            "pre_mask": zero_mask,
            "mid_mask": zero_mask.clone(),
            "change_mask": change_mask,
            "image_name": name,
            "weeks_elapsed": 1,
        }


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = SAM2_IMAGE_SIZE,
    crop_size: Optional[int] = 256,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders for LEVIR-CD."""
    train_ds = LEVIRCDDataset(
        data_dir, split="train", image_size=image_size,
        augment=True, crop_size=crop_size,
    )
    val_ds = LEVIRCDDataset(data_dir, split="val", image_size=image_size, augment=False)
    test_ds = LEVIRCDDataset(data_dir, split="test", image_size=image_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
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
    root = Path("/home/tals/Documents/csm-sam/data/raw/LEVIR-CD")
    ds = LEVIRCDDataset(root, split="train", augment=True, crop_size=256)
    print(f"len(train) = {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {v!r}")
