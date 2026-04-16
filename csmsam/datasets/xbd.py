"""
xBD Building Damage Change Detection Dataset for CSM-SAM.

The xBD dataset (xView2 challenge) provides paired pre- and post-disaster
satellite imagery with building polygon annotations. The pre-disaster JSONs
contain building footprints only; the post-disaster JSONs additionally carry
a damage `subtype` per building.

Disk layout:
    data/raw/xBD/
        train/
            images/{disaster}_{id}_pre_disaster.png
            images/{disaster}_{id}_post_disaster.png
            labels/{disaster}_{id}_pre_disaster.json
            labels/{disaster}_{id}_post_disaster.json
        test/
            images/...
            labels/...
            targets/...

Each label JSON contains `features.xy[i].wkt` polygon strings in pixel space
(1024x1024). Post-disaster features carry `properties.subtype` in
{"no-damage", "minor-damage", "major-damage", "destroyed", "un-classified"}.

We reuse SAM2 normalization helpers from `hnts_mrg` so the tensors produced
here are drop-in compatible with the rest of CSM-SAM.

Dependency choice:
    Prefer `shapely` + `rasterio.features.rasterize` for polygon rasterization
    (fast and correct). If either is missing we fall back to `PIL.ImageDraw`
    with a small regex WKT parser. The fallback is slightly slower but has
    no native dependencies. The choice is made once at import time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from csmsam.datasets.hnts_mrg import (
    SAM2_IMAGE_SIZE,
    SAM2_MEAN,
    SAM2_STD,
    to_mask_tensor,
    to_rgb_tensor,
)


# --- Rasterization backend selection ---------------------------------------
try:
    from shapely import wkt as _shapely_wkt  # type: ignore
    from shapely.geometry import Polygon as _ShapelyPolygon  # type: ignore
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False

try:
    from rasterio.features import rasterize as _rio_rasterize  # type: ignore
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

USE_SHAPELY_RASTERIO = HAS_SHAPELY and HAS_RASTERIO


# Damage subtype → class index mapping (post-disaster)
DAMAGE_CLASS_MAP: dict[str, int] = {
    "un-classified": 0,
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}

# Regex fallback: POLYGON ((x1 y1, x2 y2, ...), (hole...)) — capture each ring.
_POLYGON_RING_RE = re.compile(r"\(([^()]+)\)")
_COORD_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def _parse_wkt_polygon(wkt_str: str) -> list[list[tuple[float, float]]]:
    """
    Parse a WKT POLYGON string into a list of rings.

    Returns a list like [outer_ring, hole1, hole2, ...] where each ring is
    a list of (x, y) tuples. Handles POLYGON and MULTIPOLYGON-ish strings
    by flattening all rings (xBD in practice uses single POLYGONs).
    """
    rings: list[list[tuple[float, float]]] = []
    for ring_match in _POLYGON_RING_RE.findall(wkt_str):
        coords = [(float(x), float(y)) for x, y in _COORD_RE.findall(ring_match)]
        if coords:
            rings.append(coords)
    return rings


def _rasterize_ring_pil(
    mask: np.ndarray,
    rings: list[list[tuple[float, float]]],
    value: int,
) -> None:
    """
    Paint one polygon (outer ring + optional holes) onto `mask` with PIL.

    The first ring is the outer boundary; subsequent rings are holes (painted
    as 0). Operates in-place on `mask`.
    """
    if not rings:
        return
    H, W = mask.shape
    img = Image.fromarray(mask, mode="L") if mask.dtype == np.uint8 else Image.fromarray(mask.astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(img)
    outer = [(float(x), float(y)) for x, y in rings[0]]
    if len(outer) >= 3:
        draw.polygon(outer, outline=value, fill=value)
    for hole in rings[1:]:
        h = [(float(x), float(y)) for x, y in hole]
        if len(h) >= 3:
            draw.polygon(h, outline=0, fill=0)
    mask[...] = np.array(img, dtype=mask.dtype)


def read_building_polygons(json_path: str | Path) -> list[dict]:
    """
    Read per-building polygons (and damage subtype if present) from an xBD
    label JSON. Returns a list of dicts with keys:

        {
            "polygon_xy" : list[tuple[float, float]]   # outer ring, native 1024 px
            "rings"      : list[list[tuple[float, float]]]  # outer + any holes
            "subtype"    : str   # "no-damage"|"minor-damage"|"major-damage"|
                                  #  "destroyed"|"un-classified"|"" (pre-disaster)
            "cls"        : int   # DAMAGE_CLASS_MAP[subtype] or 1 if pre-disaster
            "uid"        : str   # feature properties.uid if present, else ""
        }

    Coordinates are in the native 1024x1024 xBD pixel grid. Callers that
    resize images before inference must rescale these polygons themselves —
    this function intentionally does NO resizing so the official xView2
    scorer (which expects 1024px polygons) can consume the output directly.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        return []
    with open(json_path) as f:
        payload = json.load(f)
    features = payload.get("features", {}).get("xy", []) or []

    out: list[dict] = []
    for feat in features:
        props = feat.get("properties", {}) or {}
        if props.get("feature_type", "building") != "building":
            continue
        wkt_str = feat.get("wkt", "")
        if not wkt_str:
            continue
        rings = _parse_wkt_polygon(wkt_str)
        if not rings:
            continue
        subtype = props.get("subtype", "")
        cls = DAMAGE_CLASS_MAP.get(subtype, 1 if subtype == "" else 0)
        out.append({
            "polygon_xy": rings[0],
            "rings": rings,
            "subtype": subtype,
            "cls": int(cls),
            "uid": str(props.get("uid", "")),
        })
    return out


def rasterize_polygons_to_mask(
    json_path: str | Path,
    H: int,
    W: int,
    damage_as_class: bool,
) -> np.ndarray:
    """
    Rasterize all polygons from an xBD label JSON into an (H, W) uint8 mask.

    Args:
        json_path       : path to xBD label JSON
        H, W            : output mask spatial size (pixels)
        damage_as_class : if True, use post-disaster subtype → class index
                          (no-damage=1, minor=2, major=3, destroyed=4,
                          un-classified=0). If False, every building is 1.

    Returns:
        mask : (H, W) uint8
    """
    json_path = Path(json_path)
    with open(json_path) as f:
        payload = json.load(f)

    features = payload.get("features", {}).get("xy", []) or []

    # Collect (ring_list, class_value) pairs
    polys: list[tuple[list[list[tuple[float, float]]], int]] = []
    for feat in features:
        props = feat.get("properties", {}) or {}
        if props.get("feature_type", "building") != "building":
            continue
        wkt_str = feat.get("wkt", "")
        if not wkt_str:
            continue

        if damage_as_class:
            subtype = props.get("subtype", "un-classified")
            cls = DAMAGE_CLASS_MAP.get(subtype, 0)
        else:
            cls = 1

        if USE_SHAPELY_RASTERIO:
            try:
                geom = _shapely_wkt.loads(wkt_str)
            except Exception:
                continue
            polys.append((geom, cls))  # type: ignore[arg-type]
        else:
            rings = _parse_wkt_polygon(wkt_str)
            if rings:
                polys.append((rings, cls))

    mask = np.zeros((H, W), dtype=np.uint8)
    if not polys:
        return mask

    if USE_SHAPELY_RASTERIO:
        shapes = [(geom, int(cls)) for geom, cls in polys if not geom.is_empty]
        if shapes:
            mask = _rio_rasterize(
                shapes,
                out_shape=(H, W),
                fill=0,
                dtype=np.uint8,
            )
        return mask

    # PIL fallback — paint in ascending class order so higher damage overrides
    # lower (helps overlapping rare edge cases).
    polys.sort(key=lambda p: p[1])
    for rings, cls in polys:
        _rasterize_ring_pil(mask, rings, int(cls))
    return mask


def _load_rgb_image(path: Path, size: int = SAM2_IMAGE_SIZE) -> torch.Tensor:
    """Load an xBD PNG as a SAM2-normalized (3, size, size) tensor."""
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    t = (t - SAM2_MEAN[:, None, None]) / SAM2_STD[:, None, None]
    return t


class XBDDataset(Dataset):
    """
    xBD building-damage change detection dataset.

    Each item is a pre-disaster + post-disaster image pair with derived
    building masks and a per-pixel damage-class map. Designed to slot into
    the CSM-SAM pipeline as a cross-domain change-detection benchmark —
    "pre" corresponds to `pre_image` (before disaster) and "mid" to
    `mid_image` (post-disaster).

    `split` semantics:
        "train" : all matching pairs under {root}/train
        "test"  : all matching pairs under {root}/test
        "val"   : if {root}/test exists, aliased to "test"; otherwise the
                  last 10% of train indices (deterministic).

    __getitem__ returns:
        {
            "pre_image"       : (3, H, W) SAM2-normalized RGB tensor,
            "mid_image"       : (3, H, W) SAM2-normalized RGB tensor,
            "pre_mask"        : (1, H, W) binary building mask,
            "mid_mask"        : (1, H, W) binary building mask,
            "damage_mask"     : (H, W) long tensor, values in [0..4],
            "change_mask"     : (1, H, W) binary XOR(pre_mask, mid_mask),
            "damage_polygons" : list[dict] — per-building {polygon_xy, rings,
                                 subtype, cls, uid} in native 1024-px coords,
                                 taken from the POST-disaster JSON so the
                                 xView2 scorer can compute per-building F1.
            "pre_polygons"    : list[dict] — same shape, from PRE-disaster JSON.
            "image_name"      : "{disaster}_{id}",
            "disaster"        : "{disaster}",
            "weeks_elapsed"   : 1,
        }

    Note: `damage_polygons` / `pre_polygons` are Python lists of dicts; they
    must be kept out of default DataLoader collation. Use a custom
    `collate_fn` (or batch_size=1 at eval time) when consuming them.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = SAM2_IMAGE_SIZE,
        val_fraction: float = 0.1,
        augment: bool = False,
    ):
        self.data_root = Path(data_dir)
        self.image_size = int(image_size)
        self.split = split
        self.val_fraction = float(val_fraction)
        self.augment = bool(augment)  # reserved for future use

        train_dir = self.data_root / "train"
        test_dir = self.data_root / "test"

        if split == "train":
            self.split_dir = train_dir
            samples = self._index_split(train_dir)
            # Hold out the last val_fraction for val; train gets the rest.
            n_val = max(1, int(len(samples) * self.val_fraction)) if samples else 0
            self.samples = samples[: len(samples) - n_val] if n_val else samples
        elif split == "test":
            self.split_dir = test_dir
            self.samples = self._index_split(test_dir)
        elif split == "val":
            if test_dir.exists() and any(test_dir.glob("images/*_pre_disaster.png")):
                self.split_dir = test_dir
                self.samples = self._index_split(test_dir)
            else:
                self.split_dir = train_dir
                all_train = self._index_split(train_dir)
                n_val = max(1, int(len(all_train) * self.val_fraction)) if all_train else 0
                self.samples = all_train[len(all_train) - n_val:] if n_val else []
        else:
            raise ValueError(f"Unknown split '{split}' (expected train|val|test)")

        if not self.samples:
            raise FileNotFoundError(
                f"No xBD samples found for split='{split}' under {self.data_root}. "
                "Expected images/*_pre_disaster.png alongside images/*_post_disaster.png."
            )

        backend = "shapely+rasterio" if USE_SHAPELY_RASTERIO else "PIL fallback"
        print(
            f"[XBDDataset] split={split}: {len(self.samples)} pairs "
            f"(rasterizer: {backend})"
        )

    @staticmethod
    def _index_split(split_dir: Path) -> list[dict]:
        """Build a sorted list of {pre_image, post_image, pre_label, post_label, name, disaster}."""
        if not split_dir.exists():
            return []
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not images_dir.exists():
            return []

        samples: list[dict] = []
        for pre_img in sorted(images_dir.glob("*_pre_disaster.png")):
            stem = pre_img.stem  # "{disaster}_{id}_pre_disaster"
            if not stem.endswith("_pre_disaster"):
                continue
            base = stem[: -len("_pre_disaster")]  # "{disaster}_{id}"
            post_img = images_dir / f"{base}_post_disaster.png"
            if not post_img.exists():
                continue
            pre_label = labels_dir / f"{base}_pre_disaster.json"
            post_label = labels_dir / f"{base}_post_disaster.json"
            # Extract disaster name (everything before the last "_{id}")
            parts = base.rsplit("_", 1)
            disaster = parts[0] if len(parts) == 2 else base
            samples.append({
                "name": base,
                "disaster": disaster,
                "pre_image": pre_img,
                "post_image": post_img,
                "pre_label": pre_label,
                "post_label": post_label,
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        size = self.image_size

        # Images → SAM2 tensors
        pre_image = _load_rgb_image(s["pre_image"], size=size)
        mid_image = _load_rgb_image(s["post_image"], size=size)

        # Masks rasterized at native 1024x1024, then resized by to_mask_tensor.
        # (Rasterizing at native resolution preserves small-building fidelity.)
        native_H = native_W = SAM2_IMAGE_SIZE
        if s["pre_label"].exists():
            pre_building = rasterize_polygons_to_mask(
                s["pre_label"], native_H, native_W, damage_as_class=False
            )
        else:
            pre_building = np.zeros((native_H, native_W), dtype=np.uint8)

        if s["post_label"].exists():
            post_damage = rasterize_polygons_to_mask(
                s["post_label"], native_H, native_W, damage_as_class=True
            )
        else:
            post_damage = np.zeros((native_H, native_W), dtype=np.uint8)

        post_building = (post_damage > 0).astype(np.uint8)

        pre_mask = to_mask_tensor(pre_building.astype(np.float32), size=size)
        mid_mask = to_mask_tensor(post_building.astype(np.float32), size=size)

        # Damage mask resized with nearest interpolation, kept as long tensor.
        damage_t = torch.from_numpy(post_damage.astype(np.int64)).unsqueeze(0).unsqueeze(0).float()
        damage_t = torch.nn.functional.interpolate(damage_t, size=(size, size), mode="nearest")
        damage_mask = damage_t.squeeze(0).squeeze(0).long()  # (H, W)

        # Change mask: XOR of pre and post building masks (binary)
        change_mask = (
            (pre_mask > 0.5).float() - (mid_mask > 0.5).float()
        ).abs().clamp(0, 1)  # (1, H, W)

        # Per-building polygons (native 1024-px coords) for official xView2
        # per-building F1 scoring. Lists of dicts — DO NOT default-collate.
        damage_polygons = read_building_polygons(s["post_label"])
        pre_polygons = read_building_polygons(s["pre_label"])

        return {
            "pre_image": pre_image,
            "mid_image": mid_image,
            "pre_mask": pre_mask,
            "mid_mask": mid_mask,
            "damage_mask": damage_mask,
            "change_mask": change_mask,
            "damage_polygons": damage_polygons,
            "pre_polygons": pre_polygons,
            "image_name": s["name"],
            "disaster": s["disaster"],
            "weeks_elapsed": 1,
        }


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = SAM2_IMAGE_SIZE,
    pin_memory: bool = True,
    val_fraction: float = 0.1,
) -> dict[str, DataLoader]:
    """
    Build {train, val, test} DataLoaders for xBD.

    val = last `val_fraction` of the training index when a separate test
    directory exists; otherwise val aliases the test split (see XBDDataset).
    """
    train_ds = XBDDataset(data_dir, split="train", image_size=image_size, val_fraction=val_fraction)
    val_ds = XBDDataset(data_dir, split="val", image_size=image_size, val_fraction=val_fraction)
    test_ds: Optional[XBDDataset]
    try:
        test_ds = XBDDataset(data_dir, split="test", image_size=image_size, val_fraction=val_fraction)
    except FileNotFoundError:
        test_ds = None

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
    loaders: dict[str, DataLoader] = {"train": train_loader, "val": val_loader}
    if test_ds is not None:
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders


if __name__ == "__main__":
    # Smoke test — indexes the dataset, loads one sample, prints shapes.
    import argparse

    parser = argparse.ArgumentParser(description="xBD dataset smoke test")
    parser.add_argument(
        "--data_dir",
        default="data/raw/xBD",
        help="Root dir containing train/ and test/ subfolders",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()

    print(f"Rasterization backend: {'shapely+rasterio' if USE_SHAPELY_RASTERIO else 'PIL fallback'}")
    ds = XBDDataset(args.data_dir, split=args.split)
    print(f"Dataset size: {len(ds)}")
    sample = ds[args.idx]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: {v!r}")
