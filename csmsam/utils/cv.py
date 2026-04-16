"""Deterministic k-fold splits for small medical datasets.

Provides stable, reproducible cross-validation splits for CSM-SAM experiments.
The splits are deterministic in the (sorted patient IDs, seed) pair, so
callers may pass IDs in any order and still obtain the same fold assignment.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def kfold_split(
    patient_ids: list[str],
    fold: int,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Return (train_ids, val_ids) for the requested fold (0-indexed).

    The patient list is sorted and then shuffled with a seeded RNG so the
    result is stable regardless of the order callers pass patients in.

    Args:
        patient_ids : list of patient directory names.
        fold        : 0-indexed fold number.
        n_folds     : total number of folds.
        seed        : RNG seed for the deterministic shuffle.

    Returns:
        (train_ids, val_ids) — disjoint lists whose union equals the input.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 (got {n_folds}).")
    if not (0 <= fold < n_folds):
        raise ValueError(f"fold must satisfy 0 <= fold < n_folds (got {fold} / {n_folds}).")
    if not patient_ids:
        raise ValueError("patient_ids is empty — nothing to split.")

    ordered = sorted(patient_ids)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n = len(ordered)
    # Use ceil-style chunking so no patient is dropped.
    fold_sizes = [n // n_folds + (1 if i < n % n_folds else 0) for i in range(n_folds)]
    start = sum(fold_sizes[:fold])
    end = start + fold_sizes[fold]

    val_ids = ordered[start:end]
    train_ids = ordered[:start] + ordered[end:]
    return train_ids, val_ids


def list_patients(data_dir: str | Path, split: str = "train") -> list[str]:
    """Return sorted patient-dir names under ``data_dir/split/``.

    Applies the same filter HNTSMRGDataset uses — the directory must contain
    ``mid_image.nii.gz`` — so CV splits stay aligned with the dataset.
    """
    root = Path(data_dir) / split
    if not root.exists():
        raise FileNotFoundError(f"Split directory not found: {root}")

    patients = [
        d.name
        for d in root.iterdir()
        if d.is_dir() and (d / "mid_image.nii.gz").exists()
    ]
    return sorted(patients)


def export_split_manifest(
    train_ids: list[str],
    val_ids: list[str],
    output_path: str | Path,
    fold: int | None = None,
    n_folds: int | None = None,
    seed: int | None = None,
) -> Path:
    """Write a JSON manifest describing the split for reproducibility.

    The manifest records fold metadata plus the exact patient lists so
    reviewers can reproduce the exact partition used for a reported number.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "fold": fold,
        "n_folds": n_folds,
        "seed": seed,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
    }
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return output_path
