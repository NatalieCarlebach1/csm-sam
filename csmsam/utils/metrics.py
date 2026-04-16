"""
Evaluation metrics for HNTS-MRG 2024.

Primary metric: aggDSC = (DSC_GTVp + DSC_GTVn) / 2
Secondary metric: HD95 (95th percentile Hausdorff distance in mm)

Reference: HNTS-MRG 2024 challenge evaluation protocol.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from medpy.metric.binary import hd95 as medpy_hd95
    HAS_MEDPY = True
except ImportError:
    HAS_MEDPY = False


def compute_dice(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """
    Compute Dice Similarity Coefficient.

    Args:
        pred   : binary np array (any shape)
        target : binary np array (same shape)
        smooth : smoothing factor to avoid division by zero

    Returns:
        DSC in [0, 1]
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()

    intersection = (pred & target).sum()
    dsc = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return float(dsc)


def compute_hd95(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Compute 95th percentile Hausdorff distance in mm.

    Requires medpy. Returns inf if no tumor in pred or target.

    Args:
        pred          : binary 3D np array (D, H, W)
        target        : binary 3D np array (D, H, W)
        voxel_spacing : (dz, dy, dx) in mm
    """
    if not HAS_MEDPY:
        return float("nan")

    pred_b = pred.astype(bool)
    target_b = target.astype(bool)

    if not pred_b.any() or not target_b.any():
        return float("inf")

    try:
        return float(medpy_hd95(pred_b, target_b, voxelspacing=voxel_spacing))
    except Exception:
        return float("inf")


def compute_agg_dsc(
    pred_gtvp: np.ndarray,
    pred_gtvn: np.ndarray,
    target_gtvp: np.ndarray,
    target_gtvn: np.ndarray,
) -> dict[str, float]:
    """
    Compute aggregated DSC (primary HNTS-MRG metric).

    aggDSC = (DSC_GTVp + DSC_GTVn) / 2

    Args:
        pred_gtvp   : (D, H, W) binary prediction for GTVp
        pred_gtvn   : (D, H, W) binary prediction for GTVn
        target_gtvp : (D, H, W) ground truth GTVp
        target_gtvn : (D, H, W) ground truth GTVn

    Returns:
        dict with 'dsc_gtvp', 'dsc_gtvn', 'agg_dsc'
    """
    dsc_p = compute_dice(pred_gtvp, target_gtvp)
    dsc_n = compute_dice(pred_gtvn, target_gtvn)
    agg = (dsc_p + dsc_n) / 2.0

    return {
        "dsc_gtvp": dsc_p,
        "dsc_gtvn": dsc_n,
        "agg_dsc": agg,
    }


def evaluate_patient(
    pred_masks: np.ndarray,
    pred_gtvp: np.ndarray,
    pred_gtvn: np.ndarray,
    target_gtvp: np.ndarray,
    target_gtvn: np.ndarray,
    voxel_spacing: tuple[float, float, float] = (3.0, 1.0, 1.0),
) -> dict[str, float]:
    """
    Full evaluation for one patient.

    Args:
        pred_masks  : (D, H, W) combined binary prediction
        pred_gtvp   : (D, H, W) GTVp binary prediction
        pred_gtvn   : (D, H, W) GTVn binary prediction
        target_gtvp : (D, H, W) GTVp ground truth
        target_gtvn : (D, H, W) GTVn ground truth
        voxel_spacing: (dz, dy, dx) in mm for HD95

    Returns:
        dict with all metric values
    """
    metrics = compute_agg_dsc(pred_gtvp, pred_gtvn, target_gtvp, target_gtvn)

    # HD95 on combined mask
    target_combined = ((target_gtvp + target_gtvn) > 0).astype(bool)
    pred_combined = pred_masks.astype(bool)
    metrics["hd95"] = compute_hd95(pred_combined, target_combined, voxel_spacing)

    # HD95 per structure
    metrics["hd95_gtvp"] = compute_hd95(
        pred_gtvp.astype(bool), target_gtvp.astype(bool), voxel_spacing
    )
    metrics["hd95_gtvn"] = compute_hd95(
        pred_gtvn.astype(bool), target_gtvn.astype(bool), voxel_spacing
    )

    return metrics


def aggregate_metrics(per_patient_metrics: list[dict]) -> dict[str, float]:
    """
    Compute dataset-level statistics from per-patient metrics.

    Returns mean and std for each metric.
    """
    if not per_patient_metrics:
        return {}

    keys = per_patient_metrics[0].keys()
    aggregated = {}

    for key in keys:
        raw = [m[key] for m in per_patient_metrics]
        if not all(isinstance(v, (int, float, np.floating, np.integer)) for v in raw):
            continue
        values = [v for v in raw if not np.isinf(v) and not np.isnan(v)]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_median"] = float(np.median(values))
        else:
            aggregated[f"{key}_mean"] = float("nan")

    return aggregated


def format_results_table(
    methods: dict[str, dict],
    title: str = "HNTS-MRG 2024 Mid-RT Segmentation Results",
) -> str:
    """
    Format comparison table for paper/README.

    Args:
        methods: {method_name: metrics_dict}
            metrics_dict should have: agg_dsc_mean, dsc_gtvp_mean, dsc_gtvn_mean, hd95_mean
    """
    header = f"\n{'=' * 70}\n{title}\n{'=' * 70}\n"
    cols = ["Method", "GTVp DSC", "GTVn DSC", "aggDSC", "HD95 (mm)"]
    row_fmt = "{:<30} {:>10} {:>10} {:>10} {:>12}"

    lines = [header, row_fmt.format(*cols), "-" * 70]

    for method, metrics in methods.items():
        p = metrics.get("dsc_gtvp_mean", float("nan"))
        n = metrics.get("dsc_gtvn_mean", float("nan"))
        agg = metrics.get("agg_dsc_mean", float("nan"))
        hd = metrics.get("hd95_mean", float("nan"))

        lines.append(row_fmt.format(
            method,
            f"{p:.4f}" if not np.isnan(p) else "-",
            f"{n:.4f}" if not np.isnan(n) else "-",
            f"{agg:.4f}" if not np.isnan(agg) else "-",
            f"{hd:.2f}" if not np.isnan(hd) and not np.isinf(hd) else "-",
        ))

    lines.append("=" * 70 + "\n")
    return "\n".join(lines)
