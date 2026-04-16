"""Registration-Warp Baseline — the "no-learning cross-session" control for CSM-SAM.

This baseline deformably registers the pre-RT image to the mid-RT image and
warps the pre-RT GTVp/GTVn masks with the learned transform. It uses NO
learning and NO information other than the pre-RT masks themselves. If
CSM-SAM cannot beat this baseline, the novelty claim of cross-session
*learned* memory propagation collapses, so it MUST appear in the paper.

Registration backends (in priority order):
  1. ``itk-elastix`` deformable B-spline registration (preferred).
  2. SimpleITK BSplineTransformInitializer + gradient-descent registration.
  3. SimpleITK rigid-only registration (fallback when the above fail).

Usage:
    python baselines/registration_warp_baseline.py \\
        --data_dir data/processed \\
        --split test \\
        --output_dir results/registration_warp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    import itk  # itk-elastix registers an Elastix module on import.
    HAS_ELASTIX = hasattr(itk, "ElastixRegistrationMethod")
except Exception:
    HAS_ELASTIX = False

from csmsam.utils.metrics import (
    aggregate_metrics,
    evaluate_patient,
    format_results_table,
)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> "sitk.Image":
    return sitk.ReadImage(str(path))


def _arr(img: "sitk.Image") -> np.ndarray:
    return sitk.GetArrayFromImage(img)


def _voxel_spacing(img: "sitk.Image") -> tuple[float, float, float]:
    # SITK spacing is (x, y, z); evaluate_patient wants (dz, dy, dx).
    sx, sy, sz = img.GetSpacing()
    return (float(sz), float(sy), float(sx))


# ---------------------------------------------------------------------------
# Registration backends
# ---------------------------------------------------------------------------

def _register_elastix(fixed: "sitk.Image", moving: "sitk.Image"):
    """Deformable B-spline registration via itk-elastix. Returns a parameter object."""
    fixed_itk = itk.GetImageFromArray(sitk.GetArrayFromImage(fixed).astype(np.float32))
    moving_itk = itk.GetImageFromArray(sitk.GetArrayFromImage(moving).astype(np.float32))
    fixed_itk.SetSpacing(tuple(float(s) for s in fixed.GetSpacing()))
    moving_itk.SetSpacing(tuple(float(s) for s in moving.GetSpacing()))

    param_obj = itk.ParameterObject.New()
    param_obj.AddParameterMap(param_obj.GetDefaultParameterMap("rigid"))
    param_obj.AddParameterMap(param_obj.GetDefaultParameterMap("bspline"))

    _, result_params = itk.elastix_registration_method(
        fixed_itk, moving_itk, parameter_object=param_obj, log_to_console=False
    )
    return result_params


def _warp_elastix(mask: "sitk.Image", result_params) -> np.ndarray:
    mask_itk = itk.GetImageFromArray(sitk.GetArrayFromImage(mask).astype(np.float32))
    mask_itk.SetSpacing(tuple(float(s) for s in mask.GetSpacing()))
    # Force nearest-neighbor interpolation for the mask warp.
    n_maps = result_params.GetNumberOfParameterMaps()
    for i in range(n_maps):
        result_params.SetParameter(i, "FinalBSplineInterpolationOrder", "0")
        result_params.SetParameter(i, "ResultImagePixelType", "float")
    warped = itk.transformix_filter(mask_itk, result_params, log_to_console=False)
    arr = itk.GetArrayFromImage(warped)
    return (arr > 0.5).astype(np.uint8)


def _register_sitk_bspline(fixed: "sitk.Image", moving: "sitk.Image") -> "sitk.Transform":
    """SimpleITK gradient-descent B-spline registration with affine init."""
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    # Affine init (fast, coarse alignment).
    init_tx = sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1, seed=42)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=80,
                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(init_tx, inPlace=False)
    affine_tx = reg.Execute(fixed_f, moving_f)

    # B-spline refinement on top of affine.
    mesh_size = [8] * fixed_f.GetDimension()
    bspline_tx = sitk.BSplineTransformInitializer(fixed_f, mesh_size, order=3)
    composite = sitk.CompositeTransform([affine_tx, bspline_tx])
    reg2 = sitk.ImageRegistrationMethod()
    reg2.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg2.SetMetricSamplingStrategy(reg2.RANDOM)
    reg2.SetMetricSamplingPercentage(0.1, seed=42)
    reg2.SetInterpolator(sitk.sitkLinear)
    reg2.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=40)
    reg2.SetInitialTransform(composite, inPlace=True)
    reg2.Execute(fixed_f, moving_f)
    return composite


def _register_sitk_rigid(fixed: "sitk.Image", moving: "sitk.Image") -> "sitk.Transform":
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)
    init_tx = sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1, seed=42)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(init_tx, inPlace=False)
    return reg.Execute(fixed_f, moving_f)


def _warp_sitk(mask: "sitk.Image", reference: "sitk.Image", tx: "sitk.Transform") -> np.ndarray:
    warped = sitk.Resample(
        mask, reference, tx, sitk.sitkNearestNeighbor, 0.0, mask.GetPixelID(),
    )
    arr = sitk.GetArrayFromImage(warped)
    return (arr > 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-patient pipeline
# ---------------------------------------------------------------------------

def _zeros_like(img: "sitk.Image") -> np.ndarray:
    return np.zeros(sitk.GetArrayFromImage(img).shape, dtype=np.uint8)


def process_patient(patient_dir: Path) -> dict | None:
    """Register pre→mid, warp pre-RT masks, and evaluate vs mid-RT GT."""
    pre_img_path = patient_dir / "pre_image.nii.gz"
    mid_img_path = patient_dir / "mid_image.nii.gz"
    if not (pre_img_path.exists() and mid_img_path.exists()):
        return None

    pre_img = _read(pre_img_path)
    mid_img = _read(mid_img_path)

    def _load_mask_or_empty(path: Path, template: "sitk.Image") -> "sitk.Image":
        if path.exists():
            return _read(path)
        zero = sitk.Image(template.GetSize(), sitk.sitkUInt8)
        zero.CopyInformation(template)
        return zero

    pre_gtvp = _load_mask_or_empty(patient_dir / "pre_GTVp.nii.gz", pre_img)
    pre_gtvn = _load_mask_or_empty(patient_dir / "pre_GTVn.nii.gz", pre_img)
    mid_gtvp = _load_mask_or_empty(patient_dir / "mid_GTVp.nii.gz", mid_img)
    mid_gtvn = _load_mask_or_empty(patient_dir / "mid_GTVn.nii.gz", mid_img)

    # Register pre (moving) → mid (fixed).
    backend = "rigid"
    try:
        if HAS_ELASTIX:
            params = _register_elastix(mid_img, pre_img)
            warped_gtvp = _warp_elastix(pre_gtvp, params)
            warped_gtvn = _warp_elastix(pre_gtvn, params)
            backend = "elastix"
        else:
            try:
                tx = _register_sitk_bspline(mid_img, pre_img)
                backend = "sitk_bspline"
            except Exception:
                tx = _register_sitk_rigid(mid_img, pre_img)
                backend = "sitk_rigid"
            warped_gtvp = _warp_sitk(pre_gtvp, mid_img, tx)
            warped_gtvn = _warp_sitk(pre_gtvn, mid_img, tx)
    except Exception as e:
        print(f"  [{patient_dir.name}] registration failed ({e}); using empty prediction.")
        warped_gtvp = _zeros_like(mid_img)
        warped_gtvn = _zeros_like(mid_img)
        backend = "failed"

    gt_gtvp = (_arr(mid_gtvp) > 0.5).astype(np.uint8)
    gt_gtvn = (_arr(mid_gtvn) > 0.5).astype(np.uint8)
    pred_combined = ((warped_gtvp + warped_gtvn) > 0).astype(np.uint8)

    metrics = evaluate_patient(
        pred_masks=pred_combined,
        pred_gtvp=warped_gtvp,
        pred_gtvn=warped_gtvn,
        target_gtvp=gt_gtvp,
        target_gtvn=gt_gtvn,
        voxel_spacing=_voxel_spacing(mid_img),
    )
    metrics["patient_id"] = patient_dir.name
    metrics["backend"] = backend
    return metrics


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def run(data_dir: str, split: str, output_dir: str, n_workers: int = 1) -> dict:
    if not HAS_SITK:
        raise RuntimeError("SimpleITK is required for this baseline. `pip install SimpleITK`.")

    in_root = Path(data_dir) / split
    if not in_root.exists():
        raise FileNotFoundError(f"Split dir not found: {in_root}")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(
        d for d in in_root.iterdir()
        if d.is_dir() and (d / "mid_image.nii.gz").exists()
    )
    print("=" * 60)
    print(f"Registration-Warp Baseline — {split} ({len(patient_dirs)} patients)")
    print(f"Backend: {'itk-elastix' if HAS_ELASTIX else 'SimpleITK'}")
    print("=" * 60)

    # n_workers kept as CLI knob; registration is CPU-heavy and SITK is not fork-safe
    # with many backends, so the simple sequential loop is the safe default.
    if n_workers != 1:
        print(f"[note] --n_workers={n_workers} ignored; running sequentially.")

    per_patient = []
    for pdir in tqdm(patient_dirs, desc="Patients"):
        try:
            m = process_patient(pdir)
            if m is not None:
                per_patient.append(m)
        except Exception as e:
            print(f"  [{pdir.name}] error: {e}")

    agg = aggregate_metrics(per_patient)
    results = {"aggregate": agg, "per_patient": per_patient, "split": split,
               "backend": "itk-elastix" if HAS_ELASTIX else "SimpleITK"}

    with open(out_root / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_root / 'metrics.json'}")

    print(format_results_table(
        {"Registration-Warp": agg},
        title=f"Registration-Warp Baseline ({split})",
    ))
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="results/registration_warp")
    parser.add_argument("--n_workers", type=int, default=1)
    args = parser.parse_args()

    run(args.data_dir, args.split, args.output_dir, args.n_workers)
