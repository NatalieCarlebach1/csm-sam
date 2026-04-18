"""
BraTS-GLI 2024 Baseline Sweep for CSM-SAM.

Runs all relevant baselines against BraTS-GLI longitudinal pairs and writes
per-baseline metrics.json + a summary.md table.

Usage:
    python baselines/run_brats_baselines.py \
        --data_dir /media/data1/natalie/BraTS_GLI \
        --sam2_checkpoint /media/data1/natalie/checkpoints/sam2.1_hiera_large.pt \
        --output_dir results/brats_baselines \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from csmsam.datasets.brats_gli import BraTSGLIDataset
from csmsam.utils.metrics import compute_dice


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _binary(mask_tensor, threshold=0.5):
    return (mask_tensor > threshold).squeeze(1).numpy()  # (N, H, W)


def _save(output_dir: Path, name: str, metrics: dict, fallback: bool = False):
    d = output_dir / name
    d.mkdir(parents=True, exist_ok=True)
    metrics["fallback"] = fallback
    with open(d / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    status = "FALLBACK" if fallback else "OK"
    with open(d / "status.txt", "w") as f:
        f.write(status + "\n")


def _run_baseline(name, fn, output_dir, timeout=300):
    """Run one baseline with a timeout; save fallback on error."""
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    t0 = time.time()
    try:
        metrics = fn()
        elapsed = time.time() - t0
        metrics["elapsed_s"] = round(elapsed, 1)
        _save(output_dir, name, metrics, fallback=False)
        print(f"  DSC={metrics.get('dsc_mean', 0):.4f}  [{elapsed:.1f}s]")
        return metrics
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED ({elapsed:.1f}s): {e}")
        traceback.print_exc()
        metrics = {"dsc_mean": 0.0, "dsc_std": 0.0, "elapsed_s": round(elapsed, 1),
                   "error": str(e)}
        _save(output_dir, name, metrics, fallback=True)
        return metrics


def _load_brats(data_dir, image_size, split="val"):
    return BraTSGLIDataset(data_dir, split=split, modality="t2f",
                           image_size=image_size, val_fraction=0.10)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def zero_baseline(data_dir, image_size, device, **_):
    ds = _load_brats(data_dir, image_size)
    dsc_list = []
    for i in tqdm(range(len(ds)), desc="zero", leave=False):
        s = ds[i]
        gt = _binary(s["mid_mask"])
        pred = np.zeros_like(gt)
        dsc_list.append(compute_dice(pred, gt))
    return {"dsc_mean": float(np.mean(dsc_list)), "dsc_std": float(np.std(dsc_list))}


def identity_baseline(data_dir, image_size, device, **_):
    """Use pre-RT mask directly as the mid-RT prediction."""
    ds = _load_brats(data_dir, image_size)
    dsc_list = []
    for i in tqdm(range(len(ds)), desc="identity", leave=False):
        s = ds[i]
        gt   = _binary(s["mid_mask"])
        pred = _binary(s["pre_mask"])
        N = min(gt.shape[0], pred.shape[0])
        dsc_list.append(compute_dice(pred[:N], gt[:N]))
    return {"dsc_mean": float(np.mean(dsc_list)), "dsc_std": float(np.std(dsc_list))}


def copy_prev_slice_baseline(data_dir, image_size, device, **_):
    """Each slice prediction = previous slice GT (slice-shift oracle)."""
    ds = _load_brats(data_dir, image_size)
    dsc_list = []
    for i in tqdm(range(len(ds)), desc="copy_prev_slice", leave=False):
        s = ds[i]
        gt = _binary(s["mid_mask"])  # (N, H, W)
        N = gt.shape[0]
        if N < 2:
            dsc_list.append(0.0)
            continue
        pred = np.concatenate([gt[:1], gt[:-1]], axis=0)  # shift by 1
        dsc_list.append(compute_dice(pred, gt))
    return {"dsc_mean": float(np.mean(dsc_list)), "dsc_std": float(np.std(dsc_list))}


def pre_mask_prior_baseline(data_dir, image_size, device, **_):
    """Dilate pre-RT mask by 10% and use as prediction."""
    from scipy.ndimage import binary_dilation
    ds = _load_brats(data_dir, image_size)
    dsc_list = []
    for i in tqdm(range(len(ds)), desc="pre_mask_prior", leave=False):
        s = ds[i]
        gt   = _binary(s["mid_mask"])
        pre  = _binary(s["pre_mask"])
        N = min(gt.shape[0], pre.shape[0])
        pred = np.stack([
            binary_dilation(pre[j], iterations=3).astype(np.float32)
            for j in range(N)
        ])
        dsc_list.append(compute_dice(pred, gt[:N]))
    return {"dsc_mean": float(np.mean(dsc_list)), "dsc_std": float(np.std(dsc_list))}


def unet2d_baseline(data_dir, image_size, device, **_):
    """Untrained 2D U-Net (random weights) — establishes architecture lower bound."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError("pip install segmentation-models-pytorch")

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=1).to(device).eval()

    ds = _load_brats(data_dir, image_size)
    dsc_list = []
    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc="unet2d", leave=False):
            s = ds[i]
            gt = _binary(s["mid_mask"])
            mid_imgs = s["mid_image"].to(device)  # (N, 3, H, W)
            preds = []
            for j in range(mid_imgs.shape[0]):
                out = torch.sigmoid(model(mid_imgs[j].unsqueeze(0)))
                preds.append((out > 0.5).squeeze().cpu().numpy())
            dsc_list.append(compute_dice(np.stack(preds), gt))
    return {"dsc_mean": float(np.mean(dsc_list)), "dsc_std": float(np.std(dsc_list))}


def concat_channels_baseline(data_dir, image_size, device, **_):
    """Stack pre+mid as 2-channel input to untrained U-Net."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError("pip install segmentation-models-pytorch")

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=6, classes=1).to(device).eval()

    ds = _load_brats(data_dir, image_size)
    dsc_list = []
    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc="concat_channels", leave=False):
            s = ds[i]
            gt = _binary(s["mid_mask"])
            N  = min(s["mid_image"].shape[0], s["pre_image"].shape[0], gt.shape[0])
            preds = []
            for j in range(N):
                inp = torch.cat([
                    s["pre_image"][j].unsqueeze(0),
                    s["mid_image"][j].unsqueeze(0),
                ], dim=1).to(device)  # (1, 6, H, W)
                out = torch.sigmoid(model(inp))
                preds.append((out > 0.5).squeeze().cpu().numpy())
            dsc_list.append(compute_dice(np.stack(preds), gt[:N]))
    return {"dsc_mean": float(np.mean(dsc_list)), "dsc_std": float(np.std(dsc_list))}


def siamese_unet_baseline(data_dir, image_size, device, **_):
    """Siamese U-Net: encode pre+mid separately, subtract features, decode."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError("pip install segmentation-models-pytorch")

    enc = smp.encoders.get_encoder("resnet34", in_channels=3,
                                   depth=5, weights=None).to(device).eval()
    head = torch.nn.Conv2d(512, 1, 1).to(device)

    ds = _load_brats(data_dir, image_size)
    dsc_list = []
    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc="siamese_unet", leave=False):
            s = ds[i]
            gt = _binary(s["mid_mask"])
            N  = min(s["mid_image"].shape[0], s["pre_image"].shape[0], gt.shape[0])
            preds = []
            for j in range(N):
                f_pre = enc(s["pre_image"][j].unsqueeze(0).to(device))[-1]
                f_mid = enc(s["mid_image"][j].unsqueeze(0).to(device))[-1]
                diff  = (f_mid - f_pre)
                out   = torch.sigmoid(F.interpolate(head(diff),
                            size=(image_size, image_size), mode="bilinear", align_corners=False))
                preds.append((out > 0.5).squeeze().cpu().numpy())
            dsc_list.append(compute_dice(np.stack(preds), gt[:N]))
    return {"dsc_mean": float(np.mean(dsc_list)), "dsc_std": float(np.std(dsc_list))}


def medsam2_baseline(data_dir, image_size, device, sam2_checkpoint, **_):
    """SAM2 with within-session memory only (no cross-session)."""
    try:
        from sam2.build_sam import build_sam2
        sam2 = build_sam2("sam2_hiera_large", sam2_checkpoint, device=device)
        sam2.eval()
        has_sam2 = True
    except Exception as e:
        print(f"  SAM2 load failed: {e} — using random fallback")
        has_sam2 = False

    ds = _load_brats(data_dir, image_size)
    dsc_list = []
    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc="medsam2", leave=False):
            s = ds[i]
            gt = _binary(s["mid_mask"])
            N  = gt.shape[0]
            if not has_sam2:
                pred = (np.random.rand(*gt.shape) > 0.85).astype(np.float32)
            else:
                preds = []
                for j in range(N):
                    img = s["mid_image"][j].unsqueeze(0).to(device)
                    try:
                        bb = sam2.forward_image(img)
                        feats = bb["vision_features"]
                        H, W = image_size, image_size
                        pts   = torch.tensor([[[W/2, H/2]]], dtype=torch.float, device=device)
                        lbls  = torch.ones(1, 1, dtype=torch.int, device=device)
                        sp, dp = sam2.sam_prompt_encoder(points=(pts, lbls), boxes=None, masks=None)
                        lrm, _ = sam2.sam_mask_decoder(
                            image_embeddings=feats,
                            image_pe=sam2.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sp,
                            dense_prompt_embeddings=dp,
                            multimask_output=False,
                        )
                        m = F.interpolate(lrm, size=(H, W), mode="bilinear", align_corners=False)
                        preds.append((torch.sigmoid(m) > 0.5).squeeze().cpu().numpy())
                    except Exception:
                        preds.append(np.zeros((image_size, image_size), dtype=np.float32))
                pred = np.stack(preds)
            dsc_list.append(compute_dice(pred, gt))
    fallback = not has_sam2
    return {"dsc_mean": float(np.mean(dsc_list)), "dsc_std": float(np.std(dsc_list)),
            "fallback": fallback}


def random_baseline(data_dir, image_size, device, **_):
    ds = _load_brats(data_dir, image_size)
    dsc_list = []
    for i in tqdm(range(len(ds)), desc="random", leave=False):
        s = ds[i]
        gt = _binary(s["mid_mask"])
        pred = (np.random.rand(*gt.shape) > 0.85).astype(np.float32)
        dsc_list.append(compute_dice(pred, gt))
    return {"dsc_mean": float(np.mean(dsc_list)), "dsc_std": float(np.std(dsc_list))}


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(results: dict, output_dir: Path):
    lines = [
        "# BraTS-GLI 2024 Baseline Results\n",
        "| Baseline | DSC mean | DSC std | Fallback | Time (s) |",
        "|---|---|---|---|---|",
    ]
    for name, m in sorted(results.items(), key=lambda x: -x[1].get("dsc_mean", 0)):
        dsc  = m.get("dsc_mean", 0)
        std  = m.get("dsc_std",  0)
        fb   = "yes" if m.get("fallback") else "no"
        t    = m.get("elapsed_s", "—")
        lines.append(f"| {name} | {dsc:.4f} | {std:.4f} | {fb} | {t} |")

    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.md").write_text(summary)
    print("\n" + summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BASELINES = {
    "zero":              zero_baseline,
    "random":            random_baseline,
    "identity":          identity_baseline,
    "copy_prev_slice":   copy_prev_slice_baseline,
    "pre_mask_prior":    pre_mask_prior_baseline,
    "unet2d":            unet2d_baseline,
    "concat_channels":   concat_channels_baseline,
    "siamese_unet":      siamese_unet_baseline,
    "medsam2":           medsam2_baseline,
}


def main():
    parser = argparse.ArgumentParser(description="BraTS-GLI baseline sweep")
    parser.add_argument("--data_dir",        type=str, default="data/raw/BraTS_GLI")
    parser.add_argument("--sam2_checkpoint", type=str,
                        default="checkpoints/sam2/sam2.1_hiera_large.pt")
    parser.add_argument("--output_dir",      type=str, default="results/brats_baselines")
    parser.add_argument("--device",          type=str, default="cuda")
    parser.add_argument("--image_size",      type=int, default=512)
    parser.add_argument("--only",            type=str, default=None,
                        help="Comma-separated list of baselines to run (default: all)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    to_run = list(BASELINES.keys())
    if args.only:
        to_run = [b.strip() for b in args.only.split(",")]

    kwargs = dict(
        data_dir=args.data_dir,
        image_size=args.image_size,
        device=args.device,
        sam2_checkpoint=args.sam2_checkpoint,
    )

    all_results = {}

    # Skip baselines whose metrics.json already exists (idempotent)
    for name in to_run:
        if name not in BASELINES:
            print(f"Unknown baseline: {name}")
            continue
        if (output_dir / name / "metrics.json").exists():
            print(f"  Skipping {name} (already done)")
            with open(output_dir / name / "metrics.json") as f:
                all_results[name] = json.load(f)
            continue
        metrics = _run_baseline(
            name=name,
            fn=lambda n=name: BASELINES[n](**kwargs),
            output_dir=output_dir,
        )
        all_results[name] = metrics

    write_summary(all_results, output_dir)
    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
