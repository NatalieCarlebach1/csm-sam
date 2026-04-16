"""
Concat-Channels Baseline — naive 2-channel (pre-RT, mid-RT) 2D U-Net.

The simplest way to "use" the pre-RT information: stack the pre-RT and mid-RT
slices together as a 2-channel input to a 2D encoder-decoder. The network is
handed both visits but has to learn all temporal correspondence from scratch
with no inductive bias.

Backbone: segmentation_models_pytorch U-Net with an ImageNet-pretrained
encoder. The first conv is re-initialized for 2 input channels (pre, mid).

Reference:
    Ronneberger, Fischer, Brox. "U-Net: Convolutional Networks for Biomedical
    Image Segmentation." MICCAI 2015.

Uniqueness vs CSM-SAM:
    CSM-SAM propagates the PRE-RT MEMORY BANK (per-slice key/value tokens
    from the frozen SAM2 encoder) to the mid-RT stream via learned
    cross-session attention, preserving fine-grained per-region
    correspondence and riding on SAM2's 1B-mask prior. This baseline merely
    stacks raw pre/mid pixels into the same 2D tensor: there is no attention,
    no per-voxel key/value correspondence, and no SAM2 prior. CSM-SAM
    additionally supervises a change head on the pre/mid MASK XOR and
    embeds weeks_elapsed — two inductive biases this baseline does not have.

Usage:
    python baselines/concat_channels_baseline.py \
        --data_dir data/processed \
        --output_dir results/baselines/concat_channels \
        --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from csmsam.datasets import HNTSMRGDataset
from csmsam.utils.metrics import aggregate_metrics, evaluate_patient

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False


def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """(N, 3, H, W) SAM2-normalized RGB --> (N, 1, H, W) grayscale."""
    return x.mean(dim=1, keepdim=True)


class ConcatChannelsBaseline:
    """
    2D U-Net operating on a 2-channel (pre-RT, mid-RT) stack.

    The pre-RT channel is simply the raw normalized pre-RT slice at the same
    index as the mid-RT slice. No registration, no attention, no memory.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        num_classes: int = 2,
        checkpoint: str | None = None,
        device: str = "cuda",
    ):
        self.device = device
        self.num_classes = num_classes
        self.model = None

        if not HAS_SMP:
            print("Warning: segmentation_models_pytorch not installed.")
            print("Install with: pip install segmentation-models-pytorch")
            print("Falling back to random predictions.")
            return

        # 2 input channels: [pre_gray, mid_gray]
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=2,
            classes=num_classes,
        ).to(device)

        if checkpoint is not None and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location=device)
            state = state.get("model", state)
            self.model.load_state_dict(state, strict=False)
            print(f"Loaded checkpoint: {checkpoint}")
        else:
            print("No checkpoint provided; using ImageNet-initialized encoder only.")

        self.model.eval()

    @torch.no_grad()
    def predict_volume(
        self,
        pre_images: torch.Tensor,
        mid_images: torch.Tensor,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Slice-wise segmentation of mid-RT from a 2-channel (pre, mid) stack.

        Args:
            pre_images : (N_pre, 3, H, W) pre-RT slices (SAM2 tensor format)
            mid_images : (N_mid, 3, H, W) mid-RT slices
        Returns:
            pred_binary : (N_mid, H, W) foreground prediction
        """
        N_mid, _, H, W = mid_images.shape

        if self.model is None:
            return (np.random.rand(N_mid, H, W) > 0.9).astype(np.float32)

        # Same-index alignment: clamp pre index to its volume length.
        N_pre = pre_images.shape[0]
        pre_gray = _rgb_to_gray(pre_images)   # (N_pre, 1, H, W)
        mid_gray = _rgb_to_gray(mid_images)   # (N_mid, 1, H, W)

        pred_slices = []
        for i in range(N_mid):
            j = min(i, N_pre - 1)
            pair = torch.cat([pre_gray[j:j + 1], mid_gray[i:i + 1]], dim=1)  # (1, 2, H, W)
            pair = pair.to(self.device)

            logits = self.model(pair)  # (1, C, H, W)
            if self.num_classes == 1:
                mask = (torch.sigmoid(logits) > threshold).squeeze().cpu().numpy()
            else:
                probs = F.softmax(logits, dim=1)
                mask = (probs[:, 1:].sum(dim=1) > threshold).squeeze().cpu().numpy()
            pred_slices.append(mask.astype(np.float32))

        return np.stack(pred_slices)


def run_concat_channels_baseline(
    data_dir: str,
    output_dir: str,
    split: str = "test",
    checkpoint: str | None = None,
    device: str = "cuda",
    threshold: float = 0.5,
    image_size: int = 512,
    encoder_name: str = "resnet34",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Concat-Channels Baseline  (2-channel [pre, mid] 2D U-Net)")
    print(f"Encoder: {encoder_name} | Split: {split}")
    print("=" * 60)

    model = ConcatChannelsBaseline(
        encoder_name=encoder_name,
        checkpoint=checkpoint,
        device=device,
    )

    dataset = HNTSMRGDataset(data_dir=data_dir, split=split, image_size=image_size)
    per_patient_metrics = []

    for idx in tqdm(range(len(dataset)), desc="Patients"):
        patient_data = dataset[idx]
        pid = patient_data["patient_id"]

        try:
            pre_images = patient_data["pre_images"]  # (N_pre, 3, H, W)
            mid_images = patient_data["mid_images"]  # (N_mid, 3, H, W)
            pred_binary = model.predict_volume(pre_images, mid_images, threshold)

            gt_gtvp = (patient_data["mid_masks_gtvp"] > 0.5).squeeze(1).numpy()
            gt_gtvn = (patient_data["mid_masks_gtvn"] > 0.5).squeeze(1).numpy()

            metrics = evaluate_patient(
                pred_masks=pred_binary,
                pred_gtvp=pred_binary,
                pred_gtvn=pred_binary,
                target_gtvp=gt_gtvp,
                target_gtvn=gt_gtvn,
            )
            metrics["patient_id"] = pid
            per_patient_metrics.append(metrics)

        except Exception as e:
            print(f"  Error on {pid}: {e}")

    agg = aggregate_metrics(per_patient_metrics)

    print(f"\nConcat-Channels Baseline Results ({split}):")
    print(f"  aggDSC  : {agg.get('agg_dsc_mean', 0):.4f} ± {agg.get('agg_dsc_std', 0):.4f}")
    print(f"  GTVp DSC: {agg.get('dsc_gtvp_mean', 0):.4f}")
    print(f"  GTVn DSC: {agg.get('dsc_gtvn_mean', 0):.4f}")

    results = {
        "aggregate": agg,
        "per_patient": per_patient_metrics,
        "config": {
            "baseline": "concat_channels",
            "encoder_name": encoder_name,
            "in_channels": 2,
            "image_size": image_size,
            "threshold": threshold,
            "split": split,
        },
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {output_dir}/metrics.json")
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concat-Channels (pre, mid) 2D U-Net baseline.")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/baselines/concat_channels")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--encoder_name", type=str, default="resnet34")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_concat_channels_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        checkpoint=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        image_size=args.image_size,
        encoder_name=args.encoder_name,
    )
