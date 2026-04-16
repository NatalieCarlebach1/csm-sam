"""
DeepLabV3+ Baseline — atrous-conv slice-wise segmentation via segmentation_models_pytorch.

DeepLabV3+ with a ResNet-50 encoder applied slice-by-slice to mid-RT volumes.
Combines atrous spatial pyramid pooling with an encoder-decoder refinement stage.

Paper: Chen, Zhu, Papandreou, Schroff, Adam. "Encoder-Decoder with Atrous
Separable Convolution for Semantic Image Segmentation." ECCV 2018.

Install:
    pip install segmentation-models-pytorch

Uniqueness vs CSM-SAM:
    CSM-SAM uses a cross-session memory attention module that conditions mid-RT
    segmentation on features and masks from the pre-RT scan; this DeepLabV3+
    baseline has no temporal/cross-session mechanism and predicts each mid-RT
    slice independently. CSM-SAM's backbone is SAM2's frozen ViT-H pretrained on
    1B masks, whereas DeepLabV3+ uses a ~40M-parameter ResNet-50 backbone with
    ImageNet weights that must be fine-tuned on only ~150 patients.

Usage:
    python baselines/deeplabv3plus_baseline.py \
        --data_dir data/processed \
        --output_dir results/baselines/deeplabv3plus \
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
from csmsam.utils.metrics import evaluate_patient, aggregate_metrics

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False


class DeepLabV3PlusBaseline:
    """
    DeepLabV3+ (ResNet-50 encoder) applied slice-by-slice to the mid-RT volume.

    Falls back to random predictions if segmentation_models_pytorch is missing.
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
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

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
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
        mid_images: torch.Tensor,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Run DeepLabV3+ on every mid-RT slice.

        Args:
            mid_images : (N, 3, H, W) mid-RT slices.

        Returns:
            pred_binary : (N, H, W) binary foreground prediction.
        """
        N, _, H, W = mid_images.shape

        if self.model is None:
            return (np.random.rand(N, H, W) > 0.9).astype(np.float32)

        pred_slices = []
        for i in range(N):
            img = mid_images[i].unsqueeze(0).to(self.device)
            logits = self.model(img)
            if self.num_classes == 1:
                probs = torch.sigmoid(logits)
                mask = (probs > threshold).squeeze().cpu().numpy()
            else:
                probs = F.softmax(logits, dim=1)
                mask = (probs[:, 1:].sum(dim=1) > threshold).squeeze().cpu().numpy()
            pred_slices.append(mask.astype(np.float32))

        return np.stack(pred_slices)


def run_deeplabv3plus_baseline(
    data_dir: str,
    output_dir: str,
    split: str = "test",
    checkpoint: str | None = None,
    device: str = "cuda",
    threshold: float = 0.5,
    image_size: int = 512,
    encoder_name: str = "resnet50",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DeepLabV3+ Baseline (mid-RT only, no cross-session)")
    print(f"Encoder: {encoder_name} | Split: {split}")
    print("=" * 60)

    model = DeepLabV3PlusBaseline(
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
            mid_images = patient_data["mid_images"]
            pred_binary = model.predict_volume(mid_images, threshold)

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

    print(f"\nDeepLabV3+ Baseline Results ({split}):")
    print(f"  aggDSC  : {agg.get('agg_dsc_mean', 0):.4f} ± {agg.get('agg_dsc_std', 0):.4f}")
    print(f"  GTVp DSC: {agg.get('dsc_gtvp_mean', 0):.4f}")
    print(f"  GTVn DSC: {agg.get('dsc_gtvn_mean', 0):.4f}")

    results = {"aggregate": agg, "per_patient": per_patient_metrics}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {output_dir}/metrics.json")
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepLabV3+ baseline for HNTS-MRG 2024")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/baselines/deeplabv3plus")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--encoder_name", type=str, default="resnet50")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_deeplabv3plus_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        checkpoint=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        image_size=args.image_size,
        encoder_name=args.encoder_name,
    )
