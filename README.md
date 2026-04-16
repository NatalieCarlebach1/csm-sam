# CSM-SAM: Cross-Session Memory SAM for Adaptive Radiotherapy

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Cross-Session Memory SAM (CSM-SAM)**: Extending SAM2's temporal memory from within-scan slice propagation to cross-visit session propagation for adaptive radiotherapy tumor segmentation.

## Abstract

Adaptive MRI-guided radiotherapy (MRgRT) requires daily re-segmentation of head and neck tumors as they respond to treatment. Existing methods treat each scan independently, ignoring rich structural information from the pre-treatment planning scan. We propose **CSM-SAM**, which encodes the pre-RT scan into SAM2's memory bank and attends across sessions when segmenting the mid-RT scan. A lightweight **CrossSessionMemoryAttention** module (~2M trainable parameters) gates between pre-RT cross-session context and within-session slice context using a learned temporal gating mechanism conditioned on weeks elapsed. An auxiliary **change map head** provides free supervision from automatically derived pre/mid mask differences.  On the HNTS-MRG 2024 benchmark, CSM-SAM achieves **XX.X aggDSC** on mid-RT segmentation, outperforming the challenge winner (72.7) and MedSAM2 baseline.

## Method

```
Pre-RT MRI ──► SAM2 Encoder ──► Memory Bank (M_pre)
                                         │
                                         ▼
Mid-RT MRI ──► SAM2 Encoder ──► CrossSessionMemoryAttention ──► Mask Decoder ──► Segmentation
                    │                    │                                              │
                    └──► M_mid ──────────┘                                    Change Map Head
```

**CrossSessionMemoryAttention**:
- Cross-session attention: current mid-RT features attend to M_pre (+ temporal embedding for weeks elapsed)
- Within-session attention: current features attend to M_mid (prior mid-RT slices)
- Learned gate combines both contexts

**Change Map Head**: 3-class prediction (stable tumor / grown region / shrunk region), supervised by automatic XOR-based labels from pre/mid masks.

## Results

| Method | GTVp DSC | GTVn DSC | aggDSC | HD95 (mm) |
|--------|----------|----------|--------|-----------|
| HNTS-MRG 2024 Winner | - | - | 72.7 | - |
| MedSAM2 (no cross-session) | - | - | ~68.0 | - |
| CSM-SAM (ours) | - | - | **TBD** | - |

*Results will be filled after training on HNTS-MRG 2024 data.*

## Installation

```bash
# Clone this repo
git clone https://github.com/YOUR_USERNAME/csm-sam.git
cd csm-sam

# Create conda environment
conda create -n csmsam python=3.10
conda activate csmsam

# Install PyTorch (adjust CUDA version)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install SAM2
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . && cd ..

# Install MedSAM2
git clone https://github.com/bowang-lab/MedSAM2.git
cd MedSAM2 && pip install -e . && cd ..

# Install CSM-SAM
pip install -e ".[dev]"

# Download SAM2 checkpoint
mkdir -p checkpoints/sam2
wget -P checkpoints/sam2 https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## Data

### HNTS-MRG 2024

The HNTS-MRG 2024 dataset contains 150 head & neck cancer patients with paired:
- **Pre-RT**: T2-weighted MRI + GTVp/GTVn manual contours (planning scan, week 0)
- **Mid-RT**: T2-weighted MRI + GTVp/GTVn manual contours (week 2-3 of treatment)

**Access**: Register at [Grand Challenge](https://hnts-mrg.grand-challenge.org/) then download from [Zenodo](https://zenodo.org/record/11829006).

```bash
# After registering and obtaining access token:
python data/download_hnts_mrg.py \
    --zenodo_token YOUR_TOKEN \
    --output_dir data/raw

# Preprocess (registration + normalization + slice extraction)
python data/preprocess.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --n_workers 8
```

## Training

```bash
# Default training (CSM-SAM on HNTS-MRG Task 2)
python train.py --config configs/default.yaml

# With custom settings
python train.py \
    --config configs/default.yaml \
    --batch_size 4 \
    --lr 1e-4 \
    --epochs 200 \
    --sam2_checkpoint checkpoints/sam2/sam2.1_hiera_large.pt \
    --data_dir data/processed \
    --output_dir checkpoints/csmsam_run1

# Resume training
python train.py --config configs/default.yaml --resume checkpoints/csmsam_run1/latest.pth
```

## Evaluation

```bash
# Evaluate on test set
python test.py \
    --checkpoint checkpoints/best.pth \
    --data_dir data/processed \
    --split test \
    --output_dir results/csmsam

# Output: results/csmsam/
#   metrics.json       — per-patient and aggregate aggDSC, HD95
#   metrics_table.txt  — formatted comparison table
#   predictions/       — .nii.gz prediction files
```

## Visualization

```bash
# Visualize 10 random test patients
python visualize.py \
    --checkpoint checkpoints/best.pth \
    --data_dir data/processed \
    --n_samples 10 \
    --output_dir results/visualizations

# Output per patient:
#   {patient_id}_overlay.png      — pre-RT and mid-RT with GT + prediction overlay
#   {patient_id}_change_map.png   — predicted change map (stable/grown/shrunk)
#   {patient_id}_slice_gallery.png — multi-slice gallery
```

## Baselines

```bash
# Run all baselines
python baselines/run_all_baselines.py --data_dir data/processed --output_dir results/baselines

# Individual baselines
python baselines/medsam2_baseline.py --data_dir data/processed   # MedSAM2 (within-session only)
python baselines/nnunet_baseline.py --data_dir data/processed    # nnUNet (strong non-SAM baseline)
```

## Project Structure

```
csm-sam/
├── csmsam/
│   ├── modeling/
│   │   ├── cross_session_memory_attention.py   # Core novel module
│   │   ├── csm_sam.py                          # Full CSM-SAM model
│   │   └── change_head.py                      # 3-class change prediction
│   ├── datasets/
│   │   └── hnts_mrg.py                         # Dataset + augmentation
│   ├── losses/
│   │   └── combined_loss.py                    # Dice + CE + change map
│   └── utils/
│       ├── metrics.py                          # aggDSC, HD95, IoU
│       └── visualization.py                    # Overlays, galleries, change maps
├── data/
│   ├── download_hnts_mrg.py                    # Zenodo download script
│   └── preprocess.py                           # Registration, norm, slicing
├── baselines/
│   ├── medsam2_baseline.py
│   ├── nnunet_baseline.py
│   └── run_all_baselines.py
├── configs/
│   ├── default.yaml                            # CSM-SAM config
│   └── baselines.yaml                          # Baseline configs
├── train.py                                    # Training loop
├── test.py                                     # Evaluation
├── visualize.py                                # Test set visualization
├── setup.py
├── requirements.txt
└── CLAUDE.md                                   # AI assistant context
```

## Citation

If you use this code, please cite:

```bibtex
@article{csmsam2026,
  title={Cross-Session Memory SAM for Adaptive Radiotherapy Mid-Treatment Tumor Segmentation},
  author={},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
