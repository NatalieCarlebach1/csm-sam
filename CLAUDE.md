# CSM-SAM: Cross-Session Memory SAM

## Project Overview
This repository implements **CSM-SAM** — Cross-Session Memory SAM for adaptive radiotherapy mid-treatment tumor segmentation. It extends SAM2's within-session memory bank to **cross-session temporal propagation**: given pre-RT (week 0) MRI, propagate a memory bank to guide segmentation of the mid-RT (week 2-3) scan.

**Target venue**: NeurIPS 2026  
**Benchmark**: HNTS-MRG 2024 Task 2 (mid-RT head & neck tumor segmentation)  
**Baseline to beat**: 72.7 aggDSC (challenge winner)  
**Target**: >75 aggDSC

## Architecture
- **Frozen**: SAM2 image encoder (ViT-H) + mask decoder (except final layer)
- **Trainable (~2M params)**: `CrossSessionMemoryAttention` + `ChangeHead`
- **Key module**: `csmsam/modeling/cross_session_memory_attention.py`
- **Loss**: Dice + 0.3 × CE(change_map) — change map labels auto-derived from pre/mid mask XOR

## Repository Structure
```
csm-sam/
├── csmsam/
│   ├── modeling/
│   │   ├── cross_session_memory_attention.py   # Core novel module
│   │   ├── csm_sam.py                          # Full model wrapper
│   │   └── change_head.py                      # Change map prediction
│   ├── datasets/
│   │   └── hnts_mrg.py                         # HNTS-MRG 2024 dataset
│   ├── losses/
│   │   └── combined_loss.py                    # Dice + CE + change map
│   └── utils/
│       ├── metrics.py                          # aggDSC, Dice, HD95
│       └── visualization.py                    # Overlay + change map plots
├── data/
│   ├── download_hnts_mrg.py                    # Zenodo downloader
│   └── preprocess.py                           # Registration + normalization
├── baselines/
│   ├── medsam2_baseline.py                     # MedSAM2 (no cross-session)
│   ├── nnunet_baseline.py                      # nnUNet runner
│   └── run_all_baselines.py
├── train.py                                    # Main training script
├── test.py                                     # Evaluation + table output
├── visualize.py                                # Random test set visualization
└── configs/
    ├── default.yaml
    └── baselines.yaml
```

## Quick Start
```bash
# 1. Install
pip install -e ".[dev]"

# 2. Download HNTS-MRG 2024 data
python data/download_hnts_mrg.py --output_dir data/raw

# 3. Preprocess
python data/preprocess.py --input_dir data/raw --output_dir data/processed

# 4. Train
python train.py --config configs/default.yaml

# 5. Evaluate
python test.py --checkpoint checkpoints/best.pth --output_dir results/

# 6. Visualize random test samples
python visualize.py --checkpoint checkpoints/best.pth --n_samples 10

# 7. Run all baselines
python baselines/run_all_baselines.py
```

## Key Design Decisions
- **Why SAM2 not nnUNet?** 150 training patients is small; SAM2's frozen ViT-H pretrained on 1B masks provides strong priors. We only train ~2M params.
- **Why cross-session not within-session?** Within-session memory (MedSAM2) propagates slice→slice in one 3D scan. We propagate visit→visit (week 0 → week 2-3), which is structurally novel.
- **Why change map?** Provides free supervision signal — automatically derived from pre/mid mask XOR. Forces the model to be spatially aware of tumor dynamics.
- **Why temporal embedding?** Treatment response varies with weeks elapsed (1-4 weeks). Embedding the elapsed time lets the model calibrate expected change magnitude.

## Data Access
HNTS-MRG 2024 is available on Zenodo (DOI: 10.5281/zenodo.11829006) and via Grand Challenge. Registration required. See `data/download_hnts_mrg.py` for instructions.

## Metrics
- **aggDSC**: Primary metric. `(DSC_GTVp + DSC_GTVn) / 2`
- **HD95**: Secondary. 95th percentile Hausdorff distance.
- **Change map IoU**: Ablation metric for change prediction head.
