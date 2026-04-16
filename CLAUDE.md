# RALM-SAM: Retrieval-Augmented Longitudinal Memory SAM

## Project Overview
This repository implements **RALM-SAM** — a retrieval-augmented longitudinal memory extension of SAM2 for adaptive radiotherapy mid-treatment tumor segmentation. It reframes mid-RT segmentation as a longitudinal distribution-shift problem, propagating a cross-session memory bank from pre-RT (week 0) to mid-RT (week 2-3), augmenting it with cross-patient change templates retrieved from a training-set bank, and conditioning on continuous weeks-elapsed. **CSM-SAM** (cross-session memory only, discrete temporal embed, no retrieval) is retained as the primary ablation variant.

**Target venue**: NeurIPS 2026  
**Benchmark**: HNTS-MRG 2024 Task 2 (mid-RT head & neck tumor segmentation)  
**Baseline to beat**: 72.7 aggDSC (challenge winner)  
**Target**: >75 aggDSC

## Architecture
- **Frozen**: SAM2 image encoder (ViT-H) + most of the mask decoder (final layer trainable).
- **Trainable (~2M params total)**:
  - `CrossSessionMemoryAttention` — cross-session (attn to M_aug) + within-session (attn to M_mid) + learned gate.
  - Continuous-time encoder — sinusoidal + MLP over weeks-elapsed (discrete `nn.Embedding` retained as ablation via `temporal_encoder_type: discrete`).
  - `CrossPatientRetrieval` + `CrossPatientBank` — top-K retrieval of pre-RT neighbors; injects their tumor-localized `(mid - pre)` change templates as extra tokens onto M_pre.
  - `ChangeHead` — 3-class change map (stable/grown/shrunk), XOR-supervised.
  - `FeatureEvolutionPredictor` (~200k params) — predicts mid-RT features from pre-RT + t; drives the consistency loss.
  - Dual-head decoder — per-structure GTVp / GTVn via pre-RT mask prompting of SAM2.
- **Training mode**: sequence training accumulates M_mid across consecutive mid-RT slices so the within-session attention and gate actually learn.
- **Loss**: `L_dice + 0.3 * CE(change_map) + 0.2 * L_consistency`, where `L_consistency = || g(pre_features, t) - mid_features ||^2` on tumor regions.

## Repository Structure
```
csm-sam/
├── csmsam/
│   ├── modeling/
│   │   ├── cross_session_memory_attention.py   # Core cross/within-session attention + gate
│   │   ├── retrieval.py                        # CrossPatientBank, CrossPatientRetrieval
│   │   ├── csm_sam.py                          # Full RALM-SAM wrapper (dual-head decoder)
│   │   └── change_head.py                      # Change map prediction
│   ├── datasets/                               # 8 dataset loaders, uniform __getitem__ contract
│   │   ├── hnts_mrg.py                         # HNTS-MRG 2024 (primary; pre-RT + mid-RT)
│   │   ├── brats_gli.py                        # BraTS-GLI 2024 post-treatment (longitudinal pairs)
│   │   ├── oaizib_cm.py                        # OAIZIB-CM knee MRI (single-timepoint, pretraining)
│   │   ├── ms_segmentation.py                  # MS brain lesion (single-timepoint, pretraining)
│   │   ├── levir_cd.py                         # LEVIR-CD binary change detection (transfer)
│   │   ├── s2looking.py                        # S2Looking off-nadir binary CD (transfer)
│   │   ├── second.py                           # SECOND semantic change detection (transfer)
│   │   └── xbd.py                              # xBD building damage (transfer)
│   ├── losses/
│   │   ├── combined_loss.py                    # Dice + CE(change) + consistency
│   │   └── consistency.py                      # FeatureEvolutionPredictor + loss
│   └── utils/
│       ├── metrics.py                          # aggDSC, Dice, HD95
│       ├── visualization.py                    # Overlays + change maps + retrieved donors
│       ├── cv.py                               # K-fold CV split utilities
│       └── tta.py                              # Test-time augmentation
├── data/
│   ├── download_hnts_mrg.py                    # Zenodo downloader (+ synthetic data generator)
│   ├── preprocess.py                           # Registration + normalization
│   └── raw/                                    # ~75 GB across 8 datasets (gitignored)
├── baselines/                                  # 36 baselines covering CSM-SAM's claim landscape
│   ├── LANDSCAPE.md                            # 6-axis positioning + per-method comparison table
│   ├── SOTA.md                                 # Current SOTA per dataset + per-method blurbs
│   ├── run_all_baselines.py                    # Python orchestrator (depends on trained CSM-SAM)
│   ├── medsam2_baseline.py                     # MedSAM2 (within-session only)
│   ├── nnunet_baseline.py / nnunet_dual_channel / hnts_winner_replica  # nnUNet family
│   ├── sam_vanilla / sam2_point_prompt / sam2_video / dinov2_linear / clipseg / totalsegmentator  # foundation models
│   ├── unet_2d / deeplabv3plus / swinunetr / unetr / mednext           # classical medical seg
│   ├── concat_channels / siamese_unet / pre_mask_prior / longisam      # longitudinal ablations
│   ├── identity / zero / random / copy_prev_slice / majority_voxel     # naive/sanity baselines
│   ├── fc_siam_diff / fc_siam_conc / snunet / bit / changeformer / tinycd  # binary CD
│   ├── bisrnet / scannet_cd / ced_net                                  # semantic CD (SECOND)
│   └── xview2_dualhrnet                                                # xBD damage CD
├── scripts/
│   ├── run_all_baselines.sh                    # End-to-end sweep (produces results/baselines/summary.md)
│   └── build_bank.py                           # (forthcoming) build CrossPatientBank over train split
├── paper/                                      # NeurIPS 2026 draft (LaTeX scaffold)
│   ├── main.tex / references.bib / Makefile
│   ├── sections/                               # intro / related / method / results / ablations / discussion / conclusion
│   └── tables/                                 # per-dataset booktabs tables
├── train.py                                    # Main training script (sequence mode + CV)
├── test.py                                     # Evaluation + table output (TTA + retrieval)
├── visualize.py                                # Random test set visualization
└── configs/
    ├── default.yaml
    └── baselines.yaml
```

## Datasets
8 datasets downloaded under `data/raw/` (~75 GB total):

| Dataset | Role | Size | Source |
|---|---|---|---|
| HNTS-MRG 2024 | Primary (longitudinal pre-RT → mid-RT) | 14 GB | Zenodo 11199559 |
| BraTS-GLI 2024 | Secondary (post-treatment glioma pairs) | 17 GB | HF: ClarkQuinn/BraTS_GLI_PRE |
| OAIZIB-CM | Pretraining (single-timepoint knee) | 12 GB | HF: YongchengYAO/OAIZIB-CM |
| MS-Segmentation | Pretraining (cross-sectional MS lesion) | 400 MB | HF: alexwang05/MS-Segmentation |
| LEVIR-CD | Transfer (binary CD) | 5.9 GB | HF: satellite-image-deep-learning/LEVIR-CD |
| S2Looking | Transfer (off-nadir binary CD) | 11 GB | HF: EVER-Z/torchange_s2looking |
| SECOND | Transfer (semantic CD, closest CSM-SAM analog) | 3.6 GB | HF: EVER-Z/torchange_second |
| xBD | Transfer (building damage) | 11 GB | HF: aryananand/xBD |

Still gated (manual registration required): true longitudinal OAI (NIH NDA 72-mo follow-up), ISBI 2015 longitudinal MS.

## Quick Start
```bash
# 1. Install
pip install -e ".[dev]"

# 2. Download HNTS-MRG 2024 data
python data/download_hnts_mrg.py --output_dir data/raw

# 3. Preprocess
python data/preprocess.py --input_dir data/raw --output_dir data/processed

# 4. Train (sequence mode, fold 0 of 5)
python train.py --config configs/default.yaml --sequence_train --fold 0 --n_folds 5

# 5. Build cross-patient bank over train split (scripts/build_bank.py is forthcoming)
python scripts/build_bank.py --checkpoint checkpoints/best.pth --output bank.pt

# 6. Evaluate (with retrieval + TTA)
python test.py --checkpoint checkpoints/best.pth --bank bank.pt --tta --output_dir results/

# 7. Visualize random test samples
python visualize.py --checkpoint checkpoints/best.pth --bank bank.pt --n_samples 10

# 8. Run all baselines (sweep script — produces results/baselines/summary.md)
bash scripts/run_all_baselines.sh
```

## Baseline sweep
`scripts/run_all_baselines.sh` runs all 36 baselines with a 300 s timeout each, writes per-baseline `metrics.json` + `status.txt` + `stdout.log` under `results/baselines/<name>/`, and aggregates into a dataset-grouped `summary.md`. It is idempotent (skips baselines whose `metrics.json` already exists). Baselines whose pretrained weights or training pipeline aren't available emit random-prediction fallbacks marked with `fallback=true` in the summary.

Dataset dispatch:
- Medical (HNTS-MRG synthetic): naive / classical / foundation-model / longitudinal baselines
- LEVIR-CD: fc_siam_diff, fc_siam_conc, snunet, bit, changeformer, tinycd
- SECOND: ced_net, bisrnet, scannet_cd
- xBD: xview2_dualhrnet

Override device with `DEVICE=cpu bash scripts/run_all_baselines.sh`; default is `cuda`.

## Key Design Decisions
- **Why SAM2 not nnUNet?** 150 training patients is small; SAM2's frozen ViT-H pretrained on 1B masks provides strong priors. We only train ~2M params.
- **Why cross-session not within-session?** Within-session memory (MedSAM2) propagates slice→slice inside one 3D scan. We propagate visit→visit (week 0 → week 2-3), which is structurally novel.
- **Why retrieval?** Each patient has only 2 time points, so per-patient trajectory learning is data-starved. Cross-patient retrieval pools evolution signals across the training set at inference, turning CSM-SAM into an in-context longitudinal segmenter.
- **Why continuous-time conditioning?** Weeks-elapsed is a continuous covariate (1–4+ weeks, real-valued); a discrete embedding wastes capacity and fails to extrapolate. A sinusoidal+MLP `f(t)` generalizes across arbitrary intervals.
- **Why feature-evolution consistency?** Forces the backbone to represent *how* tumors evolve, not just *where* they are — a self-supervised temporal regularizer with no extra labels.
- **Why change map?** Provides free supervision from pre/mid mask XOR; makes the model spatially aware of tumor dynamics.
- **Why dual-head decoder?** GTVp and GTVn have very different appearances and priors; per-structure prompting via the pre-RT mask beats a single-argmax head.
- **Why sequence-mode training?** Without accumulating M_mid across slices, the within-session attention receives an empty bank and the gate collapses to using only cross-session context.

## Data Access
HNTS-MRG 2024 is available on Zenodo (DOI: 10.5281/zenodo.11199559) and via Grand Challenge. Registration required. See `data/download_hnts_mrg.py` for instructions.

## Metrics
- **aggDSC**: Primary metric. `(DSC_GTVp + DSC_GTVn) / 2`.
- **HD95**: Secondary. 95th percentile Hausdorff distance.
- **Change map IoU**: Ablation metric for the change prediction head.
- **Feature consistency MSE**: Ablation metric for the evolution predictor.

## Reference documents
- `baselines/LANDSCAPE.md` — 6-axis positioning system (temporal scope / pre-signal / backbone / params / change head / temporal embed) with one row per baseline and CSM-SAM placed in the top-right "strongest-backbone × richest-pre-session-mechanism" quadrant.
- `baselines/SOTA.md` — current state-of-the-art numbers per dataset, plus method blurbs (what each SOTA method does + why it's different from CSM-SAM).
- `paper/` — NeurIPS 2026 LaTeX scaffold with populated results tables (SOTA rows from published papers, `CSM-SAM (Ours)` placeholder rows awaiting training).
