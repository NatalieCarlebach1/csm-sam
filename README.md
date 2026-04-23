# RALM-SAM: Retrieval-Augmented Longitudinal Memory SAM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **RALM-SAM**: A retrieval-augmented longitudinal memory framework for adaptive radiotherapy tumor segmentation. Reframes mid-treatment segmentation as a longitudinal distribution-shift problem addressed by (i) cross-session memory, (ii) cross-patient retrieval, and (iii) continuous-time conditioning.

## Abstract

Adaptive MRI-guided radiotherapy (MRgRT) requires re-segmentation of head and neck tumors during a treatment course as they respond — shrinking, growing, or migrating over weeks. Treating each scan independently ignores the rich structural prior from the pre-treatment planning scan, while naive per-patient temporal modeling is data-starved: each patient contributes only two time points.

We recast mid-treatment segmentation as a **longitudinal distribution-shift** problem and propose **RALM-SAM**, a retrieval-augmented longitudinal memory extension of SAM2. RALM-SAM (1) encodes the pre-RT scan into a cross-session memory bank M_pre and attends across visits when segmenting the mid-RT scan; (2) retrieves the top-K most similar pre-RT patients from a **cross-patient bank** built on the training set, concatenating their tumor-localized "change templates" onto M_pre so that longitudinal evolution can be borrowed in-context from similar cases; (3) replaces the discrete week embedding with a **continuous-time encoder** f(t) over weeks elapsed; and (4) is regularized by a **feature-evolution consistency loss** that trains a light auxiliary predictor g(pre_features, t) to forecast mid-RT features on tumor regions. A dual-head decoder produces per-structure GTVp and GTVn masks via pre-RT mask prompting of SAM2, and training uses a sequence mode that accumulates within-session memory across consecutive slices. On the HNTS-MRG 2024 benchmark, RALM-SAM targets **>75 aggDSC**, exceeding the challenge winner (72.7) and MedSAM2 baselines.

## Key Novelties

- **Cross-patient retrieval memory.** A `CrossPatientBank` stores pre-RT summaries and tumor-localized `(mid - pre)` change templates for every training patient. At inference, top-K similar patients are retrieved and their templates are injected as additional tokens onto M_pre, turning longitudinal segmentation into an in-context retrieval task.
- **Continuous-time temporal conditioning.** A sinusoidal + MLP encoder `f(t)` over continuous weeks-elapsed replaces the discrete `nn.Embedding(weeks)`. The discrete variant is retained as an ablation.
- **Feature-evolution consistency loss.** A small `FeatureEvolutionPredictor` (~200k params) is trained so that `g(pre_features, t) ~ mid_features` on tumor regions, forcing the backbone to represent *how* tumors evolve, not just *where* they are.
- **Dual-head decoder with pre-RT mask prompting.** SAM2 is prompted separately for GTVp and GTVn using the pre-RT mask as a spatial prior, giving per-structure probability maps instead of a single argmax.
- **Sequence-mode training.** Within-session memory M_mid is accumulated across consecutive mid-RT slices during training, so the within-session attention and temporal gate actually learn instead of collapsing to a no-op.

## Method

```
Pre-RT  ─► Encoder ─► M_pre ─┐
                              ├─► M_aug ─► CrossSessionAttn ─► enhanced_feats ─┐
Bank ─ top-K retrieval ─►  ──┘                │                                │
                                              ├─► ChangeHead   (aux)           │
                                              └─► EvolutionPred → L_cons       ▼
                                                                  SAM2 Mask Decoder ─► (GTVp, GTVn)
Mid-RT  ─► Encoder ─► mid_feats                                         ▲
                                                              dense_prompt
                          pre-RT mask ─► Bridge MaskEncoder ─► cross-attn ─┘
                                         (galash path, encoder_type=dino_sam)
```

**Encoder (`model.encoder_type`).**
- `sam2` — original: frozen SAM2.1 Hiera (Base+ or Large), ~2.9 M trainable (attention, memory encoder, change head, retrieval) + ~4 M SAM2 mask-decoder at `lr×0.1`.
- `dino_sam` — new, default on `tals-local` branch: **frozen DINOv2** (`facebook/dinov2-base` at 518 → 37×37 patch grid) + **trainable Bridge (~13.9 M)** that fuses 4 equally-spaced DINO hidden-states into the SAM2 decoder's expected feature pyramid. Ported from a prior change-detection pipeline. The claim — *frozen DINOv2 beats SAM2-Hiera for medical image features* — is supported by a separate 11-backbone KiTS19 ablation where DINOv2-Base @ 518 wins tumor Dice by +6 pts over SAM2-Hiera.

**Mask handling differs by encoder path:**
- `sam2`: pre-RT mask → frozen SAM2 prompt_encoder → `dense_embed`; decoder called with `multimask_output=True` (argmax over 3 candidates), `lambda_iou=0.1`.
- `dino_sam`: pre-RT mask → **Bridge `MaskEncoder` + learned 2D pos-emb → TransformerDecoderLayer(mask_tokens ↔ image_emb)** → `dense_prompt`. Decoder called with `multimask_output=False` (1 mask per structure), `lambda_iou=0.0`. Empty masks (BG slices, prompt dropout) yield a learned "no prior" signature from MaskEncoder biases.

**CrossSessionMemoryAttention**
- Cross-session attention: mid-RT features attend to the augmented memory `M_aug = [M_pre ; retrieved change templates]`, conditioned on the continuous-time embedding f(t).
- Within-session attention: current features attend to M_mid (prior mid-RT slices in sequence mode).
- A learned gate combines cross-session and within-session contexts.

**CrossPatientRetrieval**. For every training patient the bank stores a pre-RT embedding key and a set of change templates obtained by spatial-pooling `(mid_features - pre_features)` at tumor locations. At inference, cosine similarity over pre-RT keys selects the top-K donors whose templates are injected into M_aug.

**ChangeHead (auxiliary).** 3-class prediction (stable / grown / shrunk), supervised by XOR-derived labels from pre/mid masks — no extra annotation required. Under `dino_sam`, `pre_feats` for the change head is extracted in `torch.no_grad()` (sequence-mode OOM fix: otherwise a second DINO autograd chain per slice blows a 16 GB GPU at `seq_len=4`).

**FeatureEvolutionPredictor (auxiliary).** Predicts mid-RT tumor-region features from pre-RT features and weeks-elapsed; the MSE between predicted and actual mid-RT features enters the loss as `lambda_consistency * L_cons`.

**Losses.** `L = L_dice + 0.5 * L_bce + 0.1 * L_change + 0.1 * L_cons + lambda_iou * L_iou` (coefficients in `configs/default.yaml`). `lambda_iou` is 0.0 under `dino_sam` (multimask=False) and 0.1 under `sam2`.

**Parameter budget.**
- `sam2`: frozen SAM2-Hiera encoder; ~2.9 M trainable + ~4 M SAM2 mask-decoder at lr×0.1 ≈ **7 M**.
- `dino_sam`: frozen DINOv2-base; ~2.9 M + **~16.3 M Bridge** (13.9 M fusion path + 2.4 M mask cross-attn) + ~4 M SAM2 mask-decoder at lr×0.1 ≈ **23 M**.

## Ablations

| Variant                               | aggDSC |
| ------------------------------------- | ------ |
| RALM-SAM (full, `dino_sam`)           | TBD    |
| — encoder_type=`sam2` (Hiera baseline)| TBD    |
| — mask via SAM2 prompt_enc instead of Bridge cross-attn | TBD    |
| — no cross-patient retrieval          | TBD    |
| — discrete temporal embed (CSM-SAM)   | TBD    |
| — no change map                       | TBD    |
| — no consistency loss                 | TBD    |
| — no sequence training (single slice) | TBD    |
| — pre-RT mask as prompt → argmax      | TBD    |

The `discrete temporal embed` row corresponds to the **CSM-SAM** ablation — the original cross-session-memory-only variant that motivated this work and remains the natural comparison for isolating the retrieval and continuous-time contributions.

## Results

| Method                        | GTVp DSC | GTVn DSC | aggDSC   | HD95 (mm) |
|-------------------------------|----------|----------|----------|-----------|
| HNTS-MRG 2024 Winner          | -        | -        | 72.7     | -         |
| nnUNet (strong baseline)      | -        | -        | TBD      | -         |
| Registration warp (pre→mid)   | -        | -        | TBD      | -         |
| MedSAM2 (within-session only) | -        | -        | ~68.0    | -         |
| CSM-SAM (ablation)            | -        | -        | TBD      | -         |
| RALM-SAM (ours)               | -        | -        | **TBD**  | -         |

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

# Install RALM-SAM
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

**Access**: Register at [Grand Challenge](https://hnts-mrg.grand-challenge.org/) then download from [Zenodo](https://zenodo.org/record/11199559).

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
# Default training — configs/default.yaml has encoder_type=dino_sam, sequence_mode=true
python train.py \
    --config configs/default.yaml \
    --sequence_train \
    --fold 0 --n_folds 5

# Local single-GPU (no SLURM), bs=2 — matches RTX 40-series 16 GB budget
bash scripts/train_local.sh

# SLURM: single GPU or 4-GPU DGX (DDP via torchrun)
sbatch slurm/train_single.sbatch
sbatch slurm/train_dgx.sbatch

# Explicit overrides — toggle encoder or DINO variant from CLI
python train.py --config configs/default.yaml \
    --encoder_type dino_sam \
    --dino_variant dinov2_base \
    --batch_size 2 --num_workers 2 \
    --fold 0 --n_folds 5 \
    --output_dir checkpoints/ralmsam_run1

# Baseline ablation: swap back to SAM2-Hiera encoder
python train.py --config configs/default.yaml --encoder_type sam2 ...

# Resume training
python train.py --config configs/default.yaml --resume checkpoints/ralmsam_run1/latest.pth
```

Key CLI flags:
- `--encoder_type {sam2, dino_sam}` — encoder swap; `dino_sam` is the default on `tals-local`.
- `--dino_variant {dinov2_small, dinov2_base, dinov2_large}` — DINO model size. `dinov2_base` is the ablation-validated choice.
- `--sequence_train` / `--no_sequence_train` — toggle within-session memory accumulation. Sequence mode is required to train the gate + `M_mid` attention.
- `--batch_size`, `--num_workers`, `--prefetch_factor` — dataloader knobs. Don't combine `num_workers=8` with `prefetch_factor>=4` on a desktop-class box; it can stall the host.
- `--fold` / `--n_folds` select the K-fold CV split.
- `--overfit N` — memorize N slices for a sanity check (skips validation, forces `bs=n`, `shuffle=False`, `prompt_dropout=0`, disables periodic checkpoints).

Config ablation toggles (in `configs/default.yaml`):
- `model.encoder_type: sam2 | dino_sam`
- `model.temporal_encoder_type: continuous | discrete` — CSM-SAM ablation uses `discrete`.
- `retrieval.enabled: false` to ablate cross-patient retrieval.
- `loss.lambda_consistency: 0.0` to ablate the evolution consistency loss.
- `training.sequence_mode: false` to ablate sequence training.

## Building the Cross-Patient Bank

Before evaluating with retrieval enabled, build the bank over the training split:

```bash
# NOTE: scripts/build_bank.py is forthcoming — not yet in the repo.
python scripts/build_bank.py \
    --checkpoint checkpoints/best.pth \
    --data_dir data/processed \
    --split train \
    --output bank.pt
```

The resulting `bank.pt` stores per-patient pre-RT keys and tumor-localized change templates consumed by `CrossPatientRetrieval` at inference time. Internally `test.py` and `visualize.py` call `model.set_patient_bank(bank)` after loading.

## Evaluation

```bash
# Evaluate on test set with retrieval and test-time augmentation
python test.py \
    --checkpoint checkpoints/best.pth \
    --bank bank.pt \
    --data_dir data/processed \
    --split test \
    --tta \
    --output_dir results/ralmsam

# K-fold CV evaluation
python test.py \
    --checkpoint checkpoints/best.pth \
    --bank bank.pt \
    --fold 0 --n_folds 5 \
    --output_dir results/ralmsam_fold0

# Output: results/ralmsam/
#   metrics.json       — per-patient and aggregate aggDSC, HD95
#   metrics_table.txt  — formatted comparison table
#   predictions/       — .nii.gz prediction files
```

## Visualization

```bash
# Visualize 10 random test patients
python visualize.py \
    --checkpoint checkpoints/best.pth \
    --bank bank.pt \
    --data_dir data/processed \
    --n_samples 10 \
    --output_dir results/visualizations

# Output per patient:
#   {patient_id}_overlay.png       — pre-RT and mid-RT with GT + prediction overlay
#   {patient_id}_change_map.png    — predicted change map (stable/grown/shrunk)
#   {patient_id}_retrieved.png     — top-K retrieved training donors
#   {patient_id}_slice_gallery.png — multi-slice gallery
```

## Baselines

The `baselines/` directory contains **36 baselines** spanning naive / classical medical / foundation-model / longitudinal-medical / binary change-detection / semantic change-detection. Each has a top-of-file uniqueness note stating what CSM-SAM does that the baseline does not. See [`baselines/LANDSCAPE.md`](baselines/LANDSCAPE.md) for a 6-axis positioning map and [`baselines/SOTA.md`](baselines/SOTA.md) for current state-of-the-art numbers + per-method blurbs.

```bash
# Run all baselines (auto-routes to the right dataset per baseline, 300s per-baseline timeout, idempotent)
bash scripts/run_all_baselines.sh
# Output: results/baselines/<name>/{metrics.json, status.txt, stdout.log} + results/baselines/summary.md

# Individual baselines
python baselines/medsam2_baseline.py --data_dir data/processed          # MedSAM2 (within-session only)
python baselines/nnunet_baseline.py --data_dir data/processed           # nnUNet (strong non-SAM baseline)
python baselines/registration_warp_baseline.py --data_dir data/processed  # Deformable pre->mid warp
python baselines/changeformer_baseline.py --dataset levir_cd --data_dir data/raw/LEVIR-CD
python baselines/scannet_cd_baseline.py --dataset_name second --data_dir data/raw/SECOND/data
```

The registration-warp baseline deformably registers the pre-RT scan onto the mid-RT scan and warps the pre-RT mask forward — a sanity check that any longitudinal method must beat.

## Datasets

Eight datasets are used — HNTS-MRG 2024 (primary), BraTS-GLI 2024 post-treatment (secondary), OAIZIB-CM + MS-Segmentation (single-timepoint pretraining), and LEVIR-CD + S2Looking + SECOND + xBD (transfer to bitemporal change detection). Each has its own loader under `csmsam/datasets/` with a uniform `__getitem__` contract (`pre_image`, `mid_image`, `pre_mask`, `mid_mask`, `change_mask`, plus per-dataset extras). Single-timepoint loaders emit `single_timepoint=True` so the trainer gates the change-head loss. See [CLAUDE.md](CLAUDE.md#datasets) for the full table.

## Paper

`paper/` contains the NeurIPS 2026 LaTeX draft. The Results section is populated with per-dataset tables (booktabs), SOTA rows cited from published papers, and placeholder rows for `CSM-SAM (Ours)`. Compile with `cd paper && make`.

## Project Structure

```
csm-sam/
├── csmsam/
│   ├── modeling/
│   │   ├── cross_session_memory_attention.py   # Cross-session + within-session attention with gating
│   │   ├── retrieval.py                        # CrossPatientBank, CrossPatientRetrieval
│   │   ├── csm_sam.py                          # Full RALM-SAM model (CSM-SAM = ablation)
│   │   ├── change_head.py                      # 3-class change prediction
│   │   └── dino_encoder.py                     # Frozen DINOv2 + trainable Bridge (with mask cross-attn)
│   ├── datasets/
│   │   └── hnts_mrg.py                         # Dataset + augmentation + sequence sampler
│   ├── losses/
│   │   ├── combined_loss.py                    # Dice + CE(change) + consistency
│   │   └── consistency.py                      # FeatureEvolutionPredictor + evolution loss
│   └── utils/
│       ├── metrics.py                          # aggDSC, HD95, IoU
│       ├── visualization.py                    # Overlays, galleries, change maps
│       ├── cv.py                               # K-fold CV split utilities
│       └── tta.py                              # Test-time augmentation
├── data/
│   ├── download_hnts_mrg.py                    # Zenodo download script
│   └── preprocess.py                           # Registration, norm, slicing
├── baselines/                                  # 36 baseline files
│   ├── LANDSCAPE.md                            # 6-axis positioning + per-method table
│   ├── SOTA.md                                 # Current SOTA + per-method blurbs
│   ├── medsam2_baseline.py                     # MedSAM2 (within-session only)
│   ├── nnunet_baseline.py                      # nnUNet + variants
│   ├── registration_warp_baseline.py           # Deformable pre->mid warp baseline
│   ├── {unet_2d, deeplabv3plus, swinunetr, unetr, mednext}_baseline.py   # classical medical seg
│   ├── {sam_vanilla, sam2_point_prompt, sam2_video, dinov2_linear, clipseg, totalsegmentator}_baseline.py  # foundation models
│   ├── {concat_channels, siamese_unet, pre_mask_prior, longisam}_baseline.py  # longitudinal
│   ├── {identity, zero, random, copy_prev_slice, majority_voxel}_baseline.py  # naive/sanity
│   ├── {fc_siam_diff, fc_siam_conc, snunet, bit, changeformer, tinycd}_baseline.py  # binary CD
│   ├── {bisrnet, scannet_cd, ced_net, xview2_dualhrnet}_baseline.py  # semantic CD
│   └── run_all_baselines.py
├── paper/                                      # NeurIPS 2026 LaTeX scaffold
│   ├── main.tex / references.bib / Makefile
│   ├── sections/                               # intro / related / method / results / ablations
│   └── tables/                                 # per-dataset booktabs tables
├── scripts/
│   ├── run_all_baselines.sh                    # End-to-end baseline sweep
│   ├── train_local.sh                          # No-SLURM launcher (bs=2, dino_sam)
│   └── build_bank.py                           # (forthcoming) build CrossPatientBank over train split
├── slurm/
│   ├── train_single.sbatch                     # Single-GPU DGX training
│   └── train_dgx.sbatch                        # 4-GPU DGX (DDP via torchrun)
├── configs/
│   ├── default.yaml                            # RALM-SAM config
│   └── baselines.yaml                          # Baseline configs
├── train.py                                    # Training loop (sequence mode + CV)
├── test.py                                     # Evaluation (TTA + retrieval)
├── visualize.py                                # Test set visualization
├── setup.py
├── requirements.txt
└── CLAUDE.md                                   # AI assistant context
```

## Citation

If you use this code, please cite:

```bibtex
@article{ralmsam2026,
  title={RALM-SAM: Retrieval-Augmented Longitudinal Memory SAM for Adaptive Radiotherapy Mid-Treatment Tumor Segmentation},
  author={},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
