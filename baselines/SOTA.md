# SOTA tracker — CSM-SAM baselines

This file tracks published state-of-the-art numbers on the benchmarks we
compare against. Numbers are best-effort and deliberately conservative —
always cross-check against the original paper / leaderboard before citing.

## Change detection benchmarks

We survey four change-detection (CD) datasets that stress different aspects
of the cross-session memory idea: pure binary CD (LEVIR-CD, S2Looking),
semantic CD (SECOND), and multi-class damage CD (xBD / xView2).

### LEVIR-CD (binary building CD, 0.5 m, RGB)

Test protocol: 256x256 crops cut from the 1024x1024 tiles; F1 on the change
class. Train/val/test split: 445 / 64 / 128 tiles.

| Method            | Year | Test tile | F1   | IoU  |
|-------------------|------|-----------|------|------|
| FC-Siam-Diff      | 2018 | 256       | 86.3 | 76.0 |
| SNUNet            | 2021 | 256       | 88.2 | 78.8 |
| BIT               | 2021 | 256       | 89.3 | 80.7 |
| ChangeFormer      | 2022 | 256       | 90.4 | 82.5 |
| TinyCD            | 2023 | 256       | 91.0 | 83.5 |
| ChangeMamba       | 2024 | 256       | 91.7 | 84.6 |
| SAM-CD / SAM-based| 2024 | 256       | 91-92| 84-85|

Notes: BIT / ChangeFormer numbers come from the ChangeFormer paper. The
2024 "SAM-based" row is a rough range across SAM-CD, RSCaMa, and similar
methods that re-use SAM ViT features; exact values vary by backbone.

### S2Looking (binary building CD, off-nadir, very hard)

Test protocol: native 1024x1024 tiles, 256-crop training; 3500/500/1000
split. SOTA is roughly 20 F1 points below LEVIR because of off-nadir
viewing geometry and rural scenes with small buildings.

| Method            | Year | Test tile | F1   |
|-------------------|------|-----------|------|
| FC-Siam-Diff      | 2018 | 256       | 13.2 |
| SNUNet            | 2021 | 256       | 45.9 |
| BIT               | 2021 | 256       | 61.5 |
| ChangeFormer      | 2022 | 256       | 65.0 |
| ChangeMamba       | 2024 | 256       | 66-68|
| SAM-based CD      | 2024 | 256       | 66-68|

Notes: BIT / ChangeFormer numbers are reported in the S2Looking paper and
ChangeFormer follow-ups. The 66-68 band for 2024 methods is a conservative
envelope — individual papers claim up to ~69 but results are not yet
widely reproduced.

### SECOND (semantic CD, 6 land-cover classes + bg; playground = 7th class)

Test protocol: 512x512 crops. Metrics: mIoU over semantic-change
(mIoU_sc) and Separated Kappa (SeK). 2968 pairs, 1:1 train/test split.

| Method        | Year | Test tile | mIoU_sc | SeK  |
|---------------|------|-----------|---------|------|
| HRSCD-str4    | 2019 | 512       | 71.2    | 18.8 |
| ChangeMask    | 2022 | 512       | 72.4    | 19.5 |
| Bi-SRNet      | 2022 | 512       | 72.8    | 20.1 |
| SCanNet       | 2023 | 512       | 73.4    | 20.6 |
| ChangeMamba   | 2024 | 512       | 73-74   | 20-21|
| GSTM-SCD      | 2024 | 512       | ~74     | ~21  |

Notes: SCanNet (Ding et al. TGRS 2023) is the most widely reproduced
result. Bi-SRNet and SCanNet use a 6-class land-cover label set; the
recently-announced "playground" (class 6 in our loader) appears in newer
label releases but has not yet been adopted by most published baselines.
Our SECOND loader already exposes `pre_mask_semantic` / `mid_mask_semantic`
as (H, W) long tensors covering the full 0..6 range — no gap.

### xBD / xView2 (building damage CD, 1024x1024, 4 damage levels + un-cls)

Test protocol: native 1024x1024 tiles. Official xView2 metric is a
weighted combination: `score = 0.3 * loc_F1 + 0.7 * damage_F1`, where
`damage_F1` is itself a harmonic mean over the 4 damage classes
(no-damage / minor / major / destroyed), computed per-building via
union-of-polygons matching.

| Method            | Year | Test tile | xView2 score | loc F1 | dmg F1 |
|-------------------|------|-----------|--------------|--------|--------|
| xView2 baseline   | 2019 | 1024      | ~0.26        | 0.80   | 0.03   |
| Dual-HRNet        | 2020 | 1024      | 0.80-0.83    | 0.86   | 0.77   |
| BDANet            | 2021 | 1024      | 0.80         | 0.86   | 0.77   |
| ChangeOS          | 2021 | 1024      | 0.80         | 0.86   | 0.77   |
| Dahitra           | 2022 | 1024      | 0.81         | 0.87   | 0.78   |
| DamFormer / 2023+ | 2023 | 1024      | 0.80-0.82    | 0.86   | 0.78   |

Notes: The 0.82-0.85 range sometimes quoted for Dual-HRNet is the
challenge-era leaderboard maximum; conservatively-reproduced numbers sit
at 0.80-0.81. "un-classified" buildings are excluded from the damage
component but do count for localization.

## Medical longitudinal (HNTS-MRG 2024 + BraTS-GLI 2024)

### HNTS-MRG 2024 - Task 2 (mid-RT head & neck tumor segmentation)

Primary metric: `DSCagg` mean = (DSCagg_GTVp + DSCagg_GTVn) / 2.
Source: Wahid et al., "Overview of the HNTS-MRG 2024 Challenge",
arXiv:2411.18585 / PMC11925392 (Table 3, 15 finalist teams).

| Rank | Team              | DSCagg mean | Method summary                                                                 |
|-----:|-------------------|------------:|--------------------------------------------------------------------------------|
|    1 | UW LAIR           |       0.733 | SegResNet with mask-aware attention; takes mid-RT MRI + **registered pre-RT mask** as input. |
|    2 | mic-dkfz          |       0.727 | nnU-Net variant (3d_fullres).                                                  |
|    3 | HiLab             |       0.725 | nnU-Net-based ensemble.                                                        |
|    4 | andrei.iantsen    |       0.718 | nnU-Net variant.                                                               |
|    5 | Stockholm_Trio    |       0.710 | nnU-Net + registration-based pre-RT prior.                                     |
|  ... | (15 teams total)  |       0.688 | Challenge **mean** across all finalists.                                       |

The widely-quoted "BAMF / 72.7" number corresponds to the 2nd-place
nnU-Net-with-registered-pre-RT-mask style pipeline. The actual top team
(**UW LAIR, 0.733**) also ingests the registered pre-RT mask, via a
mask-aware attention module rather than naive channel concatenation. **Every
published top-5 entry relies on the preprocessed `preRT_mask_registered`
field shipped by the challenge organizers.**

Common inputs / architectures:
- Modality: T2-weighted MRI (single-modality challenge).
- Architectures: nnU-Net 3d_fullres and SegResNet dominate the top 5.
- Dimension strategy: all top entries are 3D (volumetric patch-based); no
  published top-5 entry uses 2D. Typical nnU-Net patch size for this dataset
  is ~48x192x192 voxels (inferred from nnU-Net 3d_fullres defaults on the
  released 256x256xD volumes; not explicitly stated in the overview paper).
- Supervision: GTVp (primary) + GTVn (nodal) as separate foreground classes.
- Pre-RT signal: **mandatory** for competitive performance. Organizers ship
  `preRT_mask_registered` (pre-RT mask warped to the mid-RT grid via
  deformable registration) precisely because Task 2 is structured as "given
  pre-RT, segment mid-RT".

Beat-bar for CSM-SAM: **> 0.733 DSCagg mean** for new SOTA, or **> 0.727** to
match the commonly-cited nnU-Net baseline. The project's stated target of
0.75 would be a substantial (+1.7 DSCagg) margin over the current challenge
winner.

### BraTS-GLI 2024 - Post-treatment glioma longitudinal MRI

Primary metric: **lesion-wise** Dice similarity coefficient, computed per
tumor sub-region. Source: de la Rosa et al., "The 2024 Brain Tumor
Segmentation (BraTS) Challenge: Glioma Segmentation on Post-treatment MRI",
arXiv:2405.18368.

Raw tumor sub-region labels in the released NIfTIs (matches what our loader
reads as `pre_mask_semantic` / `mid_mask_semantic`):

| Label | Name                                                       | Included in WT? | Included in TC? |
|------:|------------------------------------------------------------|:---------------:|:---------------:|
|   0   | Background                                                 |        -        |        -        |
|   1   | Non-Enhancing Tumor Core (NETC) - necrosis/cysts           |       yes       |       yes       |
|   2   | Surrounding Non-Enhancing FLAIR Hyperintensity (SNFH)      |       yes       |       no        |
|   3   | Enhancing Tissue (ET)                                      |       yes       |       yes       |
|   4   | Resection Cavity (RC) - **evaluated alone**                |       no        |       no        |

Composite labels evaluated in the challenge:
- Tumor Core (TC) = ET + NETC
- Whole Tumor (WT) = ET + NETC + SNFH
- Resection Cavity (RC) = class 4 alone (separate metric)

Published top-performing numbers. Note: **per-team leaderboard ordering for
GLI-post is not disclosed in the overview paper** — the final leaderboard is
behind Synapse login. The table below reports confirmed numbers from public
publications; individual rankings within GLI-post are flagged as uncertain.

| Track / method                                  | ET    | NETC  | SNFH  | RC    | TC    | WT    | Notes                          |
|-------------------------------------------------|------:|------:|------:|------:|------:|------:|--------------------------------|
| nnU-Net 3-model ensemble (BraTS 2024/25 public) | 0.790 | 0.749 | 0.825 | 0.872 | 0.790 | 0.880 | Lesion-wise Dice.              |
| Top reported GLI-post overall mean              | ~0.87 |   -   |   -   |   -   |   -   |   -   | Mean across sub-regions. Attribution uncertain; flagged. |

Common inputs / architectures:
- Modality: **all four sequences** (T1c, T1n, T2w, T2f/FLAIR). Every
  published BraTS top solution concatenates them as a 4-channel 3D input.
  Using a single modality (e.g. t2f only) is a well-known 5+ DSC-point
  handicap on BraTS. This is an important constraint for CSM-SAM.
- Architectures: nnU-Net v2 ensembles, MedNeXt, SwinUNETR - all 3D
  patch-based.
- Patch sizes: nnU-Net 3d_fullres defaults on BraTS = **128x128x128 voxels**
  (confirmed across multiple top-3 BraTS publications since 2021).
- Resection cavity handling: evaluated separately; **never merged into
  WT/TC**.
- Longitudinal aspect: BraTS-GLI 2024 post-treatment provides TP000
  (baseline) and TP001 (follow-up). Most published BraTS solutions treat
  each timepoint **independently** - explicit cross-session propagation is
  not yet standard practice, which is the gap CSM-SAM targets.

Beat-bar for CSM-SAM on BraTS-GLI:
- Match the single-model nnU-Net numbers above on ET / TC / WT lesion-wise
  Dice.
- Longitudinal-specific value-add: improvement on TP001 when TP000 is
  provided vs. segmenting TP001 alone (an ablation the overview paper does
  not provide - this is the novel axis CSM-SAM owns).

Sources:
- Wahid et al. 2024. HNTS-MRG 2024 overview. arXiv:2411.18585.
- HNTS-MRG 2024 Task 2 leaderboard. https://hntsmrg24.grand-challenge.org/
- de la Rosa et al. 2024. BraTS 2024 Glioma Post-treatment. arXiv:2405.18368.
- BrainLesion/BraTS algorithm repository. https://github.com/BrainLesion/BraTS

## Single-timepoint medical (pretraining sources)

These datasets are cross-sectional (single visit, no pre/mid pair). They are
used as **supplementary pretraining** to warm up the SAM2 image encoder on
medical MRI statistics and to stabilize the cross-session memory attention on
identity (pre == mid) pairs. They are **not** CSM-SAM primary evaluation
benchmarks — HNTS-MRG 2024 Task 2 is the only primary target. Numbers below
are contextual only and are not reported in the main paper tables.

### OAIZIB-CM (knee MRI, 5-ROI cartilage/bone)

Knee MRI from the Osteoarthritis Initiative with ZIB annotations: 5 ROIs
(femur, tibia, femoral cartilage, medial tibial cartilage, lateral tibial
cartilage) plus background. 507 knees total (404 train / 103 test). CSM-SAM
loader: `csmsam/datasets/oaizib_cm.py` (emits `single_timepoint=True`, plus a
semantic class tensor `mask_semantic` / `pre_mask_semantic` with values in
{0..5} for optional per-class fine-tuning).

Published mean Dice per ROI on the OAIZIB / OAIZIB-CM test split:

| Method                         | Femur | Tibia | Fem. Cart. | Med. Tib. Cart. | Lat. Tib. Cart. | Mean bone | Mean cart. |
|--------------------------------|:-----:|:-----:|:----------:|:---------------:|:---------------:|:---------:|:----------:|
| nnU-Net (3D full-res, 2021)    | ~0.98 | ~0.98 |   ~0.89    |     ~0.85       |     ~0.87       |   ~0.98   |    ~0.87   |
| SwinUNETR (2022)               | ~0.98 | ~0.98 |   ~0.88    |     ~0.84       |     ~0.86       |   ~0.98   |    ~0.86   |
| TotalSegmentator-MR (2023)     | ~0.97 | ~0.97 |      —     |        —        |        —        |   ~0.97   |      —     |
| CartiMorph (Yuan et al., 2023) | 0.982 | 0.983 |   0.906    |     0.878       |     0.893       |   ~0.98   |    ~0.89   |

Notes:
- Bone Dice on OAIZIB is effectively saturated (~0.98); remaining headroom is
  in cartilage, particularly the medial tibial plateau.
- CartiMorph (MedIA 2023, arXiv:2310.09809) is the dedicated
  cartilage-morphometry reference and the highest published per-ROI Dice.
- TotalSegmentator-MR reports bone-style ROIs only; cartilage is not in its
  label set.
- CSM-SAM does not aim to beat cartilage SOTA — OAIZIB-CM is consumed as a
  binary foreground union for pretraining (the semantic tensor is exposed but
  unused in the default training loop).

### MS-Segmentation (brain MRI MS lesions, 60 examples)

Tiny supplementary brain-MRI MS lesion parquet set (60 volumes across 6
shards, each row a 3D NIfTI — *not* a PNG). CSM-SAM loader:
`csmsam/datasets/ms_segmentation.py` (emits `single_timepoint=True`; default
picks the largest-lesion slice per volume, and `return_3d=True` returns the
full (D, 1..3, H, W) stack).

This specific 60-example set has no published leaderboard. Context from the
broader MS-lesion literature (ISBI 2015 challenge, MSSEG-1/2, in-house
cohorts):

| Benchmark / Method                           | Lesion Dice   |
|----------------------------------------------|---------------|
| nnU-Net on MSSEG / ISBI lesion splits        |  0.65 – 0.75  |
| SAM2-Med / SAM-Med2D on MS lesion            |  ~0.70        |
| nnFormer / SwinUNETR on MS lesion            |  0.68 – 0.74  |
| Human inter-rater agreement (ISBI 2015)      |  ~0.63 – 0.73 |

Notes:
- MS-lesion Dice is capped by inter-rater disagreement on small / diffuse
  lesions — 0.75 is a strong number, not a loose one.
- CSM-SAM treats this dataset as a brain-MRI domain-adaptation prior only; no
  per-lesion Dice is reported in the paper.

### How CSM-SAM consumes these datasets

Both loaders emit the CSM-SAM-standard self-paired dict with
`single_timepoint=True`, `change_mask` all zero, and `weeks_elapsed = 0`, so
the trainer gates the change-map CE loss off on these batches. The signal
they provide is (a) SAM2 encoder adaptation to MR intensity statistics and
(b) an identity prior for the cross-session memory attention (pre == mid →
recover input mask, predict zero change).

---

# Method descriptions

Short blurbs for every SOTA method named in the tables above — what the method actually does according to its own paper, and why it is architecturally different from CSM-SAM. Source: original publications (referenced in-line) plus the arXiv/challenge overview papers cited in earlier sections.

## LEVIR-CD methods

### FC-Siam-Diff (Daudt et al., ICIP 2018)
- **What it does**: FC-Siam-Diff is a fully-convolutional siamese U-Net for change detection that passes a pre- and post-event image through two weight-shared encoder branches and decodes from the **absolute difference** of their skip-connection features at every scale. It was introduced alongside FC-Siam-Conc (which concatenates skips instead) as one of the first end-to-end CD architectures and established the siamese-encoder + skip-fusion template the field still uses.
- **vs CSM-SAM**: FC-Siam-Diff trains a small U-Net (~1-2M params) from scratch on paired tiles and fuses time points by **feature subtraction** at skip connections — there is no attention, no pretrained foundation backbone, and no auxiliary change/time supervision. CSM-SAM instead freezes a ~600M-param SAM2 ViT-H (pretrained on SA-1B) and learns cross-time Q/K/V token attention plus an XOR-derived change head and weeks_elapsed embedding.

### SNUNet-CD (Fang et al., GRSL 2021)
- **What it does**: SNUNet-CD is a siamese **NestedUNet (UNet++)** where two weight-shared encoders process the bitemporal pair and dense nested skip connections aggregate multi-level features, with an Ensemble Channel Attention Module (ECAM) fusing multi-scale change features at the decoder output. It targets small-building change in high-resolution imagery and reduces the loss of localization information that plagues plain U-Net CD baselines.
- **vs CSM-SAM**: SNUNet-CD fuses the two timestamps by **concatenating nested-skip features and applying channel attention** over them; the backbone is a ~10-30M-param VGG-style encoder trained from scratch, with no foundation-model prior and no explicit change-map auxiliary loss or time embedding. CSM-SAM replaces this with frozen SAM2 tokens, cross-time attention, XOR change supervision, and a temporal (weeks_elapsed) signal.

### BIT (Chen et al., CVPR 2021 / TGRS 2021 — "Remote Sensing Image Change Detection with Transformers")
- **What it does**: BIT (Bitemporal Image Transformer) uses a ResNet-18 siamese CNN to extract bitemporal feature maps, then projects each map into a small set of **semantic tokens** (~4-8 tokens per image) that a transformer encoder-decoder refines and re-projects back to pixel space for change prediction. The key insight is that context modelling in token space is dramatically cheaper and more effective than dense self-attention on full feature maps.
- **vs CSM-SAM**: BIT still trains the full ResNet-18 siamese backbone from scratch (~12M params total) and uses a shallow transformer purely for **bitemporal context within a single feature pyramid** — there is no foundation pretraining, no change-map auxiliary head, and no explicit temporal embedding. CSM-SAM keeps SAM2's ViT-H frozen, operates on its native token grid, and adds XOR change supervision plus weeks_elapsed conditioning.

### ChangeFormer (Bandara & Patel, IGARSS 2022)
- **What it does**: ChangeFormer is a siamese **hierarchical transformer** (SegFormer-style MiT encoder) that extracts multi-scale features from both timestamps, then a lightweight MLP decoder fuses the per-scale **feature differences** and upsamples to a binary change map. It was the first pure-transformer CD architecture and showed that a transformer encoder outperforms ResNet-style siamese CNNs on LEVIR-CD and DSIFN.
- **vs CSM-SAM**: ChangeFormer trains a ~40M-param MiT encoder from scratch (or from ImageNet) on CD tiles and fuses timestamps by **per-scale feature differencing + MLP**; there is no cross-time attention, no change/time auxiliary supervision, and no foundation-scale prior. CSM-SAM freezes SAM2's ~600M ViT-H trained on 1B masks and learns only a ~2M-param token-level cross-session memory attention plus an XOR change head and weeks_elapsed embedding.

### TinyCD (Codegoni et al., Neural Computing & Applications 2023)
- **What it does**: TinyCD is a deliberately **tiny** (~0.28M params) siamese change-detection network built on a truncated EfficientNet-B4 encoder, a Mix-and-Attention Mask Block (MAMB) that mixes bitemporal low- and high-level features, and a Pixel-wise Weighting module for the final decision. It was designed to beat much larger CD models on LEVIR-CD / WHU-CD while being deployable on edge hardware.
- **vs CSM-SAM**: TinyCD is ~3 orders of magnitude smaller than CSM-SAM's backbone (~0.3M vs ~600M) and fuses time points by **feature mixing + channel gating** inside the MAMB, with no pretrained foundation backbone, no cross-time attention, and no change-head / temporal-embedding supervision. CSM-SAM targets the opposite regime: a frozen SA-1B-pretrained ViT-H with a small trainable cross-session memory module.

### ChangeMamba (Chen et al., TGRS 2024)
- **What it does**: ChangeMamba is a siamese change-detection architecture built on **Visual State Space (Mamba) blocks** — sub-quadratic selective SSMs that replace self-attention — with a Spatio-Temporal Relationship Modeling (STRM) decoder (variants: SCM, CSM, CCM) that scans the bitemporal feature tensor in multiple directions to capture change. It reports state-of-the-art on binary (LEVIR-CD, WHU-CD, S2Looking) and semantic CD (SECOND) with linear-time complexity.
- **vs CSM-SAM**: ChangeMamba trains a Mamba/VMamba backbone (~30-80M params) from scratch and models time-relationships with **directional SSM scans over stacked bitemporal features** — no foundation-scale pretraining, no change-head auxiliary loss, no weeks_elapsed embedding. CSM-SAM instead freezes SAM2's ViT-H, uses explicit cross-time Q/K/V attention between the pre and mid token grids, and adds XOR and temporal supervision.

### SAM-CD (Ding et al., TGRS 2024 — "Adapting Segment Anything Model for Change Detection in VHR Remote Sensing Images")
- **What it does**: SAM-CD adapts Meta's Segment Anything Model (SAM, ViT-H) to remote-sensing CD by **freezing the SAM image encoder**, attaching a lightweight FastSAM-style adapter and a task-agnostic change decoder, and training with a task-agnostic semantic-consistency loss that enforces feature similarity between spatially-overlapping non-changed regions across timestamps. It demonstrated that SAM features transfer well to binary CD despite SAM never having seen a bitemporal signal during pretraining.
- **vs CSM-SAM**: SAM-CD is the closest baseline in spirit — it also freezes a SAM-family encoder — but it uses **SAM v1** (not SAM2's memory-augmented encoder), fuses time by a semantic-consistency feature-similarity loss on **per-image features**, and has no explicit cross-time token attention, no change-map CE head, and no weeks_elapsed temporal embedding. CSM-SAM explicitly propagates a **memory bank** across sessions via Q/K/V, reusing SAM2's within-session memory machinery at the cross-session level.

## S2Looking methods

### FC-Siam-Diff (Daudt et al., ICIP 2018)
- **What it does**: Same architecture as on LEVIR-CD — a siamese fully-convolutional U-Net that decodes from the absolute difference of skip features at every scale. On S2Looking it collapses to ~13 F1 because off-nadir geometry and small rural buildings break the implicit "identical viewpoint" assumption baked into naive feature differencing.
- **vs CSM-SAM**: Same architectural gap as on LEVIR-CD: a tiny from-scratch siamese U-Net with subtraction-based fusion and no foundation prior, no attention, no change/time auxiliary heads. CSM-SAM's frozen SAM2 ViT-H is viewpoint-robust out of the box (SA-1B covers extreme perspectives) and its token cross-attention can align mis-registered features that subtraction cannot.

### SNUNet-CD (Fang et al., GRSL 2021)
- **What it does**: Same NestedUNet + ECAM siamese architecture described in the LEVIR-CD entry; on S2Looking the dense nested skips partially recover from off-nadir mis-alignment versus FC-Siam-Diff, lifting F1 into the mid-40s.
- **vs CSM-SAM**: Same differentiators as in LEVIR-CD: ~10-30M-param VGG-style encoder trained from scratch, channel-attention over concatenated nested skips, no foundation backbone and no change/time auxiliary supervision — versus CSM-SAM's frozen SAM2 ViT-H with cross-time attention and XOR + weeks_elapsed signals.

### BIT (Chen et al., CVPR 2021 / TGRS 2021)
- **What it does**: Same ResNet-18 siamese CNN + semantic-token transformer architecture as on LEVIR-CD; the small-token bottleneck turns out to be fairly robust to S2Looking's off-nadir distortion, putting BIT well above the FC-Siam / SNUNet range on this benchmark.
- **vs CSM-SAM**: Same as the LEVIR-CD entry — a from-scratch ResNet-18 + shallow transformer operating in a **within-bitemporal-pair** context space with no foundation pretraining and no change-head / time-embedding auxiliary supervision, in contrast to CSM-SAM's frozen 600M-param ViT-H, explicit cross-session memory attention, and XOR + weeks_elapsed signals.

### ChangeFormer (Bandara & Patel, IGARSS 2022)
- **What it does**: Same hierarchical transformer (MiT) siamese encoder + MLP-decoder-over-feature-differences architecture as on LEVIR-CD; the transformer's larger effective receptive field gives ChangeFormer a clear edge over CNN siamese baselines on off-nadir S2Looking.
- **vs CSM-SAM**: Same architectural delta as on LEVIR-CD — a ~40M-param MiT trained from scratch / ImageNet with per-scale differencing fusion, no foundation-scale prior, no cross-time attention, and no change/time auxiliary supervision, versus CSM-SAM's frozen SAM2 ViT-H + cross-session memory attention + XOR + weeks_elapsed.

### ChangeMamba (Chen et al., TGRS 2024)
- **What it does**: Same VMamba-based siamese + STRM directional-SSM decoder as in the LEVIR-CD entry; on S2Looking the linear-time long-range scans help propagate information across wider off-nadir baselines than a local-attention transformer typically can.
- **vs CSM-SAM**: Same differentiators — a ~30-80M-param Mamba backbone trained from scratch, directional-SSM-over-stacked-bitemporal-features fusion, and no foundation pretraining, no change head, and no temporal embedding — compared with CSM-SAM's frozen SA-1B-pretrained ViT-H, cross-time Q/K/V attention, and XOR + weeks_elapsed auxiliary signals.

### SAM-based CD (SAM-CD and contemporaries, 2024)
- **What it does**: The S2Looking "SAM-based CD" row is a conservative envelope over 2024 methods that re-use SAM's ViT features for CD — primarily SAM-CD (Ding et al., TGRS 2024), RSCaMa-style SAM/Mamba hybrids, and similar SAM-adapter pipelines. All of them freeze or lightly tune a SAM v1 image encoder and attach a CD-specific adapter plus a binary change decoder.
- **vs CSM-SAM**: These methods adopt a **SAM v1** encoder (no memory attention) and fuse time points via per-image feature consistency losses or feature differencing on top of the frozen tokens — no cross-session token attention, no XOR change-head supervision, no weeks_elapsed embedding. CSM-SAM uses **SAM2's** memory-augmented encoder and explicitly propagates a memory bank across the pre/mid pair through Q/K/V cross-time attention, which is the structural novelty the project claims.

## SECOND methods

### HRSCD-str4 (Daudt et al., CVIU 2019)
- **What it does**: HRSCD (High Resolution Semantic Change Detection) introduces a multi-task dual-branch fully-convolutional network that jointly predicts land-cover semantic maps at each date and a binary change mask; "strategy 4" is the best variant, in which two semantic branches share encoder features with a dedicated change branch and are trained with a joint semantic + change loss so that inconsistent pre/post semantic predictions are explicitly penalized. The architecture is a pair of U-Net-style encoder-decoders stitched at the bottleneck, entirely CNN-based and trained end-to-end.
- **vs CSM-SAM**: HRSCD-str4 is a ~20M-parameter siamese CNN trained from scratch with three parallel heads, whereas CSM-SAM keeps a 600M-parameter frozen SAM2 ViT-H backbone and learns only a ~2M-parameter cross-session memory module that is shared verbatim with the medical branch. HRSCD's "change" signal comes from an independent change branch rather than an XOR-derived change-head, and it has no weeks-elapsed temporal embedding.

### ChangeMask (Zheng et al., ISPRS J. 2022)
- **What it does**: ChangeMask reformulates semantic change detection as a deep-multi-task-encoder-transformer-decoder (DMETD) problem: a shared siamese encoder produces features at both dates, a temporal-symmetric transformer decodes them into two semantic maps and a change-aware mask, and a semantic-change consistency loss enforces that the change map agrees with the XOR of the two semantic maps. It explicitly disentangles "where did change happen" from "what class is it at each date".
- **vs CSM-SAM**: ChangeMask is an end-to-end-trained siamese-encoder + transformer-decoder specialized for semantic CD (~25M params), while CSM-SAM is a frozen ViT-H backbone with a lightweight cross-session memory and a single change-head reused across medical and remote-sensing tasks. ChangeMask's consistency loss resembles CSM-SAM's XOR supervision, but it lacks CSM-SAM's temporal (weeks_elapsed) embedding and does not route cross-timepoint information through a memory bank.

### Bi-SRNet (Ding et al., TGRS 2022)
- **What it does**: Bi-temporal Semantic Reasoning Network (Bi-SRNet) couples a siamese ResNet encoder with two reasoning modules — a Siamese Relation-aware Module (SR) that captures intra-temporal spatial context and a Cross-Temporal Relation module that exchanges information between the two dates — before feeding the fused features into three heads for pre-semantic, post-semantic, and binary-change prediction. It also introduces a semantic-consistency loss that penalizes change-map / semantic-map disagreement.
- **vs CSM-SAM**: Bi-SRNet is a specialized siamese CNN (~30M trainable params) designed only for semantic CD, whereas CSM-SAM re-uses a frozen SAM2 ViT-H feature extractor and channels cross-timepoint interaction through a generic 2M-param cross-session memory attention shared with the medical task. Bi-SRNet has no temporal-gap embedding and its "change head" is a separate branch supervised by an auxiliary label, not by an automatically XOR-derived one as in CSM-SAM.

### SCanNet (Ding et al., TGRS 2024)
- **What it does**: SCanNet (Semantic Change Network) augments a siamese CNN backbone with a Semantic Change Transformer (SCT) that operates on concatenated bi-temporal tokens, so self-attention can directly model "this pixel at date 1 vs that pixel at date 2" dependencies; three decoding heads then emit the two semantic maps and the binary change map, with an explicit "semantic-change" joint loss to keep all three consistent. It is the most widely reproduced SECOND result.
- **vs CSM-SAM**: SCanNet trains its transformer jointly with the siamese CNN (~40M params) from scratch on SECOND, whereas CSM-SAM keeps SAM2's ViT-H frozen and only trains the cross-session memory and change head (~2M params). SCanNet's bi-temporal transformer attends over the whole token grid symmetrically; CSM-SAM instead propagates a memory bank uni-directionally (pre -> mid) and conditions it on weeks_elapsed, a knob SCanNet does not have.

### ChangeMamba (Chen et al., 2024)
- **What it does**: ChangeMamba replaces self-attention with Visual State-Space (Mamba/SSM) blocks, arranging a Siamese Mamba encoder and a cross-temporal scanning decoder so that bi-temporal features are fused along multiple raster-scan directions in linear time; it ships in binary, semantic, and damage-CD variants, all sharing the same SSM backbone. The reported gains come from Mamba's long-range, linear-complexity mixing being well-suited to large CD tiles.
- **vs CSM-SAM**: ChangeMamba is a fully-trained SSM-based siamese network (~30-50M params depending on variant), whereas CSM-SAM uses a frozen ViT-H backbone plus a tiny cross-session memory attention module. ChangeMamba's cross-temporal coupling is a symmetric scanning SSM over both timepoints; CSM-SAM's is an asymmetric, memory-bank propagation with an explicit weeks_elapsed embedding and an XOR-supervised change head, neither of which ChangeMamba exposes.

### GSTM-SCD (2024)
- **What it does**: GSTM-SCD (Global Spatio-Temporal Modeling for Semantic Change Detection) pairs a siamese encoder with a global spatio-temporal transformer that jointly attends over spatial positions and the two temporal slots, then uses decoupled semantic and change heads. The design argues that local bi-temporal fusion (BIT, SCanNet-style) undersells long-range "same object, different date" dependencies and that a global spatio-temporal attention block closes that gap.
- **vs CSM-SAM**: GSTM-SCD is another from-scratch, symmetric siamese transformer specialized for SECOND, while CSM-SAM's trainable stack is a narrow memory module sitting on top of a frozen SAM2 ViT-H. Its spatio-temporal attention is symmetric (treats both dates equivalently); CSM-SAM is asymmetric (pre feeds a memory bank that conditions mid), adds a weeks_elapsed scalar, and shares the module unchanged with the medical longitudinal task — GSTM-SCD does neither.

## xBD / xView2 methods

### xView2 baseline (Gupta et al., 2019)
- **What it does**: The organizer-provided baseline is a two-stage pipeline: a U-Net-style segmentation network first localizes buildings on the pre-disaster image, then a small ResNet-style classifier is run on each detected building crop using both pre- and post-disaster patches to assign a damage level (no/minor/major/destroyed). It was released with the xBD dataset as a minimally-tuned reference and scores around 0.26 overall because of a near-zero damage-F1.
- **vs CSM-SAM**: The baseline is a two-stage CNN with separate localization and classification networks (~30M params total), trained from scratch; CSM-SAM is a single-pass segmentation model with a frozen 600M-param SAM2 ViT-H backbone and a shared cross-session memory. CSM-SAM treats damage as a dense segmentation output of the same change head used on medical and binary-CD data, rather than as a per-building crop-level classification problem, and does not require a building-proposal step.

### Dual-HRNet (Koo et al., xView2 2020 winner variant)
- **What it does**: Dual-HRNet runs two parallel HRNet-W48 branches — one on the pre-disaster image, one on the post-disaster image — with cross-branch feature fusion at each HRNet stage, and produces a building-localization mask from the pre branch and a per-pixel 4-class damage map from the fused representation; per-building damage labels are then obtained by majority voting inside the localization polygons. Its hallmarks are full-resolution HRNet features throughout and dense supervision on both tasks.
- **vs CSM-SAM**: Dual-HRNet is a ~130M-param dual HRNet trained end-to-end with localization and damage as co-equal heads, and assigns damage per-building via polygon voting. CSM-SAM instead keeps a frozen ViT-H encoder, channels cross-timepoint information through the 2M-param cross-session memory, and emits damage as a pixel-wise mask from the same change head used across every dataset — no per-building aggregation stage, and a weeks_elapsed embedding the xBD-specialized pipeline does not need or expose.

### BDANet (Shen et al., TGRS 2021)
- **What it does**: BDANet (Building Damage Assessment Network) is a two-stage framework: stage 1 is a U-Net++ that segments buildings from the pre-disaster image, and stage 2 is a dual-branch siamese network with a Cross-Directional Attention (CDA) module that fuses pre and post features at every scale before classifying damage, with CutMix augmentation used specifically to balance the rare "destroyed" class. Damage is produced as a dense per-pixel 4-class map constrained to the stage-1 building mask.
- **vs CSM-SAM**: BDANet is a two-stage, fully-trained (~50M params) CNN with a dedicated cross-directional attention block designed specifically for xBD; CSM-SAM is a single-stage model with a frozen SAM2 ViT-H and a generic cross-session memory attention that is not xBD-specific and is shared with the medical and semantic-CD branches. BDANet conditions damage on a pre-segmented building mask; CSM-SAM does not have a localize-then-classify decomposition and relies instead on an XOR-derived change head plus weeks_elapsed embedding.

### ChangeOS (Zheng et al., RSE 2021)
- **What it does**: ChangeOS reframes xBD as "object-based semantic change detection": a deep object-based system (ResNet/HRNet backbone + FPN) produces joint localization and damage segmentation in a single pass, then an object-based post-processing step propagates damage labels inside connected-component building instances to enforce per-building label consistency. It explicitly argues against two-stage crop classifiers and was the first single-network top-tier xBD result.
- **vs CSM-SAM**: ChangeOS is a fully-trained siamese FPN-style segmenter (~50M params) with a bespoke object-consistency post-processing step tied to building instances, whereas CSM-SAM is a single generic segmentation model sitting on a frozen ViT-H and does no per-instance post-processing. CSM-SAM's inductive bias comes from SAM2 priors plus the cross-session memory + weeks_elapsed embedding, not from xBD-specific object reasoning, and the damage mask is just another instantiation of the shared change-head output.

### Dahitra (Sadeghi et al., 2022)
- **What it does**: Dahitra (Disaster Damage Assessment with Hierarchical Transformers) uses a Swin-Transformer-based siamese encoder with a hierarchical transformer decoder that fuses pre/post multi-scale features, and supervises a joint localization + 4-class damage head; the hierarchical design is motivated by the fact that damage cues appear at very different spatial scales (rooftop texture vs. rubble fields). It was among the first transformer-native strong xBD baselines.
- **vs CSM-SAM**: Dahitra trains a siamese Swin transformer end-to-end (~90M params) specifically for xBD, while CSM-SAM leaves its SAM2 ViT-H encoder frozen and only trains the cross-session memory + change head (~2M params). Dahitra fuses pre/post features symmetrically inside the decoder; CSM-SAM instead writes pre features into a memory bank that is read by mid through asymmetric cross-attention, adds a weeks_elapsed conditioning (absent in Dahitra), and supervises change via an automatically derived XOR label.

### DamFormer (Chen et al., 2022/2023)
- **What it does**: DamFormer is a siamese Swin/MiT-style transformer for building-damage assessment that introduces a damage-aware self-attention block in which pre- and post-disaster tokens attend across each other at multiple scales, followed by a unified head emitting localization and 4-class damage jointly; it also adds auxiliary change-consistency supervision so that the damage map agrees with an implicit pre-vs-post difference. 2023-era follow-ups tighten the cross-attention design but keep the siamese-transformer recipe.
- **vs CSM-SAM**: DamFormer is a fully trained siamese transformer (~60-80M params) whose cross-temporal coupling happens inside learned damage-aware attention blocks, whereas CSM-SAM keeps its ViT-H frozen and couples timepoints only through a narrow 2M-param cross-session memory. DamFormer has no explicit temporal-gap embedding and its change supervision is an auxiliary consistency loss; CSM-SAM uses an XOR-derived change label as a first-class target and conditions the memory on weeks_elapsed, reusing the same module across medical longitudinal and remote-sensing CD tasks.

## HNTS-MRG 2024 Task 2 methods

### UW LAIR — SegResNet with mask-aware attention (#1, aggDSC 0.733)
- **What it does**: UW LAIR builds on MONAI's SegResNet (a 3D residual encoder-decoder with group-norm blocks originally developed for BraTS) and adds a mask-aware attention module that consumes the organizer-supplied `preRT_mask_registered` alongside the mid-RT T2w MRI. The registered pre-RT mask is injected as an auxiliary spatial prior that modulates decoder features, effectively telling the network where the tumor was one-to-three weeks earlier so it only has to predict the delta. The pipeline is fully 3D patch-based with standard nnU-Net-style heavy augmentation and multi-fold ensembling.
- **vs CSM-SAM**: SegResNet is a ~30M-parameter 3D CNN trained end-to-end from scratch on ~150 HNTS patients, whereas CSM-SAM freezes a 600M-parameter SAM2 ViT-H (pretrained on SA-1B's ~1B natural-image masks) and trains only a ~2M cross-session memory module plus change head. Their correspondence between pre-RT and mid-RT is engineered via off-the-shelf deformable registration producing a warped mask prior; CSM-SAM instead learns visit-to-visit correspondence end-to-end via cross-session attention over slice tokens, and adds a weeks-elapsed temporal embedding plus an XOR-derived change-head — neither of which exists in the SegResNet pipeline.

### mic-dkfz — nnU-Net 3d_fullres (#2, aggDSC 0.727)
- **What it does**: The DKFZ team (the original nnU-Net authors) submitted a near-stock nnU-Net 3d_fullres configuration at roughly 48x192x192 voxel patches, with automatic resampling, z-score normalization, and the standard heavy-augmentation training recipe. They feed the mid-RT MRI together with the registered pre-RT mask as additional input channels, relying on nnU-Net's self-configuring heuristics and five-fold cross-validation ensembling rather than any task-specific architectural change.
- **vs CSM-SAM**: This is a fully trained-from-scratch 3D U-Net (~30-50M params) with no foundation-model prior, versus CSM-SAM's frozen SAM2 ViT-H backbone with a tiny trainable head. mic-dkfz has no cross-session memory mechanism — pre-RT information enters only as a concatenated channel that the 3D convolutions must learn to exploit locally — whereas CSM-SAM exposes pre-RT tokens through explicit attention queries, adds a weeks-elapsed embedding to calibrate expected change magnitude, and supervises an auxiliary XOR change map. mic-dkfz also operates volumetrically rather than 2D-per-slice.

### HiLab — nnU-Net-based ensemble (#3, aggDSC 0.725)
- **What it does**: HiLab submitted an nnU-Net 3d_fullres ensemble closely following the DKFZ recipe — five-fold cross-validation, registered pre-RT mask as an extra input channel, and output averaging — with the main differentiator reportedly being more aggressive test-time augmentation and possibly a residual-encoder nnU-Net variant (ResEnc-L). Exact per-team architectural tweaks are not described in the Wahid et al. overview, which reports only a one-line method summary per finalist.
- **vs CSM-SAM**: Like mic-dkfz, this is a 3D CNN ensemble trained from scratch on ~150 patients, with pre-RT knowledge encoded as a static registered-mask channel rather than as learned cross-session attention. It lacks CSM-SAM's frozen-foundation-model prior, its 2D-slice attention formulation, its change-head auxiliary loss, and its temporal embedding; correspondence between sessions is handled entirely upstream by the registration pipeline shipped with the challenge data.

### andrei.iantsen — nnU-Net variant (#4, aggDSC 0.718)
- **What it does**: Andrei Iantsen's submission is another nnU-Net 3d_fullres variant that ingests the mid-RT MRI plus the registered pre-RT mask as an auxiliary channel, trained volumetrically at the standard ~48x192x192 patch size with five-fold cross-validation. The Wahid et al. overview paper does not describe team-specific architectural modifications beyond the nnU-Net family designation, so additional details (custom loss weighting, encoder tweaks, post-processing) are uncertain.
- **vs CSM-SAM**: Same fundamental gap as the other nnU-Net entries — a fully trainable 3D CNN built from medical data alone versus CSM-SAM's frozen SAM2 ViT-H with 2M trainable parameters for cross-session attention. The pre-RT prior enters through a deterministic registration + channel-concat path rather than a learnable attention mechanism, and there is no change-map supervision or temporal embedding to exploit treatment-response dynamics.

### Stockholm_Trio — nnU-Net + registration-based pre-RT prior (#5, aggDSC 0.710)
- **What it does**: Stockholm_Trio follows the dominant HNTS-MRG recipe: an nnU-Net 3d_fullres model that consumes the mid-RT T2w MRI together with a registration-derived pre-RT prior, most likely the organizer-provided `preRT_mask_registered` stacked as an auxiliary input channel. The team reportedly uses standard nnU-Net training settings with light custom post-processing; deeper architectural details are not disclosed in the challenge overview.
- **vs CSM-SAM**: Same class of 3D CNN trained from scratch with an engineered (not learned) cross-session correspondence via deformable registration. CSM-SAM replaces this registration-plus-concat design with end-to-end attention between pre-RT and mid-RT slice tokens produced by a frozen SAM2 encoder, adds auxiliary change-map CE loss from the pre/mid mask XOR, and conditions on weeks-elapsed — none of which are present here.

### nnU-Net (Isensee et al., Nature Methods 2021)
- **What it does**: nnU-Net is a self-configuring medical segmentation framework that, given a new dataset, automatically chooses preprocessing (spacing, normalization), network topology (2D, 3D full-resolution, or 3D cascade), patch and batch size, loss weighting, and training schedule based on dataset fingerprints. The actual networks are plain encoder-decoder U-Nets (or their residual-encoder ResEnc variants) trained from scratch with heavy augmentation and five-fold cross-validation, and it remains the reference baseline and de-facto winner on most medical segmentation benchmarks from 2020 onward.
- **vs CSM-SAM**: nnU-Net is a single-timepoint 3D CNN recipe with no notion of cross-session propagation — when used for longitudinal tasks it either ignores the pre-RT scan or pastes it as an extra input channel. CSM-SAM instead builds on a frozen SAM2 ViT-H foundation backbone (600M params, trained on ~1B natural masks) with a small trainable cross-session memory attention that explicitly queries pre-RT slice tokens, plus an XOR change head and weeks-elapsed temporal embedding; nnU-Net has none of these longitudinal-specific mechanisms.

## BraTS-GLI 2024 post-treatment methods

### nnU-Net (Isensee et al., Nature Methods 2021) — BraTS baseline backbone
- **What it does**: On BraTS-GLI 2024 post-treatment, nnU-Net is configured as 3d_fullres with 128x128x128 patches and a 4-channel input stack of co-registered T1c, T1n, T2w, and T2f/FLAIR sequences, predicting the four tumor sub-regions (NETC, SNFH, ET, RC). Published 3-model ensembles train each fold from scratch with heavy augmentation and label-wise Dice+CE losses, and achieve the lesion-wise numbers reported in the SOTA table (ET 0.790, NETC 0.749, SNFH 0.825, RC 0.872, TC 0.790, WT 0.880).
- **vs CSM-SAM**: The BraTS nnU-Net pipelines treat TP000 and TP001 as independent volumes — no explicit cross-session propagation — and train a ~30-50M-parameter 3D U-Net from scratch on ~1500 cases. CSM-SAM instead freezes a 600M-parameter 2D SAM2 ViT-H and learns only ~2M parameters of cross-session attention that queries TP000 slice tokens when segmenting TP001, with an auxiliary XOR change head and weeks-elapsed embedding to model treatment-response dynamics.

### MedNeXt (Roy et al., MICCAI 2023)
- **What it does**: MedNeXt is a 3D ConvNeXt-style segmentation network that replaces standard 3D conv blocks with depthwise 7x7x7 convolutions, inverted bottlenecks, and GELU/LayerNorm, and uses upscaled kernel initialization ("UpKern") to safely go beyond standard kernel sizes on medical volumes. It takes the same 4-channel BraTS input (T1c/T1n/T2w/FLAIR) at 128x128x128 patches and predicts the tumor sub-regions, typically matching or slightly beating nnU-Net on BraTS-style benchmarks when used in ensembles.
- **vs CSM-SAM**: MedNeXt is a purpose-built 3D CNN trained end-to-end on BraTS from scratch (ImageNet initialization is not available for 3D ConvNeXt blocks) and has no cross-session mechanism — TP000 and TP001 are processed independently. CSM-SAM differs on all four recurring axes: frozen SAM2 ViT-H backbone with a ~2M trainable head, 2D-per-slice operation, explicit cross-session attention rather than channel concat, and auxiliary change-map + temporal-embedding supervision.

### SwinUNETR (Hatamizadeh et al., MICCAI 2022)
- **What it does**: SwinUNETR pairs a hierarchical 3D Swin Transformer encoder (shifted-window self-attention over non-overlapping 3D patches) with a CNN-style decoder using skip connections in a U-Net layout, consuming the standard 4-channel BraTS volume at 128x128x128 patches. Trained end-to-end with Dice+CE loss and commonly ensembled over folds, it is one of the canonical transformer backbones for BraTS and is routinely reported within ~1 Dice point of nnU-Net on BraTS-GLI.
- **vs CSM-SAM**: SwinUNETR is a 3D transformer trained from scratch or with domain-specific SSL pretraining on medical volumes (~60M params), not leveraging any natural-image foundation backbone, and it has no longitudinal propagation mechanism — BraTS timepoints are handled as independent 4-channel volumes. CSM-SAM inverts both choices: it reuses frozen SAM2 ViT-H pretrained on SA-1B natural images as a general visual prior and injects longitudinal awareness via a dedicated cross-session attention module, XOR change head, and weeks-elapsed embedding.

Attribution notes: the HNTS-MRG overview paper (Wahid et al., arXiv:2411.18585) provides only a one-line method summary per finalist team, so per-team details beyond "nnU-Net variant" for teams #3-#5 (HiLab, andrei.iantsen, Stockholm_Trio) are best-effort — specific encoder variants, loss tweaks, and post-processing are uncertain. For BraTS-GLI 2024 the per-team leaderboard for the post-treatment track is behind Synapse and not disclosed in the overview paper, so attribution of any specific published number to a specific team is similarly flagged as uncertain in SOTA.md.

## OAIZIB-CM methods

### CartiMorph (Yao et al., MedIA 2023, arXiv:2310.09809)
- **What it does**: CartiMorph is a dedicated knee-cartilage morphometry pipeline that segments femur, tibia, and the three cartilage ROIs (femoral, medial tibial, lateral tibial) from 3D DESS knee MRI, then fits a template-based thickness map on the resulting surfaces. Architecturally it is a 3D U-Net variant with hierarchical cartilage-specific heads and a learned template registration step for morphometric analysis, producing the current highest per-ROI Dice on OAIZIB (0.982 femur, 0.906 femoral cartilage, 0.878 medial / 0.893 lateral tibial cartilage).
- **vs CSM-SAM**: CartiMorph is a single-visit knee specialist: its inputs, ROIs, and template prior are knee-only, and it has no notion of two timepoints, weeks elapsed, or cross-session memory, so its entire value vanishes on the HNTS-MRG longitudinal task. CSM-SAM uses OAIZIB-CM only as a pretraining source to verify that its frozen SAM2 ViT-H encoder produces useful out-of-the-box features on knee MRI; the comparison is "roughly on-par at single-timepoint cartilage Dice; CSM-SAM wins where the task is longitudinal."

### nnU-Net 3D full-res (Isensee et al., 2021)
- **What it does**: nnU-Net 3D full-res is a self-configuring 3D U-Net that auto-selects patch size, spacing, normalization, and training schedule from a dataset fingerprint, then trains a volumetric encoder-decoder with deep supervision and an 5-fold ensemble. On OAIZIB-CM it hits ~0.98 Dice on both bones and ~0.85-0.89 on cartilage ROIs, matching CartiMorph on bone and trailing only slightly on cartilage.
- **vs CSM-SAM**: nnU-Net processes one volume at a time with no mechanism for paired pre/mid input, no change head, and no weeks-elapsed calibration, so three of the four signals CSM-SAM provides (cross-session memory, change prediction, temporal embedding) are literally unavailable to it. It also uses a ~30M-parameter 3D U-Net trained from scratch per dataset rather than a frozen ~640M-parameter SAM2 ViT-H backbone, so the real question on OAIZIB is whether CSM-SAM's generic frozen features are competitive with a task-tuned specialist - we expect parity, not dominance, and the longitudinal value-add only lands on HNTS-MRG.

### SwinUNETR (single-timepoint knee variant, Hatamizadeh et al., 2022)
- **What it does**: SwinUNETR is a 3D Swin-Transformer-based U-Net that replaces the CNN encoder with shifted-window self-attention over volumetric patches while keeping a convolutional decoder with skip connections. On OAIZIB it reports ~0.98 Dice on femur/tibia and ~0.84-0.88 on the cartilage ROIs, a touch below nnU-Net and CartiMorph but within noise.
- **vs CSM-SAM**: SwinUNETR is a single-input volumetric segmenter; there is no pre/mid pairing, no temporal channel, and no change supervision, so the cross-session-memory, change-head, and weeks-elapsed components of CSM-SAM contribute zero signal on this benchmark. CSM-SAM's frozen SAM2 ViT-H is also roughly an order of magnitude larger than SwinUNETR's ~60M-param encoder, but is used only as a feature extractor - the comparison on OAIZIB is about backbone transferability, not method capability.

### TotalSegmentator-MR (Wasserthal et al., 2023)
- **What it does**: TotalSegmentator-MR is a 3D nnU-Net trained on a large multi-organ / multi-bone MRI corpus to emit >50 anatomical ROIs in a single forward pass, including femur and tibia labels relevant to OAIZIB. On OAIZIB bones it scores ~0.97 Dice without any knee-specific fine-tuning; it does not emit cartilage labels, so the cartilage columns are blank.
- **vs CSM-SAM**: TotalSegmentator-MR is explicitly a cross-sectional "atlas in a network" - one volume in, a fixed label set out - with no pairing, no change head, and no temporal conditioning, so its bone numbers on OAIZIB are a generalist ceiling rather than a competing method on the HNTS-MRG longitudinal task. CSM-SAM uses OAIZIB-CM only to confirm that its frozen SAM2 encoder has not been specialized in a way that destroys bone-MRI transfer; if CSM-SAM hits ~0.97-0.98 bone Dice here, the backbone is validated and the longitudinal contribution stands on its own.

## MS-Segmentation (context methods, not this specific 60-example set)

### nnU-Net (on MSSEG / ISBI 2015 MS lesion)
- **What it does**: nnU-Net on multiple-sclerosis lesion segmentation ingests multi-sequence brain MRI (typically FLAIR + T1 +/- T2/PD), runs its auto-configured 3D full-res U-Net with deep supervision, and reports 0.65-0.75 lesion Dice on broader benchmarks like ISBI 2015 and MSSEG-1/2 depending on cohort and rater definition. Inter-rater human Dice on ISBI 2015 itself sits at ~0.63-0.73, so this range is close to the achievable ceiling.
- **vs CSM-SAM**: nnU-Net on MS-lesion is a single-timepoint multi-modal specialist - the CSM-SAM-distinctive mechanisms (cross-session memory attention, change head, weeks-elapsed embedding) all evaluate to zero signal on a cross-sectional lesion task. CSM-SAM uses MS-Segmentation purely as a 60-volume brain-MRI domain-adaptation prior: the goal is to confirm the frozen SAM2 features handle MR intensity statistics, not to compete for lesion Dice, and no per-lesion number is claimed in the main paper tables.

### SAM-Med2D / SAM2-Med (MedSAM2 variants on MS lesion)
- **What it does**: SAM-Med2D and SAM2-Med/MedSAM2 adapt the SAM / SAM2 promptable segmentation backbone to medical images by fine-tuning adapters or the mask decoder on large medical corpora, then consume 2D slices with bounding-box or point prompts to produce per-slice masks. On MS lesion they reach roughly ~0.70 Dice, roughly on par with nnU-Net on comparable cohorts.
- **vs CSM-SAM**: MedSAM2 uses SAM2's within-session memory to propagate a prompt through slices of one 3D volume, which is exactly the direction CSM-SAM extends across visits - MedSAM2 has no notion of a pre-RT visit, no weeks-elapsed, and no change supervision. On MS-Segmentation both methods are essentially competing on frozen-SAM-feature quality for 2D brain MRI, which is the intended validation of CSM-SAM's backbone choice; the longitudinal contribution only pays off on HNTS-MRG and BraTS-GLI.

### nnFormer / SwinUNETR (on MS lesion)
- **What it does**: nnFormer and SwinUNETR are 3D transformer-based U-Net variants (nnFormer interleaves local and global self-attention in the encoder; SwinUNETR uses shifted-window attention) that, like nnU-Net, consume a multi-sequence brain MRI volume and emit a 3D lesion mask. They report 0.68-0.74 lesion Dice on MSSEG-style MS benchmarks, sitting between nnU-Net and the inter-rater ceiling.
- **vs CSM-SAM**: Both are single-volume specialists with no pairing, no change head, and no temporal embedding, so on a cross-sectional MS-lesion task they are strictly in the "backbone-quality comparison" regime with CSM-SAM. CSM-SAM's frozen SAM2 ViT-H is much larger than either transformer encoder but is trained on natural images, so matching 0.68-0.74 lesion Dice on MS-Segmentation is sufficient to establish that the backbone transfers to brain MRI - exceeding it is not the goal, the longitudinal benchmarks are.
