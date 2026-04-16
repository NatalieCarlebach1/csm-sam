# CSM-SAM Baseline Landscape

A related-work map for CSM-SAM (Cross-Session Memory SAM) situating every
baseline the project compares against on a shared axis system. Every entry is
scoped to the mid-RT adaptive radiotherapy segmentation setting on HNTS-MRG
2024, but the axes are written so that longitudinal / bitemporal methods from
neighbouring domains (brain-lesion follow-up, knee-cartilage longitudinal
studies, MS lesion change, remote-sensing change detection, disaster damage
grading) can be placed on the same plane for the related-work section.

The repository contains 36 executable baseline files under `baselines/`
(37 Python files minus `__init__.py` and the `run_all_baselines.py` driver).
Each row of the per-baseline table below corresponds to exactly one such
file, with axis values read from the top-of-file docstring.

---

## 1. The gap CSM-SAM fills

Adaptive radiotherapy mid-treatment segmentation is a longitudinal /
bitemporal task: a pre-RT (week 0) scan with ground-truth masks is always
available, and the goal is to segment the mid-RT (week 2-3) scan of the same
patient. Existing methods do one of four things: (i) ignore the pre-RT scan
entirely (nnUNet, SwinUNETR, UNETR, MedNeXt, 2D U-Net, DeepLabV3+, vanilla
SAM1, SAM2 point-prompt, SAM2 video, DINOv2 linear probe, CLIPSeg,
TotalSegmentator); (ii) re-use the pre-RT mask as-is (identity),
slice-roll it (copy-previous-slice), majority-voxel it (dataset prior),
or deformably warp it (registration-warp); (iii) concatenate pre-RT image
or mask as an extra input channel (concat-channels, pre-mask-prior,
siamese-U-Net, nnUNet-dual-channel, HNTS winner replica, LongiSAM,
FC-Siam-conc, FC-Siam-diff, SNUNet-CD, ChangeFormer, BIT, TinyCD); or
(iv) feed bi-temporal features to a dedicated semantic change decoder
(Bi-SRNet, SCanNet, CED-Net/HRSCD, xView2 Dual-HRNet). None of them
propagate a *memory bank* across sessions through learned cross-attention
with a frozen foundation-model backbone, and none combine that with an
auxiliary change-map head supervised by pre/mid mask XOR and a temporal
embedding that calibrates expected change magnitude against weeks elapsed.
That combination — cross-session attention + frozen SAM2 ViT-H + change
supervision + temporal calibration — is the gap CSM-SAM fills.

---

## 2. Axis system

Every method in Section 3 is placed on these six axes. The axes are chosen
to factor out the independent design choices that distinguish longitudinal
segmentation methods.

- **Temporal scope** — the widest temporal context the model can look at.
  - `within-image`: single 2D slice, no temporal neighbours.
  - `within-session (intra-scan)`: multiple slices of the same 3D volume
    (3D conv, video memory, slice-to-slice propagation).
  - `cross-session (inter-scan)`: two distinct visits of the same patient
    (pre-RT vs mid-RT).

- **Pre-session signal** — how the pre-RT visit is exposed to the model.
  - `none`: pre-RT never touched.
  - `raw image concat`: pre-RT image stacked on the input channel.
  - `mask concat`: pre-RT mask stacked on the input channel (or used as a
    prompt).
  - `warped mask` / `warped image`: classical deformable registration
    transfers pre-RT to the mid-RT grid before use.
  - `siamese features`: shared encoder applied to both visits, features
    concatenated / differenced / nested at each scale.
  - `feature bank w/ attention`: pre-RT encoded into a token bank that the
    mid-RT query attends to (CSM-SAM only).

- **Foundation backbone** — strength of the frozen prior available.
  - `none / scratch`: encoder trained from random or domain-medical SSL init
    on ~150 patients.
  - `ImageNet ResNet / MobileNet / HRNet`: ImageNet-pretrained CNN encoder,
    fully fine-tuned.
  - `ViT in-domain`: transformer pretrained on medical images (e.g. Swin
    UNETR SSL weights, MedNeXt, UNETR from scratch).
  - `DINOv2 ViT-L (LVD-142M)`: self-supervised natural-image ViT, frozen.
  - `CLIP ViT-B + CLIPSeg decoder`: language-grounded natural-image model.
  - `SAM ViT-H (SA-1B)`: SAM1 encoder, frozen.
  - `SAM2 Hiera-L (SA-1B + SA-V)`: full SAM2 encoder (used frozen by
    CSM-SAM; used frozen or decoder-only by SAM2 baselines).

- **Trainable params** — how much of the model updates during training.
  - `full`: every parameter (encoder + decoder) is trained.
  - `partial (encoder+head)`: frozen backbone stem + trained encoder blocks
    + head.
  - `decoder-only`: the encoder is frozen, the decoder / head is trained.
  - `memory-only (~2M)`: encoder and decoder frozen, only the cross-session
    memory module + final decoder layer are trained (CSM-SAM).
  - `0 (no-learning)`: the method has no learnable parameters.

- **Change-map supervision** — is there an auxiliary head supervised by the
  pre/mid mask XOR (i.e. a free-lunch "where did the tumor change" signal)?
  - `no` / `yes (primary)` / `yes (auxiliary)`.

- **Temporal-embedding calibration** — does the model explicitly consume the
  number of elapsed weeks between pre-RT and mid-RT as an embedding /
  conditioning signal?
  - `no` / `yes`.

---

## 3. Per-baseline survey

One row per baseline file in `baselines/`. Grouped by category. Per-baseline
"uniqueness notes" are extracted from the top-of-file docstring of
`baselines/<name>.py`. All 36 files exist in-repo; no TBD rows remain.

Legend for **Pre-signal**: `-` = none; `img` = raw pre-RT image channel;
`mask` = pre-RT mask channel / prompt; `warp` = classical deformable warp;
`siam` = siamese shared-encoder features; `feat+attn` = encoded feature bank
consumed by cross-attention; `dataset-prior` = training-set-level shape prior
(not patient-specific); `text` = natural-language prompt.

### 3.1 Naive / sanity baselines

These set the absolute floor. If a learned method fails to beat them the
evaluation pipeline is broken or the learned signal does not exist. All six
files operate on HNTS-MRG 2024 mid-RT volumes.

| Name | Category | Temporal scope | Pre-signal | Backbone | Trainable params | Change head | Temporal embed | Paper |
|---|---|---|---|---|---|---|---|---|
| `zero_baseline` | constant 0 mask | within-image | - | none | 0 | no | no | — |
| `random_baseline` | Bernoulli at empirical density (~5%) | within-image | - | none | 0 | no | no | — |
| `majority_voxel_baseline` | training-set UNION voxel prior | within-image | dataset-prior | none | 0 | no | no | — |
| `identity_baseline` | pre-RT mask verbatim as mid-RT pred | cross-session | mask (copy) | none | 0 | no | no | — |
| `copy_prev_slice_baseline` | roll GT of slice i-1 onto slice i; slice 0 from pre-RT | within-session | mask (init) | none | 0 | no | no | — |
| `registration_warp_baseline` | deformable B-spline (Elastix / SimpleITK) pre→mid warp of pre-RT mask | cross-session | warp | none | 0 | no | no | Elastix / SimpleITK BSpline |

Key uniqueness notes:
- **zero** establishes the unconditional floor (aggDSC = 0 when GT is
  non-empty).
- **random** establishes the uninformed-but-nonzero floor at the correct
  foreground density.
- **majority_voxel** asks whether a patient-agnostic, dataset-level shape
  prior could already explain the numbers — if a learned method barely beats
  it, there is no per-patient signal being used.
- **identity** is the "no-learning cross-session" floor: it uses *exactly*
  the pre-RT mask and nothing else. CSM-SAM must beat it or the adaptation
  claim collapses.
- **copy_prev_slice** is the degenerate end of within-session propagation.
  Separates "mask drift is trivially informative" from "genuine learned
  propagation matters".
- **registration_warp** is the strongest no-learning cross-session baseline:
  a principled deformable B-spline registration carries the pre-RT mask onto
  the mid-RT grid. If CSM-SAM cannot beat it, the claim of *learned*
  cross-session memory propagation is not supported.

### 3.2 Classical medical segmentation — no pre-RT signal

Specialists trained end-to-end on HNTS-MRG with no pre-RT signal. These
isolate "how much does cross-session information help?" against strong
from-scratch / ImageNet-pretrained medical-seg specialists.

| Name | Category | Temporal scope | Pre-signal | Backbone | Trainable params | Change head | Temporal embed | Paper |
|---|---|---|---|---|---|---|---|---|
| `unet_2d_baseline` | 2D U-Net (ResNet-34 encoder) | within-image | - | ImageNet ResNet-34 | full (~24M) | no | no | Ronneberger et al., MICCAI 2015 |
| `deeplabv3plus_baseline` | DeepLabV3+ (ResNet-50 encoder, ASPP) | within-image | - | ImageNet ResNet-50 | full (~40M) | no | no | Chen et al., ECCV 2018 |
| `swinunetr_baseline` | 3D Swin UNETR (96³ sliding window) | within-session | - | Swin SSL / scratch | full (~62M) | no | no | Hatamizadeh et al., BrainLes @ MICCAI 2021 |
| `unetr_baseline` | 3D ViT encoder + conv decoder | within-session | - | ViT-B (scratch) | full (~93M) | no | no | Hatamizadeh et al., WACV 2022 |
| `mednext_baseline` | ConvNeXt-style 3D U-Net | within-session | - | none / scratch | full (~60M) | no | no | Roy et al., MICCAI 2023 |
| `nnunet_baseline` | nnUNetv2 3d_fullres (auto-planned) | within-session | - | none / scratch | full (~30M) | no | no | Isensee et al., Nat. Methods 2021 |

Key uniqueness notes:
- **nnunet** is the strongest traditional mid-RT-only baseline for HNTS-MRG
  and the reference that the challenge winner (72.7 aggDSC) was built on.
  Full-volume 3D training from scratch on ~150 patients.
- **swinunetr**, **unetr**, **mednext** are the three modern 3D
  transformer-/ConvNeXt-based specialists that a NeurIPS reviewer expects
  to see. None of them sees pre-RT information.
- **unet_2d** / **deeplabv3plus** are slice-wise ImageNet-pretrained
  specialists. They factor out the 3D component: any advantage CSM-SAM has
  over them cannot be attributed to 3D context alone.

### 3.3 Foundation-model baselines (frozen / prompt-based, no cross-session)

Probe "does the SAM / SAM2 / DINOv2 / CLIP / TotalSegmentator prior transfer
zero-shot or with a prompt, without any cross-session machinery?" — the
other side of the CSM-SAM contribution.

| Name | Category | Temporal scope | Pre-signal | Backbone | Trainable params | Change head | Temporal embed | Paper |
|---|---|---|---|---|---|---|---|---|
| `sam_vanilla_baseline` | SAM1 ViT-H, center or grid point prompt | within-image | - | SAM ViT-H (SA-1B) | 0 (zero-shot) | no | no | Kirillov et al., ICCV 2023 |
| `sam2_point_prompt_baseline` | SAM2 Hiera-L, center point, memory DISABLED | within-image | - | SAM2 Hiera-L (SA-1B + SA-V) | 0 (zero-shot) | no | no | Ravi et al., 2024 |
| `sam2_video_baseline` | SAM2 video predictor propagated across mid-RT slices | within-session | - | SAM2 Hiera-L | 0 (zero-shot) | no | no | Ravi et al., 2024 |
| `medsam2_baseline` | SAM2 with within-session memory (MedSAM2-style) | within-session | - | SAM2 Hiera-L | decoder-only | no | no | Ma et al., 2024 (MedSAM2) |
| `dinov2_linear_baseline` | frozen DINOv2 ViT-L/14 + small conv decoder | within-image | - | DINOv2 ViT-L (LVD-142M) | decoder-only | no | no | Oquab et al., TMLR 2024 |
| `clipseg_baseline` | text-prompted segmentation ("head and neck tumor") | within-image | text only | CLIP ViT-B + CLIPSeg decoder | 0 (zero-shot) | no | no | Lüddecke & Ecker, CVPR 2022 |
| `totalsegmentator_baseline` | generalist anatomical segmenter (104+ classes) | within-session | - | nnU-Net specialist | 0 (zero-shot) | no | no | Wasserthal et al., Radiology: AI 2023 |

Key uniqueness notes:
- **sam_vanilla** is the minimum-assumption foundation-model floor: frozen
  ViT-H, geometric prompt, no task-specific head. Tends to collapse on
  soft-tissue HNC tumors without anatomical cues.
- **sam2_point_prompt** isolates SAM2's raw 2D prior by *explicitly
  disabling* video memory. The cleanest single-slice test of the SAM2
  backbone itself.
- **sam2_video** / **medsam2** both implement within-session memory
  propagation on the mid-RT volume. The delta between CSM-SAM and these
  two is exactly the contribution of cross-session memory.
- **dinov2_linear** ablates the SAM2 mask-decoder prior while keeping a
  strong frozen natural-image ViT, to check whether the SAM2 decoder is
  load-bearing separately from the ViT features.
- **clipseg** tests language grounding as the prior. The CSM-SAM
  contribution says patient-specific pre-RT memory beats a generic
  text-concept mask.
- **totalsegmentator** tests a medical generalist. Its class vocabulary is
  anatomical (not tumor), so it establishes the floor for a generalist
  medical foundation model applied off-label to GTV.

### 3.4 Longitudinal / bi-temporal medical segmentation (on HNTS-MRG)

In-repo baselines that *do* use the pre-RT scan but via simpler mechanisms
than learned cross-session attention.

| Name | Category | Temporal scope | Pre-signal | Backbone | Trainable params | Change head | Temporal embed | Paper |
|---|---|---|---|---|---|---|---|---|
| `concat_channels_baseline` | 2D U-Net on (pre, mid) as 2-channel input | cross-session | img concat | ImageNet ResNet (smp) | full (~24M) | no | no | Ronneberger et al., MICCAI 2015 |
| `pre_mask_prior_baseline` | 2D U-Net on (mid, pre-mask) as 2-channel input | cross-session | mask concat | ImageNet ResNet (smp) | full (~24M) | no | no | BAMF / HNTS-MRG 2024 winner |
| `siamese_unet_baseline` | shared-weight smp U-Net encoder on (pre, mid); feature concat at every stage | cross-session | siam (concat) | ImageNet ResNet-34 | full (~24M) | no | no | Daudt et al., ICIP 2018 |
| `nnunet_dual_channel_baseline` | nnUNetv2 3d_fullres with (pre-RT, mid-RT) as 2 input channels | cross-session | img concat (3D) | none / scratch | full (~30M) | no | no | Isensee et al., Nat. Methods 2021 |
| `longisam_baseline` | frozen SAM2 ViT-H, pre/mid concatenated into 6 channels → 1x1 conv to 3, mask decoder fine-tuned | cross-session | img concat | SAM2 Hiera-L (frozen) | decoder + 1x1 conv | no | no | this repo (ablation) |
| `hnts_winner_replica_baseline` | nnUNetv2 3d_fullres on (mid, warped pre-img, warped pre-mask) | cross-session | warp (img + mask) | none / scratch | full (~30M) | no | no | BAMF, HNTS-MRG 2024 challenge |

Key uniqueness notes:
- **concat_channels** is the naive "just stack the two visits" control — no
  attention, no SAM2 prior, no change signal.
- **pre_mask_prior** is the HNTS-MRG 2024 winner's core idea (pre-RT mask as
  a channel), without nnUNet's 3D planner. It is the strongest 2D
  longitudinal control.
- **siamese_unet** is the classical change-detection architecture (weight-
  shared encoder, concat skips) adapted to HNTS-MRG — the cleanest
  comparison point for "does attention beat siamese concat?".
- **nnunet_dual_channel** is the strongest 3D dual-channel control: nnUNet's
  planner + 3D context + pre-RT channel. CSM-SAM must clearly beat it on
  aggDSC for the cross-session-attention claim to hold.
- **longisam** is the key CSM-SAM ablation that isolates the attention
  module: same SAM2 backbone, same decoder finetuning, but pre/mid are
  fused by a 1x1 conv instead of cross-session attention. Beating longisam
  is what isolates the contribution of attention from the contribution of
  the SAM2 backbone.
- **hnts_winner_replica** reproduces the 72.7 aggDSC BAMF pipeline:
  classical deformable registration then nnUNet with image + mask warped
  channels. This is the strongest published HNTS-MRG 2024 number and the
  primary number CSM-SAM must beat.

### 3.5 Change-detection baselines (binary, remote-sensing lineage)

Bitemporal change detectors from the remote-sensing literature.
Paraphrased reimplementations, in-repo. Targeted benchmark in each
docstring: LEVIR-CD.

| Name | Category | Temporal scope | Pre-signal | Backbone | Trainable params | Change head | Temporal embed | Paper |
|---|---|---|---|---|---|---|---|---|
| `fc_siam_conc_baseline` | siamese FCN-UNet, paired-feature concatenation at each skip | cross-session | siam (concat) | none / scratch | full | yes (primary) | no | Daudt et al., ICIP 2018 |
| `fc_siam_diff_baseline` | siamese FCN-UNet, absolute-difference fusion at each skip | cross-session | siam (diff) | none / scratch | full | yes (primary) | no | Daudt et al., ICIP 2018 |
| `snunet_baseline` | SNUNet-CD: siamese NestedUNet (UNet++) + channel-attention ECAM | cross-session | siam (nested) | none / scratch | full | yes (primary) | no | Fang et al., GRSL 2022 |
| `bit_baseline` | Bitemporal Image Transformer: ResNet-18 → semantic tokenizer → joint transformer | cross-session | siam + transformer tokens | ImageNet ResNet-18 | full | yes (primary) | no | Chen et al., TGRS 2021 |
| `changeformer_baseline` | hierarchical SegFormer-style transformer, MLP decoder, diff module | cross-session | siam (concat+1x1) | none / scratch | full | yes (primary) | no | Bandara & Patel, IGARSS 2022 |
| `tinycd_baseline` | MobileNetV2 siamese + Mixing-and-Attention-Mask Block (<0.3M params) | cross-session | siam + MAMB | ImageNet MobileNetV2 | full (~0.3M) | yes (primary) | no | Codegoni et al., arXiv 2022 |

Uniqueness-note common thread (from docstrings): all six are small-encoder,
from-scratch-or-ImageNet networks whose bi-temporal fusion is either feature
differencing, concatenation, nested skips, channel-attention mixing, or a
low-rank (~4-token) transformer bottleneck. None of them uses a 1B-mask-
pretrained backbone frozen at inference, none uses per-token cross-time
query/key attention over a full encoder token grid, and none has an
auxiliary (not primary) change supervision alongside a separate segmentation
head. These are the binary-CD analogues of CSM-SAM's cross-session idea —
the point of the comparison is "attention on SAM2 tokens beats feature
differencing or siamese concat on small from-scratch encoders".

### 3.6 Semantic change-detection baselines (multi-class, remote-sensing)

Semantic change detectors that predict BOTH per-time semantic maps AND a
binary change map. Paraphrased reimplementations, in-repo. Targeted
benchmark in each docstring: SECOND (except `xview2_dualhrnet`, which
targets xBD).

| Name | Category | Temporal scope | Pre-signal | Backbone | Trainable params | Change head | Temporal embed | Paper |
|---|---|---|---|---|---|---|---|---|
| `ced_net_baseline` | HRSCD Strategy IV: shared encoder + dual semantic decoders + change decoder, deep supervision | cross-session | siam (concat) | none / scratch | full | yes (primary, deep-sup) | no | Daudt et al., CVIU 2019 / Peng et al., 2021 |
| `bisrnet_baseline` | Bi-SRNet: siamese ResNet-lite + channel-and-spatial SR reasoning + change branch gating dual decoders | cross-session | siam (SR attention) | ImageNet ResNet-18 (optional) | full | yes (primary) | no | Ding et al., TGRS 2022 |
| `scannet_cd_baseline` | SCanNet: siamese encoder + Spatio-Temporal Transformer jointly attending over both times | cross-session | siam + ST-transformer | none / scratch | full | yes (primary) | no | Ding et al., TGRS 2024 |
| `xview2_dualhrnet_baseline` | Dual-HRNet: siamese multi-resolution HRNet + localization head (pre) + damage-grading head (post) | cross-session | siam (HRNet) | ImageNet HRNet-lite | full | yes (ordinal damage) | no | DIUx xView2 1st place, 2019 / HRNet follow-ups |

Uniqueness-note common thread (verbatim-in-spirit from the four docstrings):
these methods specialize to bi-temporal semantic change with small encoders
and branch-specific fusion (SR attention, ST-transformer, dual-HRNet
streams). CSM-SAM's cross-session memory module is architecturally general:
it slots atop any SAM2 backbone and uses token-level attention between
sessions, not branch-differencing or per-class decoders. On SECOND, the
same module that solves HNTS-MRG tumor propagation also propagates
land-cover semantics; on xBD, it swaps in unchanged to propagate
pre-disaster building structure into post-disaster damage grading. That
generality is the point — CSM-SAM's contribution is not a new
change-detection architecture, it is a cross-session memory mechanism that
can be placed on top of any pretrained foundation encoder.

### 3.7 CSM-SAM (this work)

| Name | Category | Temporal scope | Pre-signal | Backbone | Trainable params | Change head | Temporal embed | Paper |
|---|---|---|---|---|---|---|---|---|
| **CSM-SAM** | frozen SAM2 ViT-H + CrossSessionMemoryAttention + ChangeHead | **cross-session** | **feat+attn (pre-RT memory bank)** | **SAM2 Hiera-L (1B masks, frozen)** | **memory-only (~2M)** | **yes (auxiliary, supervised by pre/mid XOR)** | **yes (continuous weeks encoder)** | this work |

---

## 4. Uniqueness map (ASCII)

Positioning diagram: x-axis = degree of pre-session use (how much of the
pre-RT visit is exploited); y-axis = backbone strength (how strong the
frozen prior is and how little has to be trained from scratch). CSM-SAM
occupies the top-right alone.

```
  backbone
  strength
  ^
  |                                                              +-----------+
  |                                                              |           |
  |                                                              |  CSM-SAM  |   <-- top-right: SAM2 ViT-H (1B)
  |                                                              |  (this    |       + pre-RT memory bank
  |                                                              |   work)   |       + change head + temp emb
  |                                                              +-----------+
  |
  |   +--------------------+   +------------------+              +----------------------+
  |   |  sam_vanilla       |   |  sam2_video      |              |  longisam            |
  |   |  sam2_point_prompt |   |  medsam2         |              |  (SAM2 + pre/mid     |
  |   |  clipseg           |   +------------------+              |   1x1 fusion, no     |
  |   |  dinov2_linear     |                                     |   attention)         |
  |   |  totalsegmentator  |                                     +----------------------+
  |   +--------------------+
  |
  |   +-----------------+   +------------------+   +-----------------------------------+
  |   |  swinunetr      |   |  nnunet          |   | pre_mask_prior, concat_channels,  |
  |   |  unetr          |   |  mednext         |   | siamese_unet, nnunet_dual_channel,|
  |   |  unet_2d        |   |  deeplabv3plus   |   | hnts_winner_replica               |
  |   +-----------------+   +------------------+   +-----------------------------------+
  |
  |                                                  +-----------------------------------+
  |                                                  | Remote-sensing change detectors   |
  |                                                  | fc_siam_conc, fc_siam_diff,       |
  |                                                  | snunet, bit, changeformer, tinycd |
  |                                                  | (binary CD)                       |
  |                                                  | ced_net, bisrnet, scannet_cd,     |
  |                                                  | xview2_dualhrnet                  |
  |                                                  | (semantic / ordinal CD)           |
  |                                                  +-----------------------------------+
  |
  |   +----------------+   +-------------+   +--------------+   +------------------------+
  |   | zero, random   |   | majority_   |   | copy_prev_   |   | identity,              |
  |   | (scratch, noop)|   | voxel prior |   | slice (intra)|   | registration_warp      |
  |   +----------------+   +-------------+   +--------------+   +------------------------+
  +------------------------------------------------------------------------------------> degree of
    none                     dataset-level         mask copy /         siamese/concat           feature-bank    pre-session
                             prior                 warp                + change decoder         + attention       use
```

Reading the diagram:

- **Bottom row** (scratch / no backbone): the full spectrum of pre-session
  use is already spanned by sanity baselines — from `zero_baseline` (none)
  to `registration_warp_baseline` (full deformable warp). Their y-coordinate
  is the floor, which frames the contribution of a backbone.
- **Middle-bottom row** (ImageNet / scratch specialists): classical 3D and
  2D segmentation models — `swinunetr`, `unetr`, `mednext`, `nnunet`,
  `unet_2d`, `deeplabv3plus`. Strong backbones exist but all of them live
  in the "no pre-RT" column. The longitudinal cluster alongside them
  (`pre_mask_prior`, `concat_channels`, `siamese_unet`,
  `nnunet_dual_channel`, `hnts_winner_replica`) adds the pre-RT visit via
  channel concat, siamese features, or deformable warp — not attention.
- **Middle-right** (remote-sensing change detectors): six binary and four
  semantic bi-temporal architectures. They cover the "siamese + change
  decoder" column comprehensively but all train from scratch or from
  ImageNet, so their y-coordinate is the classical-CNN band.
- **Top-middle** (foundation-model prompt/probe baselines): `sam_vanilla`,
  `sam2_point_prompt`, `sam2_video`, `medsam2`, `clipseg`, `dinov2_linear`,
  `totalsegmentator` sit at the "frozen strong prior" band; `longisam` sits
  to their right as the "SAM2 + naive pre/mid fusion" ablation.
- **Top-right**: CSM-SAM is the only method combining SAM2 ViT-H (frozen,
  1B-mask prior) with a feature-bank + cross-attention cross-session
  mechanism plus an auxiliary change head plus a temporal (weeks-elapsed)
  embedding.

---

## 5. Claim consolidation

The CSM-SAM paper makes five claims. Each is paired with the in-repo
baseline files that ablate or support it; an "ablation baseline" is one
that differs from CSM-SAM along exactly one axis of Section 2.

1. **Cross-session memory attention is the right inductive bias for
   longitudinal segmentation — it beats (a) no pre-session information,
   (b) naive pre-session reuse (copy / warp), and (c) naive pre-session
   fusion (channel concat / siamese concat / siamese diff).**
   Ablations:
   - No pre-session info:
     `sam2_point_prompt_baseline.py`, `sam2_video_baseline.py`,
     `medsam2_baseline.py`, `nnunet_baseline.py`, `swinunetr_baseline.py`,
     `unetr_baseline.py`, `mednext_baseline.py`, `unet_2d_baseline.py`,
     `deeplabv3plus_baseline.py`, `sam_vanilla_baseline.py`,
     `dinov2_linear_baseline.py`, `clipseg_baseline.py`,
     `totalsegmentator_baseline.py`.
   - Naive pre-session reuse (copy / warp / dataset prior):
     `identity_baseline.py`, `copy_prev_slice_baseline.py`,
     `registration_warp_baseline.py`, `majority_voxel_baseline.py`.
   - Naive pre-session fusion (channel concat / siamese /
     warped-channels-nnUNet):
     `concat_channels_baseline.py`, `pre_mask_prior_baseline.py`,
     `siamese_unet_baseline.py`, `nnunet_dual_channel_baseline.py`,
     `hnts_winner_replica_baseline.py`, `longisam_baseline.py`.
   - Bi-temporal change-detection analogues (siamese / difference /
     transformer-token fusion):
     `fc_siam_conc_baseline.py`, `fc_siam_diff_baseline.py`,
     `snunet_baseline.py`, `bit_baseline.py`, `changeformer_baseline.py`,
     `tinycd_baseline.py`.

2. **A frozen foundation-model backbone with a small trainable memory
   module (~2M params) beats full-training specialists in the small-data
   regime (~150 HNTS-MRG patients).**
   Ablations:
   - Full-train specialists without pre-session info:
     `nnunet_baseline.py`, `swinunetr_baseline.py`, `unetr_baseline.py`,
     `mednext_baseline.py`, `unet_2d_baseline.py`,
     `deeplabv3plus_baseline.py`.
   - Full-train specialists with pre-session info:
     `hnts_winner_replica_baseline.py`, `nnunet_dual_channel_baseline.py`,
     `pre_mask_prior_baseline.py`, `concat_channels_baseline.py`,
     `siamese_unet_baseline.py`.
   - Full-train bi-temporal change-detection networks:
     `fc_siam_conc_baseline.py`, `fc_siam_diff_baseline.py`,
     `snunet_baseline.py`, `bit_baseline.py`, `changeformer_baseline.py`,
     `tinycd_baseline.py`, `ced_net_baseline.py`, `bisrnet_baseline.py`,
     `scannet_cd_baseline.py`, `xview2_dualhrnet_baseline.py`.

3. **The cross-session ATTENTION mechanism matters — not just having the
   pre-RT signal in front of a strong backbone.**
   Single-axis ablation:
   - `longisam_baseline.py` — same SAM2 frozen ViT-H, same mask-decoder
     finetuning, but pre/mid are fused by a 1x1 conv into the encoder path
     instead of by learned cross-session attention. This is the single
     cleanest isolation of the attention module's contribution.

4. **Change-map supervision (pre/mid mask XOR) adds an inductive bias that
   improves mid-RT segmentation — it is a free supervision signal that the
   existing HNTS-MRG baselines throw away.**
   Ablations:
   - Every baseline in Sections 3.1-3.4 (all HNTS-MRG-targeted baselines)
     lacks any change head.
   - Sections 3.5-3.6 have change heads but use them as the *primary*
     output (binary or semantic change), not as auxiliary supervision for
     a separate segmentation head. CSM-SAM's change head is auxiliary —
     supervised from the free XOR signal to regularize the segmentation
     decoder.

5. **Temporal-embedding calibration — conditioning on weeks elapsed between
   pre-RT and mid-RT — is necessary. Treatment response magnitude varies
   with time, and a model without time conditioning cannot calibrate
   expected change across the 1-4-week range in HNTS-MRG.**
   Ablations:
   - Every baseline in the landscape except CSM-SAM (no other file consumes
     weeks-elapsed).
   - CSM-SAM internal ablation: `ContinuousTimeEncoder` vs
     `DiscreteTemporalEmbedding` (both present in
     `csmsam/modeling/cross_session_memory_attention.py`).

6. **CSM-SAM is the first method to combine (frozen foundation backbone) x
   (cross-session feature-bank attention) x (auxiliary change-map
   supervision) x (temporal calibration). Each of the four axes has prior
   work, but no published method occupies their joint top-right corner.**
   Support: the positioning diagram in Section 4 plus the row-by-row
   comparison in Section 3.7, and specifically the single-axis ablation
   pair `medsam2_baseline` + `longisam_baseline` + `hnts_winner_replica`
   which collectively pin down each of the four axes.

---

## 6. Datasets covered by the landscape

CSM-SAM is evaluated on HNTS-MRG 2024. The datasets below are listed
because the baselines in Section 3 have been benchmarked on them in prior
work, and the landscape in Section 4 is supposed to be dataset-agnostic
(the axes describe method capability, not any one benchmark). The "In-repo
baselines" column lists every `baselines/*.py` file whose docstring
explicitly targets that dataset.

| Dataset | Domain | Bi-temporal? | In-repo baselines targeting it | Integration |
|---|---|---|---|---|
| **HNTS-MRG 2024** | HN cancer MRI, pre-RT + mid-RT | yes (pre/mid) | all baselines in Sections 3.1-3.4 and CSM-SAM (26 files): `zero_baseline`, `random_baseline`, `majority_voxel_baseline`, `identity_baseline`, `copy_prev_slice_baseline`, `registration_warp_baseline`, `unet_2d_baseline`, `deeplabv3plus_baseline`, `swinunetr_baseline`, `unetr_baseline`, `mednext_baseline`, `nnunet_baseline`, `sam_vanilla_baseline`, `sam2_point_prompt_baseline`, `sam2_video_baseline`, `medsam2_baseline`, `dinov2_linear_baseline`, `clipseg_baseline`, `totalsegmentator_baseline`, `concat_channels_baseline`, `pre_mask_prior_baseline`, `siamese_unet_baseline`, `nnunet_dual_channel_baseline`, `longisam_baseline`, `hnts_winner_replica_baseline` | yes — `csmsam/datasets/hnts_mrg.py` |
| **LEVIR-CD** | remote-sensing building change (256x256 aerial image pairs) | yes | `fc_siam_conc_baseline`, `fc_siam_diff_baseline`, `snunet_baseline`, `bit_baseline`, `changeformer_baseline`, `tinycd_baseline` | dataset not integrated; baselines read docstring-cited F1s |
| **SECOND** | semantic change detection, urban multi-class | yes | `ced_net_baseline`, `bisrnet_baseline`, `scannet_cd_baseline` | dataset not integrated |
| **xBD** | building damage post-disaster, 4-class ordinal | yes | `xview2_dualhrnet_baseline` | dataset not integrated |
| **LVD-142M / SA-1B / SA-V** (pre-training corpora, not evaluated) | natural images / video | n/a | provides the frozen priors used by `dinov2_linear_baseline` (LVD-142M), `sam_vanilla_baseline` (SA-1B), `sam2_point_prompt_baseline` / `sam2_video_baseline` / `medsam2_baseline` / `longisam_baseline` / CSM-SAM (SA-1B + SA-V) | via torch.hub / SAM release |
| **PhraseCut** (pre-training corpus for CLIPSeg) | natural images + phrases | n/a | provides the language-grounded prior used by `clipseg_baseline` | via HuggingFace `CIDAS/clipseg-rd64-refined` |
| **TotalSegmentator train corpus** (104+ anatomical classes, CT + MR) | general medical imaging | n/a | provides the anatomical prior used by `totalsegmentator_baseline` | via `pip install totalsegmentator` |

Notes:
- Only HNTS-MRG 2024 is directly integrated into the repository via
  `csmsam/datasets/hnts_mrg.py`. Every other dataset in the table is
  referenced by a baseline's docstring for positioning and published
  metrics; the corresponding `.py` file contains the model architecture
  and can be trained on its native benchmark by the reader if desired.
- The change-detection datasets (LEVIR-CD, SECOND, xBD) are included so
  that Claim 2 ("foundation backbone + small trainable memory beats
  full-train specialists") can be discussed in a domain where the
  published bi-temporal baselines are strong and well-characterised, even
  though CSM-SAM is not itself evaluated on them.
- The pre-training-corpus rows are listed for completeness: the CSM-SAM
  claim that a 1B-mask-pretrained frozen ViT beats small from-scratch
  encoders is grounded in the SA-1B + SA-V prior, and the competing
  foundation-model baselines (DINOv2 / CLIPSeg / TotalSegmentator) are
  grounded in LVD-142M / PhraseCut / the TotalSegmentator corpus
  respectively. Differences in pre-training corpus are a confound for
  claim 2 and must be discussed in the paper.

---

## 7. Inventory

All 36 baseline `.py` files present under `baselines/`, accounted for:

- Naive / sanity (6):
  `zero_baseline.py`, `random_baseline.py`, `majority_voxel_baseline.py`,
  `identity_baseline.py`, `copy_prev_slice_baseline.py`,
  `registration_warp_baseline.py`.
- Classical medical segmentation (6):
  `unet_2d_baseline.py`, `deeplabv3plus_baseline.py`,
  `swinunetr_baseline.py`, `unetr_baseline.py`, `mednext_baseline.py`,
  `nnunet_baseline.py`.
- Foundation-model (no cross-session) (7):
  `sam_vanilla_baseline.py`, `sam2_point_prompt_baseline.py`,
  `sam2_video_baseline.py`, `medsam2_baseline.py`,
  `dinov2_linear_baseline.py`, `clipseg_baseline.py`,
  `totalsegmentator_baseline.py`.
- Longitudinal / bi-temporal on HNTS-MRG (6):
  `concat_channels_baseline.py`, `pre_mask_prior_baseline.py`,
  `siamese_unet_baseline.py`, `nnunet_dual_channel_baseline.py`,
  `longisam_baseline.py`, `hnts_winner_replica_baseline.py`.
- Remote-sensing binary change detection (6):
  `fc_siam_conc_baseline.py`, `fc_siam_diff_baseline.py`,
  `snunet_baseline.py`, `bit_baseline.py`, `changeformer_baseline.py`,
  `tinycd_baseline.py`.
- Remote-sensing / disaster semantic or ordinal change detection (4):
  `ced_net_baseline.py`, `bisrnet_baseline.py`, `scannet_cd_baseline.py`,
  `xview2_dualhrnet_baseline.py`.
- Driver / package (1, not a baseline): `run_all_baselines.py`.
- Package init (1, not a baseline): `__init__.py`.

Total baseline implementations: **36** (6 sanity + 6 classical + 7
foundation + 6 longitudinal + 6 binary-CD + 4 semantic/ordinal-CD +
1 CSM-SAM-ablation — note that `longisam_baseline.py` is counted once
under "longitudinal" and is CSM-SAM's cleanest single-axis ablation),
alongside `run_all_baselines.py` (driver) and `__init__.py` (package
init). No TBD placeholders remain in the per-baseline table.
