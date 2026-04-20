#!/bin/bash
# Fix SAM2 install and launch training on GPUs 4-7.
# Run from repo root: bash scripts/fix_and_train.sh

set -e

DATA_ROOT="/media/data1/natalie"
export TMPDIR="$DATA_ROOT/tmp"
export MPLCONFIGDIR="$DATA_ROOT/tmp/matplotlib"
export PIP_CACHE_DIR="$DATA_ROOT/pip_cache"
export CUDA_VISIBLE_DEVICES=4,5,6,7
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$DATA_ROOT/conda/envs/csmsam"

# ── Reinstall SAM2 from source ────────────────────────────────────────────
echo "[1/2] Installing SAM2 from source..."
SAM2_SRC="$DATA_ROOT/sam2_src"
if [ ! -d "$SAM2_SRC" ]; then
    git clone https://github.com/facebookresearch/sam2.git "$SAM2_SRC"
fi
pip install -q --cache-dir "$DATA_ROOT/pip_cache" --no-build-isolation -e "$SAM2_SRC"
# Copy SAM2.1 configs to package root so Hydra finds them via pkg://sam2
cp "$SAM2_SRC/sam2/configs/sam2.1/sam2.1_hiera_l.yaml" "$SAM2_SRC/sam2/sam2.1_hiera_l.yaml"
echo "  SAM2 installed from source OK"

# ── Launch training ───────────────────────────────────────────────────────
echo "[2/2] Launching training on GPUs 4-7..."
LOG="$DATA_ROOT/train.log"

nohup env TMPDIR="$DATA_ROOT/tmp" MPLCONFIGDIR="$DATA_ROOT/tmp/matplotlib" torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_brats.py \
    --config configs/brats.yaml \
    --data_dir "$DATA_ROOT/BraTS_GLI" \
    --output_dir "$DATA_ROOT/checkpoints/csmsam_brats" \
    --sam2_checkpoint "$DATA_ROOT/checkpoints/sam2.1_hiera_large.pt" \
    --sequence_train \
    > "$LOG" 2>&1 &

echo "Training running in background. Log: $LOG"
tail -f "$LOG"
