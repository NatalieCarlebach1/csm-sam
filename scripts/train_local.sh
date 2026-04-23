#!/bin/bash
# Local CSM-SAM training (no SLURM, single GPU, bs=2).
# Mirrors slurm/train_single.sbatch but runs directly via python.
#
# Usage:
#   bash scripts/train_local.sh
#   BATCH_SIZE=4 bash scripts/train_local.sh         # override bs
#   DEVICE=cpu  bash scripts/train_local.sh          # CPU run
#   SAM2_CKPT=checkpoints/sam2/sam2.1_hiera_large.pt bash scripts/train_local.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/default.yaml}"
DATA_DIR="${DATA_DIR:-data/processed}"
SAM2_CKPT="${SAM2_CKPT:-checkpoints/sam2/sam2.1_hiera_base_plus.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/csmsam_local_$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"

mkdir -p "$OUTPUT_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HYDRA_FULL_ERROR=1

echo "=============================================================="
echo " Local CSM-SAM run (no SLURM)"
echo " Config:      $CONFIG"
echo " Data:        $DATA_DIR"
echo " Checkpoint:  $SAM2_CKPT"
echo " Output:      $OUTPUT_DIR"
echo " Batch size:  $BATCH_SIZE"
echo " Workers:     $NUM_WORKERS"
echo " Prefetch:    $PREFETCH_FACTOR"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "(no GPU visible)"

python train.py \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --sam2_checkpoint "$SAM2_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --prefetch_factor "$PREFETCH_FACTOR" \
    --no_sequence_train \
    --no_wandb

echo ""
echo "Training finished. Artifacts in $OUTPUT_DIR"
