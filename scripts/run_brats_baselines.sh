#!/bin/bash
# Run all BraTS-GLI baselines on rack-wolf-g01.
# Idempotent — skips baselines whose metrics.json already exists.
#
# Usage:
#   bash scripts/run_brats_baselines.sh
#
# Overrides:
#   DATA_DIR=/path/to/BraTS_GLI bash scripts/run_brats_baselines.sh
#   DEVICE=cpu bash scripts/run_brats_baselines.sh

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DATA_ROOT="/media/data1/natalie"
DATA_DIR="${DATA_DIR:-$DATA_ROOT/BraTS_GLI}"
SAM2_CKPT="${SAM2_CKPT:-$DATA_ROOT/checkpoints/sam2.1_hiera_large.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/results/brats_baselines}"
DEVICE="${DEVICE:-cuda}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"

# Activate the csmsam conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$DATA_ROOT/conda/envs/csmsam"

mkdir -p "$OUTPUT_DIR" logs

LOG_FILE="logs/brats_baselines_$(date +%Y%m%d_%H%M%S).log"
echo "============================================================"
echo " BraTS-GLI Baseline Sweep"
echo " $(date)"
echo " Data:    $DATA_DIR"
echo " Output:  $OUTPUT_DIR"
echo " Device:  $DEVICE"
echo " Log:     $LOG_FILE"
echo "============================================================"

python baselines/run_brats_baselines.py \
    --data_dir     "$DATA_DIR" \
    --sam2_checkpoint "$SAM2_CKPT" \
    --output_dir   "$OUTPUT_DIR" \
    --device       "$DEVICE" \
    --image_size   "$IMAGE_SIZE" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Done. Summary: $OUTPUT_DIR/summary.md"
