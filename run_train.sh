#!/bin/bash
# Usage:
#   bash run_train.sh --gpus 1
#   bash run_train.sh --gpus 4
#   bash run_train.sh --gpus 4 --cv
#
# Optional overrides:
#   --data_dir      path to preprocessed data   (default: data/processed)
#   --checkpoint    path to SAM2 checkpoint      (default: checkpoints/sam2/sam2.1_hiera_large.pt)
#   --output_dir    where to save results        (default: checkpoints/csmsam_hnts)
#   --config        config file                  (default: configs/default.yaml)
#   --n_folds       number of CV folds           (default: 5, only used with --cv)

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
GPUS=1
CV=false
DATA_DIR="data/processed"
SAM2_CKPT="checkpoints/sam2/sam2.1_hiera_large.pt"
OUTPUT_DIR="checkpoints/csmsam_hnts"
CONFIG="configs/default.yaml"
N_FOLDS=5

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)        GPUS="$2";       shift 2 ;;
    --cv)          CV=true;         shift   ;;
    --data_dir)    DATA_DIR="$2";   shift 2 ;;
    --checkpoint)  SAM2_CKPT="$2";  shift 2 ;;
    --output_dir)  OUTPUT_DIR="$2"; shift 2 ;;
    --config)      CONFIG="$2";     shift 2 ;;
    --n_folds)     N_FOLDS="$2";    shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Launcher helper ───────────────────────────────────────────────────────────
run() {
  local extra_args="$1"
  if [ "$GPUS" -gt 1 ]; then
    torchrun --nproc_per_node="$GPUS" train.py \
      --config "$CONFIG" \
      --data_dir "$DATA_DIR" \
      --sam2_checkpoint "$SAM2_CKPT" \
      $extra_args
  else
    python train.py \
      --config "$CONFIG" \
      --data_dir "$DATA_DIR" \
      --sam2_checkpoint "$SAM2_CKPT" \
      $extra_args
  fi
}

# ── Run ───────────────────────────────────────────────────────────────────────
if [ "$CV" = true ]; then
  echo "Running ${N_FOLDS}-fold cross-validation on ${GPUS} GPU(s)"
  for fold in $(seq 0 $((N_FOLDS - 1))); do
    echo ""
    echo "======== Fold ${fold}/${N_FOLDS} ========"
    run "--output_dir ${OUTPUT_DIR}/fold${fold} --fold ${fold} --n_folds ${N_FOLDS}"
  done
  echo ""
  echo "All folds done. Results in ${OUTPUT_DIR}/"
else
  echo "Running single training on ${GPUS} GPU(s)"
  run "--output_dir ${OUTPUT_DIR}"
fi
