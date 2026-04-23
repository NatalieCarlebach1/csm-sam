#!/bin/bash
#SBATCH --job-name=ralmsam-dino-4gpu
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/nfs/tals/csm-sam/logs/ralmsam_%j.out
#SBATCH --error=/home/nfs/tals/csm-sam/logs/ralmsam_%j.err

set -euo pipefail

# ── Env ───────────────────────────────────────────────────────────────────────
source /home/nfs/tals/miniconda3/etc/profile.d/conda.sh
conda activate ns-sam3

cd /home/nfs/tals/csm-sam

# ── Config knobs ──────────────────────────────────────────────────────────────
# configs/default.yaml defaults on tals-local:
#   encoder_type: dino_sam       (frozen DINOv2-base + trainable Bridge)
#   sequence_mode: true, sequence_length: 4
#   multimask_output: false  (lambda_iou=0)
# SAM2 checkpoint is still needed: its prompt_encoder (for sparse placeholder)
# and mask_decoder (trainable at lr*0.1) are kept.
CONFIG=configs/default.yaml
DATA_DIR=/home/nfs/tals/csm-sam/data/processed
SAM2_CKPT=/home/nfs/tals/csm-sam/checkpoints/sam2/sam2.1_hiera_large.pt
OUTPUT_DIR=/home/nfs/tals/csm-sam/checkpoints/ralmsam_dino_4gpu_${SLURM_JOB_ID}
NPROC=4                                   # matches --gres=gpu:4

mkdir -p "$OUTPUT_DIR"

# ── Runtime ──────────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HYDRA_FULL_ERROR=1                 # clearer SAM2 hydra errors

echo "=============================================================="
echo " Job:        $SLURM_JOB_ID on $SLURM_NODELIST"
echo " GPUs:       $NPROC (B200)"
echo " Encoder:    DINOv2-base @ 518 + Bridge (see configs/default.yaml)"
echo " Mode:       sequence training (S=4)"
echo " Config:     $CONFIG"
echo " Data:       $DATA_DIR"
echo " SAM2 ckpt:  $SAM2_CKPT  (prompt_enc frozen, mask_decoder trainable lr×0.1)"
echo " Output:     $OUTPUT_DIR"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# ── Train ────────────────────────────────────────────────────────────────────
# Sequence mode (config default) stays on — without it the within-session
# memory M_mid never accumulates across slices and the gate doesn't learn.
# Pass --no_sequence_train only for an ablation row.
torchrun \
    --standalone \
    --nproc_per_node="$NPROC" \
    train.py \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --sam2_checkpoint "$SAM2_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

echo ""
echo "Training finished. Artifacts in $OUTPUT_DIR"
