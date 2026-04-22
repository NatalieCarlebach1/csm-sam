#!/bin/bash
#SBATCH --job-name=csmsam-hnts-1024
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/nfs/tals/csm-sam/logs/csmsam_%j.out
#SBATCH --error=/home/nfs/tals/csm-sam/logs/csmsam_%j.err

set -euo pipefail

# ── Env ───────────────────────────────────────────────────────────────────────
source /home/nfs/tals/miniconda3/etc/profile.d/conda.sh
conda activate ns-sam3

cd /home/nfs/tals/csm-sam

# ── Config knobs (SOTA: SAM2-L @ 1024) ────────────────────────────────────────
CONFIG=configs/default.yaml
DATA_DIR=/home/nfs/tals/csm-sam/data/processed
SAM2_CKPT=/home/nfs/tals/csm-sam/checkpoints/sam2/sam2.1_hiera_large.pt
OUTPUT_DIR=/home/nfs/tals/csm-sam/checkpoints/csmsam_hnts_1024_${SLURM_JOB_ID}
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
echo " Config:     $CONFIG (image_size=1024, sam2_cfg=configs/sam2.1/sam2.1_hiera_l)"
echo " Data:       $DATA_DIR"
echo " Checkpoint: $SAM2_CKPT"
echo " Output:     $OUTPUT_DIR"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# ── Train ────────────────────────────────────────────────────────────────────
torchrun \
    --standalone \
    --nproc_per_node="$NPROC" \
    train.py \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --sam2_checkpoint "$SAM2_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --sequence_train \
    --no_wandb

echo ""
echo "Training finished. Artifacts in $OUTPUT_DIR"
