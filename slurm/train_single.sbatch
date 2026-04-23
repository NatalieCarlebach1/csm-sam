#!/bin/bash
#SBATCH --job-name=ralmsam-dino
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/nfs/tals/csm-sam/logs/ralmsam_%j.out
#SBATCH --error=/home/nfs/tals/csm-sam/logs/ralmsam_%j.err

set -euo pipefail

source /home/nfs/tals/miniconda3/etc/profile.d/conda.sh
conda activate ns-sam3

cd /home/nfs/tals/csm-sam

# configs/default.yaml defaults to encoder_type=dino_sam + sequence_mode=true.
# SAM2 checkpoint is still needed (prompt_encoder + mask_decoder are used).
CONFIG=configs/default.yaml
DATA_DIR=/home/nfs/tals/csm-sam/data/processed
SAM2_CKPT=/home/nfs/tals/csm-sam/checkpoints/sam2/sam2.1_hiera_base_plus.pt
OUTPUT_DIR=/home/nfs/tals/csm-sam/checkpoints/ralmsam_dino_${SLURM_JOB_ID}

mkdir -p "$OUTPUT_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1

echo "=============================================================="
echo " Job:        $SLURM_JOB_ID on $SLURM_NODELIST"
echo " Encoder:    DINOv2-base @ 518 + Bridge (galash-style)"
echo " Decoder:    SAM2 ($SAM2_CKPT), mask_decoder trainable at lr×0.1"
echo " Mode:       sequence training (S=4), 1 GPU"
echo " Data:       $DATA_DIR"
echo " Output:     $OUTPUT_DIR"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

python train.py \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --sam2_checkpoint "$SAM2_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

echo ""
echo "Training finished. Artifacts in $OUTPUT_DIR"
