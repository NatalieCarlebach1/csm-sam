#!/bin/bash
#SBATCH --job-name=overfit4-lora
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00
#SBATCH --output=/home/nfs/tals/csm-sam/logs/overfit_%j.out
#SBATCH --error=/home/nfs/tals/csm-sam/logs/overfit_%j.err

set -euo pipefail

source /home/nfs/tals/miniconda3/etc/profile.d/conda.sh
conda activate ns-sam3

cd /home/nfs/tals/csm-sam

CONFIG=configs/overfit.yaml
DATA_DIR=/home/nfs/tals/csm-sam/data/processed
SAM2_CKPT=/home/nfs/tals/csm-sam/checkpoints/sam2/sam2.1_hiera_base_plus.pt
OUTPUT_DIR=/home/nfs/tals/csm-sam/checkpoints/overfit4_lora8_${SLURM_JOB_ID}

mkdir -p "$OUTPUT_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1

echo "=============================================================="
echo " Job:        $SLURM_JOB_ID on $SLURM_NODELIST"
echo " Mode:       overfit-4 capability test"
echo " Encoder:    DINOv2-base + LoRA(r=8) + Bridge"
echo " Config:     $CONFIG"
echo " Output:     $OUTPUT_DIR"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# Single GPU for overfit (4 patients split across 4 GPUs = 1/GPU → too small).
python train.py \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --sam2_checkpoint "$SAM2_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --overfit 4 \
    --no_wandb

echo ""
echo "Overfit test complete. Train DSC trajectory in $OUTPUT_DIR"
