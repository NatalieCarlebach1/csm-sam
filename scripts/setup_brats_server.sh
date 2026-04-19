#!/bin/bash
# One-shot setup + launch script for TAU rack-wolf-g01
# All conda/pip/data goes to /media/data1/natalie to avoid home disk full.
#
# Run from the repo root:
#   bash scripts/setup_brats_server.sh

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# ── All heavy I/O redirected to data1 ─────────────────────────────────────
DATA_ROOT="/media/data1/natalie"
CONDA_ENVS="$DATA_ROOT/conda/envs"
CONDA_PKGS="$DATA_ROOT/conda/pkgs"
PIP_CACHE="$DATA_ROOT/pip_cache"
TMPDIR="$DATA_ROOT/tmp"
BRATS_DIR="$DATA_ROOT/BraTS_GLI"
SAM2_CKPT="$DATA_ROOT/checkpoints/sam2.1_hiera_large.pt"
LOGS_DIR="$REPO_DIR/logs"

mkdir -p "$CONDA_ENVS" "$CONDA_PKGS" "$PIP_CACHE" "$TMPDIR" \
         "$DATA_ROOT/checkpoints" "$LOGS_DIR"

export TMPDIR
export PIP_CACHE_DIR="$PIP_CACHE"
export CONDA_PKGS_DIRS="$CONDA_PKGS"
export HF_HOME="$DATA_ROOT/hf_cache"
export HF_DATASETS_CACHE="$DATA_ROOT/hf_cache/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

echo "============================================================"
echo " CSM-SAM BraTS-GLI Setup — TAU rack-wolf-g01"
echo " $(date)"
echo " All data/envs → $DATA_ROOT"
echo "============================================================"

# ── 1. Conda environment ──────────────────────────────────────────────────
conda config --add pkgs_dirs  "$CONDA_PKGS"  2>/dev/null || true
conda config --add envs_dirs  "$CONDA_ENVS"  2>/dev/null || true

ENV_PATH="$CONDA_ENVS/csmsam"
if [ -d "$ENV_PATH" ]; then
    echo "[1/6] conda env already exists: $ENV_PATH"
else
    echo "[1/6] Creating conda env 'csmsam' (Python 3.10) in $CONDA_ENVS"
    conda create -y -p "$ENV_PATH" python=3.10
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"
echo "  Active env: $(which python)"

# ── 2. PyTorch (CUDA 12.1) ────────────────────────────────────────────────
echo "[2/6] Installing PyTorch with CUDA 12.1"
pip install --quiet --cache-dir "$PIP_CACHE" \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'  torch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')
"

# ── 3. SAM2 (must install from source so config files are available) ──────
echo "[3/6] Installing SAM2 from source"
SAM2_SRC="$DATA_ROOT/sam2_src"
if python -c "from sam2.build_sam import build_sam2" 2>/dev/null; then
    echo "  SAM2 already installed"
else
    if [ ! -d "$SAM2_SRC" ]; then
        git clone https://github.com/facebookresearch/sam2.git "$SAM2_SRC"
    fi
    pip install --quiet --cache-dir "$PIP_CACHE" -e "$SAM2_SRC"
fi

# ── 4. Project dependencies ───────────────────────────────────────────────
echo "[4/6] Installing project dependencies"
pip install --quiet --cache-dir "$PIP_CACHE" -e ".[dev]"
pip install --quiet --cache-dir "$PIP_CACHE" huggingface_hub omegaconf tqdm

# ── 5. SAM2 checkpoint ────────────────────────────────────────────────────
echo "[5/6] Downloading SAM2-L checkpoint → $SAM2_CKPT"
if [ -f "$SAM2_CKPT" ]; then
    echo "  Already exists"
else
    wget --show-progress \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
        -O "$SAM2_CKPT"
fi

# ── 6. Download BraTS-GLI ─────────────────────────────────────────────────
echo "[6/6] Downloading BraTS-GLI 2024 (~17 GB) → $BRATS_DIR"
if [ -d "$BRATS_DIR/Training" ] && [ "$(ls -A "$BRATS_DIR/Training" 2>/dev/null)" ]; then
    echo "  Data already present"
else
    python data/download_brats_gli.py \
        --output_dir "$BRATS_DIR" \
        ${HF_TOKEN:+--token "$HF_TOKEN"}
fi

echo ""
echo "============================================================"
echo " Setup complete. Launching 4-GPU training (GPUs 4-7)..."
echo "============================================================"
echo ""

# ── Launch training ───────────────────────────────────────────────────────
LOG_FILE="$LOGS_DIR/brats_train_$(date +%Y%m%d_%H%M%S).log"
echo "Log: $LOG_FILE"

# GPUs 4-7 for training; GPUs 0-3 reserved for baselines
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_brats.py \
    --config configs/brats.yaml \
    --data_dir "$BRATS_DIR" \
    --output_dir "$DATA_ROOT/checkpoints/csmsam_brats" \
    --sam2_checkpoint "$SAM2_CKPT" \
    --sequence_train \
    2>&1 | tee "$LOG_FILE"
