#!/bin/bash
# One-shot setup + launch script for TAU rack-wolf-g01
# Run from the repo root:  bash scripts/setup_brats_server.sh

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "============================================================"
echo " CSM-SAM BraTS-GLI Setup — TAU rack-wolf-g01"
echo " $(date)"
echo "============================================================"

# ── 1. Conda environment ──────────────────────────────────────────────────
if conda info --envs | grep -q "^csmsam "; then
    echo "[1/6] conda env 'csmsam' already exists — activating"
else
    echo "[1/6] Creating conda env 'csmsam' (Python 3.10)"
    conda create -y -n csmsam python=3.10
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate csmsam

# ── 2. PyTorch (CUDA 12.1) ────────────────────────────────────────────────
echo "[2/6] Installing PyTorch with CUDA 12.1"
pip install --quiet torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; \
    print(f'  torch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"

# ── 3. SAM2 ───────────────────────────────────────────────────────────────
echo "[3/6] Installing SAM2"
if python -c "import sam2" 2>/dev/null; then
    echo "  SAM2 already installed"
else
    pip install --quiet "git+https://github.com/facebookresearch/sam2.git"
fi

# ── 4. Project dependencies ───────────────────────────────────────────────
echo "[4/6] Installing project dependencies"
pip install --quiet -e ".[dev]"
pip install --quiet huggingface_hub omegaconf tqdm

# ── 5. SAM2 checkpoint ────────────────────────────────────────────────────
echo "[5/6] Downloading SAM2-L checkpoint"
mkdir -p checkpoints/sam2
CKPT="checkpoints/sam2/sam2.1_hiera_large.pt"
if [ -f "$CKPT" ]; then
    echo "  Checkpoint already exists: $CKPT"
else
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
        -O "$CKPT"
    echo "  Saved to $CKPT"
fi

# ── 6. Download BraTS-GLI ─────────────────────────────────────────────────
echo "[6/6] Downloading BraTS-GLI 2024 (~17 GB)"
if [ -d "data/raw/BraTS_GLI/Training" ] && [ "$(ls -A data/raw/BraTS_GLI/Training 2>/dev/null)" ]; then
    echo "  Data already present at data/raw/BraTS_GLI/"
else
    # Pass your HF token here or export HF_TOKEN before running this script
    python data/download_brats_gli.py \
        --output_dir data/raw/BraTS_GLI \
        ${HF_TOKEN:+--token "$HF_TOKEN"}
fi

echo ""
echo "============================================================"
echo " Setup complete. Launching 8-GPU training..."
echo "============================================================"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_brats.py \
    --config configs/brats.yaml \
    --sequence_train \
    2>&1 | tee logs/brats_train_$(date +%Y%m%d_%H%M%S).log
