#!/usr/bin/env bash
# Full environment setup for CSM-SAM
# Run from the project root: bash scripts/setup_env.sh

set -e

echo "=================================================="
echo "CSM-SAM Environment Setup"
echo "=================================================="

# ─── 1. Create conda environment ─────────────────────
echo ""
echo "[1/5] Creating conda environment 'csmsam'..."
conda env create -f environment.yml
echo "Environment created."

# Activate (this only works if this script is sourced, not executed)
# For non-interactive shells, manually run: conda activate csmsam
eval "$(conda shell.bash hook)"
conda activate csmsam

# ─── 2. Install SAM2 from source ─────────────────────
echo ""
echo "[2/5] Installing SAM2 (Meta)..."
if [ ! -d "sam2" ]; then
    git clone https://github.com/facebookresearch/sam2.git
fi
cd sam2
pip install -e .
cd ..

# ─── 3. Install MedSAM2 from source ──────────────────
echo ""
echo "[3/5] Installing MedSAM2 (bowang-lab)..."
if [ ! -d "MedSAM2" ]; then
    git clone https://github.com/bowang-lab/MedSAM2.git
fi
cd MedSAM2
pip install -e .
cd ..

# ─── 4. Install CSM-SAM (this repo) ──────────────────
echo ""
echo "[4/5] Installing CSM-SAM..."
pip install -e ".[dev]"

# ─── 5. Download SAM2 checkpoint ─────────────────────
echo ""
echo "[5/5] Downloading SAM2.1 Hiera-Large checkpoint..."
mkdir -p checkpoints/sam2
wget -q --show-progress \
    -P checkpoints/sam2 \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

echo ""
echo "=================================================="
echo "Setup complete!"
echo ""
echo "Activate environment:"
echo "  conda activate csmsam"
echo ""
echo "Next steps:"
echo "  # Test with synthetic data:"
echo "  python data/download_hnts_mrg.py --synthetic --n_synthetic 15"
echo "  python train.py --config configs/default.yaml --data_dir data/processed"
echo ""
echo "  # Real data (register at https://hnts-mrg.grand-challenge.org/):"
echo "  zenodo_get 11829006 -o data/raw/"
echo "  python data/preprocess.py --input_dir data/raw --output_dir data/processed"
echo "  python train.py --config configs/default.yaml"
echo "=================================================="
