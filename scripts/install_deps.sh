#!/usr/bin/env bash
# Install all dependencies for CSM-SAM
# Run from the project root: bash scripts/install_deps.sh

set -e

echo "=== CSM-SAM Dependency Installation ==="

# 1. PyTorch (adjust CUDA version as needed)
echo ""
echo "[1/5] Installing PyTorch..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 2. SAM2 (Meta)
echo ""
echo "[2/5] Installing SAM2..."
if [ ! -d "sam2" ]; then
    git clone https://github.com/facebookresearch/sam2.git
fi
cd sam2 && pip install -e . && cd ..

# 3. MedSAM2 (bowang-lab)
echo ""
echo "[3/5] Installing MedSAM2..."
if [ ! -d "MedSAM2" ]; then
    git clone https://github.com/bowang-lab/MedSAM2.git
fi
cd MedSAM2 && pip install -e . && cd ..

# 4. CSM-SAM
echo ""
echo "[4/5] Installing CSM-SAM..."
pip install -e ".[dev]"

# 5. Additional requirements
echo ""
echo "[5/5] Installing additional requirements..."
pip install -r requirements.txt

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next: bash scripts/setup_checkpoints.sh"
