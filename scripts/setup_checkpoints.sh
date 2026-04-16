#!/usr/bin/env bash
# Download SAM2 model checkpoints

set -e

CHECKPOINT_DIR="checkpoints/sam2"
mkdir -p "$CHECKPOINT_DIR"

echo "Downloading SAM2 checkpoints..."

# SAM2.1 Hiera Large (default for CSM-SAM)
wget -q --show-progress \
    -P "$CHECKPOINT_DIR" \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

# Optional: smaller model for testing
# wget -q --show-progress \
#     -P "$CHECKPOINT_DIR" \
#     "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"

echo "Checkpoints downloaded to $CHECKPOINT_DIR/"
echo ""
echo "Next steps:"
echo "  1. python data/download_hnts_mrg.py --output_dir data/raw"
echo "  2. python data/preprocess.py --input_dir data/raw --output_dir data/processed"
echo "  3. python train.py --config configs/default.yaml"
