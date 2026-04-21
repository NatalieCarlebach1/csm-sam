#!/bin/bash
# One-time machine setup for CSM-SAM.
# Run this once on a new machine before using run_train.sh.
#
# Usage:
#   bash setup_machine.sh                          # basic setup
#   bash setup_machine.sh --s3_data s3://bucket/raw --s3_checkpoint s3://bucket/sam2.pt
#   bash setup_machine.sh --synthetic              # use synthetic data (no real dataset needed)
#
# Options:
#   --raw_dir        where to put raw HNTS-MRG data   (default: data/raw)
#   --processed_dir  where to put preprocessed data   (default: data/processed)
#   --ckpt_dir       where to put SAM2 checkpoint     (default: checkpoints/sam2)
#   --s3_data        s3://bucket/path  — download raw data from S3
#   --s3_checkpoint  s3://bucket/file  — download SAM2 checkpoint from S3
#   --synthetic      generate synthetic data instead of real dataset
#   --n_synthetic    number of synthetic patients     (default: 15)
#   --skip_install   skip pip install
#   --skip_preprocess skip preprocessing (if data/processed already exists)

set -e

RAW_DIR="data/raw"
PROCESSED_DIR="data/processed"
CKPT_DIR="checkpoints/sam2"
S3_DATA=""
S3_CKPT=""
SYNTHETIC=false
N_SYNTHETIC=15
SKIP_INSTALL=false
SKIP_PREPROCESS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw_dir)        RAW_DIR="$2";        shift 2 ;;
    --processed_dir)  PROCESSED_DIR="$2";  shift 2 ;;
    --ckpt_dir)       CKPT_DIR="$2";       shift 2 ;;
    --s3_data)        S3_DATA="$2";        shift 2 ;;
    --s3_checkpoint)  S3_CKPT="$2";        shift 2 ;;
    --synthetic)      SYNTHETIC=true;      shift   ;;
    --n_synthetic)    N_SYNTHETIC="$2";    shift 2 ;;
    --skip_install)   SKIP_INSTALL=true;   shift   ;;
    --skip_preprocess) SKIP_PREPROCESS=true; shift  ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "========================================================"
echo " CSM-SAM Machine Setup"
echo "========================================================"

# ── 1. Install Python package ─────────────────────────────────────────────────
if [ "$SKIP_INSTALL" = false ]; then
  echo ""
  echo "[1/4] Installing csmsam package..."
  pip install -e ".[dev]" --quiet
  echo "      Done."
else
  echo "[1/4] Skipping install."
fi

# ── 2. SAM2 checkpoint ────────────────────────────────────────────────────────
SAM2_CKPT="$CKPT_DIR/sam2.1_hiera_large.pt"
echo ""
echo "[2/4] SAM2 checkpoint..."
mkdir -p "$CKPT_DIR"

if [ -f "$SAM2_CKPT" ]; then
  echo "      Already exists: $SAM2_CKPT"
elif [ -n "$S3_CKPT" ]; then
  echo "      Downloading from $S3_CKPT"
  aws s3 cp "$S3_CKPT" "$SAM2_CKPT"
else
  echo "      Downloading from Meta (Hugging Face)..."
  pip install -q huggingface_hub
  python3 - <<'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(
    repo_id="facebook/sam2.1-hiera-large",
    filename="sam2.1_hiera_large.pt",
)
os.makedirs("checkpoints/sam2", exist_ok=True)
shutil.copy(path, "checkpoints/sam2/sam2.1_hiera_large.pt")
print(f"      Saved to checkpoints/sam2/sam2.1_hiera_large.pt")
PYEOF
fi

# ── 3. Dataset ────────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Dataset..."

if [ -d "$PROCESSED_DIR" ] && [ "$(ls -A $PROCESSED_DIR 2>/dev/null)" ]; then
  echo "      Preprocessed data already exists: $PROCESSED_DIR"
  SKIP_PREPROCESS=true
fi

if [ "$SYNTHETIC" = true ]; then
  echo "      Generating $N_SYNTHETIC synthetic patients..."
  python3 data/download_hnts_mrg.py --synthetic --n_synthetic "$N_SYNTHETIC" --output_dir "$PROCESSED_DIR"
  SKIP_PREPROCESS=true
elif [ -n "$S3_DATA" ] && [ "$SKIP_PREPROCESS" = false ]; then
  echo "      Downloading raw data from $S3_DATA → $RAW_DIR"
  mkdir -p "$RAW_DIR"
  aws s3 sync "$S3_DATA" "$RAW_DIR" --no-progress
elif [ ! -d "$RAW_DIR" ] || [ -z "$(ls -A $RAW_DIR 2>/dev/null)" ]; then
  echo ""
  echo "  !! No raw data found at $RAW_DIR"
  echo "     Options:"
  echo "       a) Download from Zenodo and extract to $RAW_DIR, then re-run this script"
  echo "       b) Pass --s3_data s3://your-bucket/raw to pull from S3"
  echo "       c) Pass --synthetic to use synthetic data for testing"
  echo ""
  echo "     Zenodo: https://zenodo.org/record/11829006"
  echo "     zenodo_get $RAW_DIR  (pip install zenodo-get)"
fi

if [ "$SKIP_PREPROCESS" = false ]; then
  echo "      Preprocessing raw data..."
  python3 data/preprocess.py \
    --input_dir "$RAW_DIR" \
    --output_dir "$PROCESSED_DIR" \
    --n_workers 4
fi

# ── 4. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Verifying setup..."

ERRORS=0

if [ ! -f "$SAM2_CKPT" ]; then
  echo "  !! MISSING: SAM2 checkpoint at $SAM2_CKPT"
  ERRORS=$((ERRORS + 1))
else
  echo "      SAM2 checkpoint: OK"
fi

if [ ! -d "$PROCESSED_DIR" ] || [ -z "$(ls -A $PROCESSED_DIR 2>/dev/null)" ]; then
  echo "  !! MISSING: preprocessed data at $PROCESSED_DIR"
  ERRORS=$((ERRORS + 1))
else
  N_PATIENTS=$(find "$PROCESSED_DIR" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l)
  echo "      Preprocessed data: OK ($N_PATIENTS patient dirs)"
fi

python3 -c "import csmsam; print('      csmsam package: OK')" 2>/dev/null || {
  echo "  !! csmsam package not importable — run: pip install -e .[dev]"
  ERRORS=$((ERRORS + 1))
}

echo ""
if [ "$ERRORS" -eq 0 ]; then
  echo "========================================================"
  echo " Setup complete. Ready to train!"
  echo ""
  echo " Single GPU:        bash run_train.sh --gpus 1"
  echo " Multi GPU (4):     bash run_train.sh --gpus 4"
  echo " 5-fold CV (4 GPU): bash run_train.sh --gpus 4 --cv"
  echo "========================================================"
else
  echo "========================================================"
  echo " Setup finished with $ERRORS issue(s). Fix above and re-run."
  echo "========================================================"
  exit 1
fi
