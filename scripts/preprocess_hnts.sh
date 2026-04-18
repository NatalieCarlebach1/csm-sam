#!/usr/bin/env bash
# Preprocess HNTS-MRG 2024 dataset from the raw Zenodo zip.
#
# Usage:
#   bash scripts/preprocess_hnts.sh
#   bash scripts/preprocess_hnts.sh --n_workers 8
#
# The script will:
#   1. Extract data/raw/HNTSMRG24_train.zip if needed
#   2. Run data/preprocess.py
#   3. Print elapsed time

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-/home/tals/miniconda3/envs/ns-sam3/bin/python}"

RAW_DIR="${PROJECT_DIR}/data/raw"
OUTPUT_DIR="${PROJECT_DIR}/data/processed"
ZIP_PATH="${RAW_DIR}/HNTSMRG24_train.zip"
EXTRACTED_DIR="${RAW_DIR}/HNTSMRG24_train"

echo "============================================================"
echo "HNTS-MRG 2024 Preprocessing Pipeline"
echo "============================================================"
echo "Project dir : ${PROJECT_DIR}"
echo "Python      : ${PYTHON}"
echo "Raw dir     : ${RAW_DIR}"
echo "Output dir  : ${OUTPUT_DIR}"
echo ""

# Step 1: Extract zip if needed
if [ -d "${EXTRACTED_DIR}" ]; then
    echo "[1/2] Already extracted: ${EXTRACTED_DIR}"
else
    if [ ! -f "${ZIP_PATH}" ]; then
        echo "ERROR: Neither extracted dir nor zip found."
        echo "  Expected: ${ZIP_PATH}"
        echo "  Download first: ${PYTHON} data/download_hnts_mrg.py --output_dir data/raw"
        exit 1
    fi
    echo "[1/2] Extracting ${ZIP_PATH} ..."
    START_EXTRACT=$(date +%s)
    unzip -q -o "${ZIP_PATH}" -d "${RAW_DIR}"
    END_EXTRACT=$(date +%s)
    echo "  Extraction took $((END_EXTRACT - START_EXTRACT))s"
fi

# Step 2: Run preprocessing
echo ""
echo "[2/2] Running preprocess.py ..."
START_PREPROCESS=$(date +%s)

"${PYTHON}" "${PROJECT_DIR}/data/preprocess.py" \
    --input_dir "${RAW_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"

END_PREPROCESS=$(date +%s)

echo ""
echo "============================================================"
echo "Total preprocessing time: $((END_PREPROCESS - START_PREPROCESS))s"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"
