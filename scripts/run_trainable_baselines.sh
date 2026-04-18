#!/usr/bin/env bash
# ============================================================================
# Run all 10 trainable baselines for HNTS-MRG 2024.
#
# Each baseline is trained from scratch on the train split, validated every
# 5 epochs, and evaluated on the test split.  Results are written to
# results/experiments/trained_baselines/<name>/metrics.json.
#
# Idempotent: skips baselines whose metrics.json already exists.
# Timeout: 3600s per baseline.
# Expected total: ~2-5 hours on RTX 4070.
#
# Usage:
#   bash scripts/run_trainable_baselines.sh
#   DEVICE=cpu bash scripts/run_trainable_baselines.sh
# ============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/tals/miniconda3/envs/ns-sam3/bin/python}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-3}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TIMEOUT=3600

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DATA_DIR="${PROJECT_ROOT}/data/processed"
RESULTS_ROOT="${PROJECT_ROOT}/results/experiments/trained_baselines"

mkdir -p "${RESULTS_ROOT}"

# Track results for summary
declare -A RESULTS
TOTAL=0
SKIPPED=0
PASSED=0
FAILED=0

run_baseline() {
    local name="$1"
    local script="$2"
    shift 2
    local extra_args=("$@")

    TOTAL=$((TOTAL + 1))
    local out_dir="${RESULTS_ROOT}/${name}"

    if [ -f "${out_dir}/metrics.json" ]; then
        echo "[SKIP] ${name} -- metrics.json already exists"
        SKIPPED=$((SKIPPED + 1))
        # Extract aggDSC from existing results
        local agg
        agg=$(${PYTHON} -c "
import json, sys
with open('${out_dir}/metrics.json') as f:
    d = json.load(f)
agg = d.get('aggregate', {}).get('agg_dsc_mean', 0)
print(f'{agg:.4f}')
" 2>/dev/null || echo "N/A")
        RESULTS["${name}"]="${agg}"
        return 0
    fi

    echo ""
    echo "================================================================"
    echo "[RUN] ${name}"
    echo "================================================================"

    mkdir -p "${out_dir}"

    if timeout "${TIMEOUT}" ${PYTHON} "${PROJECT_ROOT}/${script}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${out_dir}" \
        --split test \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --image_size "${IMAGE_SIZE}" \
        --device "${DEVICE}" \
        "${extra_args[@]}" \
        2>&1 | tee "${out_dir}/stdout.log"; then

        echo "OK" > "${out_dir}/status.txt"
        PASSED=$((PASSED + 1))

        local agg
        agg=$(${PYTHON} -c "
import json
with open('${out_dir}/metrics.json') as f:
    d = json.load(f)
agg = d.get('aggregate', {}).get('agg_dsc_mean', 0)
print(f'{agg:.4f}')
" 2>/dev/null || echo "N/A")
        RESULTS["${name}"]="${agg}"
    else
        echo "FAILED (exit=$?)" > "${out_dir}/status.txt"
        FAILED=$((FAILED + 1))
        RESULTS["${name}"]="FAILED"
        echo "[FAIL] ${name}"
    fi
}

# ============================================================================
# 1. Mid-RT only baselines (uses_pre=False)
# ============================================================================

run_baseline "unet2d" \
    "baselines/unet_2d_baseline.py" \
    --batch_size "${BATCH_SIZE}"

run_baseline "deeplabv3plus" \
    "baselines/deeplabv3plus_baseline.py" \
    --batch_size "${BATCH_SIZE}"

run_baseline "swinunetr" \
    "baselines/swinunetr_baseline.py" \
    --batch_size "${BATCH_SIZE}"

run_baseline "unetr" \
    "baselines/unetr_baseline.py" \
    --batch_size "${BATCH_SIZE}"

run_baseline "mednext" \
    "baselines/mednext_baseline.py" \
    --batch_size "${BATCH_SIZE}"

# ============================================================================
# 2. Longitudinal baselines (uses_pre=True)
# ============================================================================

run_baseline "concat_channels" \
    "baselines/concat_channels_baseline.py" \
    --batch_size "${BATCH_SIZE}"

run_baseline "pre_mask_prior" \
    "baselines/pre_mask_prior_baseline.py" \
    --batch_size "${BATCH_SIZE}"

run_baseline "siamese_unet" \
    "baselines/siamese_unet_baseline.py" \
    --batch_size "${BATCH_SIZE}"

# ============================================================================
# 3. SAM2-based baseline (uses_pre=True, lower LR, smaller batch)
# ============================================================================

SAM2_CKPT="${PROJECT_ROOT}/checkpoints/sam2/sam2.1_hiera_large.pt"

run_baseline "longisam" \
    "baselines/longisam_baseline.py" \
    --sam2_checkpoint "${SAM2_CKPT}" \
    --batch_size 4 \
    --lr 1e-4 \
    --image_size 1024

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "================================================================"
echo "TRAINABLE BASELINES SUMMARY"
echo "================================================================"
printf "%-25s %s\n" "Baseline" "aggDSC"
printf "%-25s %s\n" "-------------------------" "------"

for name in unet2d deeplabv3plus swinunetr unetr mednext \
            concat_channels pre_mask_prior siamese_unet longisam; do
    val="${RESULTS[${name}]:-N/A}"
    printf "%-25s %s\n" "${name}" "${val}"
done

echo ""
echo "Total: ${TOTAL} | Passed: ${PASSED} | Skipped: ${SKIPPED} | Failed: ${FAILED}"
echo "Results: ${RESULTS_ROOT}"
echo "================================================================"
