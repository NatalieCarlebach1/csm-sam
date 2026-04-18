#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_foundation_baselines.sh
# Runs the 7 foundation-model zero-shot baselines on REAL preprocessed
# HNTS-MRG data and produces a summary table.
#
# Usage:
#   bash scripts/run_foundation_baselines.sh
#   DEVICE=cpu bash scripts/run_foundation_baselines.sh
#   SPLIT=val bash scripts/run_foundation_baselines.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-/home/tals/miniconda3/envs/ns-sam3/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASELINES_DIR="${REPO_ROOT}/baselines"
DATA_DIR="${REPO_ROOT}/data/processed"
RESULTS_DIR="${REPO_ROOT}/results/hnts_foundation_baselines"
SAM2_CKPT="${REPO_ROOT}/checkpoints/sam2/sam2.1_hiera_large.pt"
SAM1_CKPT="${REPO_ROOT}/checkpoints/sam/sam_vit_h_4b8939.pth"

DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-test}"
TIMEOUT_SEC=1200

mkdir -p "${RESULTS_DIR}"

# ── SAM checkpoint warnings ───────────────────────────────────────────────
if [[ ! -f "${SAM2_CKPT}" ]]; then
    echo "=================================================================="
    echo "WARNING: SAM2 checkpoint not found at:"
    echo "  ${SAM2_CKPT}"
    echo ""
    echo "SAM2-based baselines will FAIL at init (no random fallback)."
    echo "To download, run:"
    echo "  mkdir -p checkpoints/sam2"
    echo "  wget -O checkpoints/sam2/sam2.1_hiera_large.pt \\"
    echo "    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    echo "=================================================================="
fi

if [[ ! -f "${SAM1_CKPT}" ]]; then
    echo "=================================================================="
    echo "WARNING: SAM1 checkpoint not found at:"
    echo "  ${SAM1_CKPT}"
    echo ""
    echo "sam_vanilla baseline will FAIL at init (no random fallback)."
    echo "To download, run:"
    echo "  mkdir -p checkpoints/sam"
    echo "  wget -O checkpoints/sam/sam_vit_h_4b8939.pth \\"
    echo "    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    echo "=================================================================="
fi

# ── Baseline definitions ──────────────────────────────────────────────────
# Format: name|script|extra_flags
BASELINES=(
    "sam_vanilla|sam_vanilla_baseline.py|--sam_checkpoint ${SAM1_CKPT} --model_type vit_h"
    "sam2_point_prompt|sam2_point_prompt_baseline.py|--sam2_checkpoint ${SAM2_CKPT} --model_cfg configs/sam2.1/sam2.1_hiera_l"
    "sam2_video|sam2_video_baseline.py|--sam2_checkpoint ${SAM2_CKPT} --model_cfg configs/sam2.1/sam2.1_hiera_l"
    "medsam2|medsam2_baseline.py|--sam2_checkpoint ${SAM2_CKPT}"
    "dinov2_linear|dinov2_linear_baseline.py|--model_name dinov2_vitl14"
    "clipseg|clipseg_baseline.py|--model_id CIDAS/clipseg-rd64-refined"
    "totalsegmentator|totalsegmentator_baseline.py|--task head_neck_muscles"
)

# ── Runner ─────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
SKIP=0
declare -A RUNTIMES
declare -A STATUSES

for entry in "${BASELINES[@]}"; do
    IFS='|' read -r NAME SCRIPT EXTRA_FLAGS <<< "${entry}"

    OUT_DIR="${RESULTS_DIR}/${NAME}"
    METRICS_FILE="${OUT_DIR}/metrics.json"
    LOG_FILE="${OUT_DIR}/stdout.log"
    STATUS_FILE="${OUT_DIR}/status.txt"

    mkdir -p "${OUT_DIR}"

    # Idempotent: skip if metrics.json already exists
    if [[ -f "${METRICS_FILE}" ]]; then
        echo "[SKIP] ${NAME} — metrics.json already exists at ${METRICS_FILE}"
        STATUSES["${NAME}"]="skipped"
        RUNTIMES["${NAME}"]="-"
        SKIP=$((SKIP + 1))
        continue
    fi

    echo "──────────────────────────────────────────────────────────────"
    echo "[RUN]  ${NAME}"
    echo "  script : ${BASELINES_DIR}/${SCRIPT}"
    echo "  output : ${OUT_DIR}"
    echo "  device : ${DEVICE}"
    echo "  split  : ${SPLIT}"
    echo "  timeout: ${TIMEOUT_SEC}s"
    echo "──────────────────────────────────────────────────────────────"

    START_TS=$(date +%s)

    # shellcheck disable=SC2086
    if timeout "${TIMEOUT_SEC}" \
        "${PYTHON}" "${BASELINES_DIR}/${SCRIPT}" \
            --data_dir "${DATA_DIR}" \
            --output_dir "${OUT_DIR}" \
            --split "${SPLIT}" \
            --device "${DEVICE}" \
            --image_size "${IMAGE_SIZE:-256}" \
            ${EXTRA_FLAGS} \
        2>&1 | tee "${LOG_FILE}"; then
        EXIT_CODE=0
    else
        EXIT_CODE=$?
    fi

    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))
    RUNTIMES["${NAME}"]="${ELAPSED}s"

    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "pass" > "${STATUS_FILE}"
        STATUSES["${NAME}"]="pass"
        PASS=$((PASS + 1))
        echo "[DONE] ${NAME} — ${ELAPSED}s — PASS"
    elif [[ ${EXIT_CODE} -eq 124 ]]; then
        echo "timeout" > "${STATUS_FILE}"
        STATUSES["${NAME}"]="timeout"
        FAIL=$((FAIL + 1))
        echo "[DONE] ${NAME} — ${ELAPSED}s — TIMEOUT (${TIMEOUT_SEC}s)"
    else
        echo "fail (exit ${EXIT_CODE})" > "${STATUS_FILE}"
        STATUSES["${NAME}"]="fail (exit ${EXIT_CODE})"
        FAIL=$((FAIL + 1))
        echo "[DONE] ${NAME} — ${ELAPSED}s — FAIL (exit ${EXIT_CODE})"
    fi
    echo ""
done

# ── Summary table ──────────────────────────────────────────────────────────
SUMMARY_FILE="${RESULTS_DIR}/summary.txt"

{
    echo "============================================================"
    echo "Foundation Baselines Summary"
    echo "Date    : $(date -Iseconds)"
    echo "Device  : ${DEVICE}"
    echo "Split   : ${SPLIT}"
    echo "Data    : ${DATA_DIR}"
    echo "Results : ${RESULTS_DIR}"
    echo "============================================================"
    echo ""
    printf "%-22s  %-10s  %-8s  %s\n" "Baseline" "Status" "Runtime" "Metrics"
    printf "%-22s  %-10s  %-8s  %s\n" "----------------------" "----------" "--------" "-----------------------------"

    for entry in "${BASELINES[@]}"; do
        IFS='|' read -r NAME _ _ <<< "${entry}"
        STATUS="${STATUSES[${NAME}]:-unknown}"
        RUNTIME="${RUNTIMES[${NAME}]:--}"
        METRICS_FILE="${RESULTS_DIR}/${NAME}/metrics.json"

        METRICS_BRIEF="-"
        if [[ -f "${METRICS_FILE}" ]]; then
            # Try to extract aggDSC and HD95 if present
            if command -v "${PYTHON}" &>/dev/null; then
                METRICS_BRIEF=$("${PYTHON}" -c "
import json, sys
try:
    m = json.load(open('${METRICS_FILE}'))
    parts = []
    for k in ['aggDSC', 'agg_dsc', 'dice', 'DSC']:
        if k in m:
            parts.append(f'{k}={m[k]:.4f}')
            break
    for k in ['HD95', 'hd95']:
        if k in m:
            parts.append(f'{k}={m[k]:.2f}')
            break
    if 'fallback' in m and m['fallback']:
        parts.append('(fallback)')
    print(', '.join(parts) if parts else json.dumps(m)[:60])
except Exception:
    print('-')
" 2>/dev/null || echo "-")
            fi
        fi

        printf "%-22s  %-10s  %-8s  %s\n" "${NAME}" "${STATUS}" "${RUNTIME}" "${METRICS_BRIEF}"
    done

    echo ""
    echo "Totals: ${PASS} pass, ${FAIL} fail/timeout, ${SKIP} skipped"
} | tee "${SUMMARY_FILE}"

echo ""
echo "Summary written to: ${SUMMARY_FILE}"
