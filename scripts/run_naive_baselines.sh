#!/usr/bin/env bash
# run_naive_baselines.sh
# Runs the 5 naive baselines + registration_warp on real preprocessed HNTS-MRG data.
# Idempotent: skips baselines whose metrics.json already exists.
#
# Usage:
#   bash scripts/run_naive_baselines.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PYTHON="${PYTHON:-/home/tals/miniconda3/envs/ns-sam3/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
BASELINES_DIR="${PROJECT_ROOT}/baselines"
DATA_DIR="${PROJECT_ROOT}/data/processed"
RESULTS_DIR="${PROJECT_ROOT}/results/hnts_real_baselines"
SPLIT="test"
IMAGE_SIZE=1024

# ---------------------------------------------------------------------------
# Baseline definitions
# Format: name:script:accepts_image_size(1/0)
# ---------------------------------------------------------------------------
BASELINES=(
    "identity:identity_baseline.py:1"
    "zero:zero_baseline.py:1"
    "random:random_baseline.py:1"
    "copy_prev_slice:copy_prev_slice_baseline.py:1"
    "majority_voxel:majority_voxel_baseline.py:1"
    "registration_warp:registration_warp_baseline.py:0"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $*"
}

# ---------------------------------------------------------------------------
# Create results directory
# ---------------------------------------------------------------------------
mkdir -p "${RESULTS_DIR}"

log "=== Naive baselines run starting ==="
log "Python:      ${PYTHON}"
log "Data dir:    ${DATA_DIR}"
log "Results dir: ${RESULTS_DIR}"
log "Split:       ${SPLIT}"
log ""

# ---------------------------------------------------------------------------
# Run each baseline
# ---------------------------------------------------------------------------
declare -A STATUS_MAP
declare -A AGGDSC_MAP
declare -A RUNTIME_MAP

for entry in "${BASELINES[@]}"; do
    IFS=":" read -r name script has_image_size <<< "${entry}"

    out_dir="${RESULTS_DIR}/${name}"
    mkdir -p "${out_dir}"

    # Idempotent: skip if metrics.json already exists
    if [[ -f "${out_dir}/metrics.json" ]]; then
        log "[SKIP] ${name} — metrics.json already exists"
        STATUS_MAP["${name}"]="skip"
        # Try to read aggDSC from existing metrics
        if command -v jq &>/dev/null; then
            AGGDSC_MAP["${name}"]=$(jq -r '.aggDSC // .agg_dsc // "N/A"' "${out_dir}/metrics.json" 2>/dev/null || echo "N/A")
        else
            AGGDSC_MAP["${name}"]=$(${PYTHON} -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('aggDSC', d.get('agg_dsc', 'N/A')))" "${out_dir}/metrics.json" 2>/dev/null || echo "N/A")
        fi
        # Try to read runtime from existing runtime.txt
        if [[ -f "${out_dir}/runtime.txt" ]]; then
            RUNTIME_MAP["${name}"]=$(cat "${out_dir}/runtime.txt")
        else
            RUNTIME_MAP["${name}"]="N/A"
        fi
        continue
    fi

    # Build command
    cmd="${PYTHON} ${BASELINES_DIR}/${script} --data_dir ${DATA_DIR} --output_dir ${out_dir} --split ${SPLIT}"
    if [[ "${has_image_size}" == "1" ]]; then
        cmd="${cmd} --image_size ${IMAGE_SIZE}"
    fi

    log "[RUN]  ${name} — ${cmd}"

    start_ts=$(date +%s)
    set +e
    ${cmd} 2>&1 | tee "${out_dir}/stdout.log"
    exit_code=${PIPESTATUS[0]}
    set -e
    end_ts=$(date +%s)

    runtime=$((end_ts - start_ts))

    # Record exit code
    echo "${exit_code}" > "${out_dir}/status.txt"

    # Record runtime
    echo "${runtime}s" > "${out_dir}/runtime.txt"

    if [[ ${exit_code} -eq 0 ]]; then
        log "[DONE] ${name} — exit 0, ${runtime}s"
        STATUS_MAP["${name}"]="ok"
    else
        log "[FAIL] ${name} — exit ${exit_code}, ${runtime}s"
        STATUS_MAP["${name}"]="fail(${exit_code})"
    fi

    RUNTIME_MAP["${name}"]="${runtime}s"

    # Extract aggDSC from metrics.json if it exists
    if [[ -f "${out_dir}/metrics.json" ]]; then
        if command -v jq &>/dev/null; then
            AGGDSC_MAP["${name}"]=$(jq -r '.aggDSC // .agg_dsc // "N/A"' "${out_dir}/metrics.json" 2>/dev/null || echo "N/A")
        else
            AGGDSC_MAP["${name}"]=$(${PYTHON} -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('aggDSC', d.get('agg_dsc', 'N/A')))" "${out_dir}/metrics.json" 2>/dev/null || echo "N/A")
        fi
    else
        AGGDSC_MAP["${name}"]="N/A"
    fi
done

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
log ""
log "==========================================="
log "          SUMMARY TABLE"
log "==========================================="
printf "%-22s | %-10s | %-10s | %-10s\n" "Baseline" "Status" "aggDSC" "Runtime"
printf "%-22s-+-%-10s-+-%-10s-+-%-10s\n" "----------------------" "----------" "----------" "----------"

for entry in "${BASELINES[@]}"; do
    IFS=":" read -r name _ _ <<< "${entry}"
    printf "%-22s | %-10s | %-10s | %-10s\n" \
        "${name}" \
        "${STATUS_MAP[${name}]:-N/A}" \
        "${AGGDSC_MAP[${name}]:-N/A}" \
        "${RUNTIME_MAP[${name}]:-N/A}"
done

log ""
log "=== Naive baselines run complete ==="
log "Results saved to: ${RESULTS_DIR}"
