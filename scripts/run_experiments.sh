#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Master orchestration script for CSM-SAM evaluation pipeline.
#
# Chains three phases sequentially:
#   Phase 1: preprocess_hnts.sh    — extract + preprocess HNTS-MRG raw → processed
#   Phase 2: run_naive_baselines.sh — 6 naive/sanity baselines on real data
#   Phase 3: run_foundation_baselines.sh — 7 foundation zero-shot baselines
#
# Each phase logs to results/experiments/logs/<phase>.log with timestamps.
# If a phase fails, the script continues to the next phase.
# After all phases, metrics.json files are aggregated into summary.md.
#
# Environment variables:
#   DEVICE          — cuda or cpu (default: cuda)
#   RESULTS_BASE    — override experiment output root
#                     (default: $PROJECT_ROOT/results/experiments)
#
# Contract with subscripts:
#   This script exports RESULTS_BASE. Each subscript (run_naive_baselines.sh,
#   run_foundation_baselines.sh) is expected to honor RESULTS_BASE from the
#   environment if set, using it as the parent directory for their per-baseline
#   output folders. Subscripts written by parallel agents should check:
#     OUTPUT_DIR="${RESULTS_BASE:-<their default>}"
#   or accept --output_dir as the first positional flag pair.
#
# Usage:
#   bash scripts/run_experiments.sh
#   DEVICE=cpu bash scripts/run_experiments.sh
# ---------------------------------------------------------------------------

set -u
# No set -e: we continue past phase failures.

PROJECT_ROOT="/home/tals/Documents/csm-sam"
PYTHON="/home/tals/miniconda3/envs/ns-sam3/bin/python"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"

export DEVICE="${DEVICE:-cuda}"
export PYTHONPATH="${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Experiment directory structure
# ---------------------------------------------------------------------------
export RESULTS_BASE="${RESULTS_BASE:-${PROJECT_ROOT}/results/experiments}"

LOGS_DIR="${RESULTS_BASE}/logs"
NAIVE_DIR="${RESULTS_BASE}/hnts_real_baselines"
FOUNDATION_DIR="${RESULTS_BASE}/hnts_foundation_baselines"

mkdir -p "${LOGS_DIR}"

# Naive baseline subdirectories
for d in identity zero random copy_prev_slice majority_voxel registration_warp; do
    mkdir -p "${NAIVE_DIR}/${d}"
done

# Foundation baseline subdirectories
for d in sam_vanilla sam2_point_prompt sam2_video medsam2 dinov2_linear clipseg totalsegmentator; do
    mkdir -p "${FOUNDATION_DIR}/${d}"
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_phase() {
    local phase_name="$1"
    local log_file="$2"
    shift 2
    # Remaining args are the command + arguments

    log "======== PHASE: ${phase_name} ========"
    log "Command: $*"
    log "Log file: ${log_file}"

    local start_ts
    start_ts=$(date +%s)

    {
        echo "# ${phase_name}"
        echo "# Started: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "# Command: $*"
        echo "# Device: ${DEVICE}"
        echo "# ----------------------------------------"
        "$@" 2>&1
        local rc=$?
        echo "# ----------------------------------------"
        echo "# Finished: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "# Exit code: ${rc}"
        exit ${rc}
    } > "${log_file}" 2>&1
    local rc=$?

    local end_ts
    end_ts=$(date +%s)
    local elapsed=$(( end_ts - start_ts ))

    if [ "${rc}" -eq 0 ]; then
        log "PHASE ${phase_name}: PASS (${elapsed}s)"
    else
        log "PHASE ${phase_name}: FAIL (rc=${rc}, ${elapsed}s) — see ${log_file}"
    fi

    return ${rc}
}

# ---------------------------------------------------------------------------
# Phase 1: Preprocess HNTS-MRG
# ---------------------------------------------------------------------------
log "Starting CSM-SAM experiment pipeline"
log "DEVICE=${DEVICE}  RESULTS_BASE=${RESULTS_BASE}"
echo ""

PHASE1_OK=0
if [ -x "${SCRIPTS_DIR}/preprocess_hnts.sh" ]; then
    run_phase "Preprocess HNTS-MRG" \
        "${LOGS_DIR}/preprocess.log" \
        bash "${SCRIPTS_DIR}/preprocess_hnts.sh" || PHASE1_OK=1
else
    log "WARN: ${SCRIPTS_DIR}/preprocess_hnts.sh not found or not executable — skipping Phase 1"
    echo "# Phase 1 skipped — script not found" > "${LOGS_DIR}/preprocess.log"
    PHASE1_OK=1
fi
echo ""

# ---------------------------------------------------------------------------
# Phase 2: Naive baselines
# ---------------------------------------------------------------------------
PHASE2_OK=0
if [ -x "${SCRIPTS_DIR}/run_naive_baselines.sh" ]; then
    run_phase "Naive Baselines" \
        "${LOGS_DIR}/naive_baselines.log" \
        bash "${SCRIPTS_DIR}/run_naive_baselines.sh" --output_dir "${NAIVE_DIR}" || PHASE2_OK=1
else
    log "WARN: ${SCRIPTS_DIR}/run_naive_baselines.sh not found or not executable — skipping Phase 2"
    echo "# Phase 2 skipped — script not found" > "${LOGS_DIR}/naive_baselines.log"
    PHASE2_OK=1
fi
echo ""

# ---------------------------------------------------------------------------
# Phase 3: Foundation baselines
# ---------------------------------------------------------------------------
PHASE3_OK=0
if [ -x "${SCRIPTS_DIR}/run_foundation_baselines.sh" ]; then
    run_phase "Foundation Baselines" \
        "${LOGS_DIR}/foundation_baselines.log" \
        bash "${SCRIPTS_DIR}/run_foundation_baselines.sh" --output_dir "${FOUNDATION_DIR}" || PHASE3_OK=1
else
    log "WARN: ${SCRIPTS_DIR}/run_foundation_baselines.sh not found or not executable — skipping Phase 3"
    echo "# Phase 3 skipped — script not found" > "${LOGS_DIR}/foundation_baselines.log"
    PHASE3_OK=1
fi
echo ""

# ---------------------------------------------------------------------------
# Aggregate: collect all metrics.json into summary.md
# ---------------------------------------------------------------------------
log "Aggregating results into summary.md"

SUMMARY="${RESULTS_BASE}/summary.md"

RESULTS_BASE_ENV="${RESULTS_BASE}" "${PYTHON}" - <<'PYAGG' > "${SUMMARY}"
import json, os
from pathlib import Path

RESULTS_BASE = Path(os.environ["RESULTS_BASE_ENV"])

# Scan both baseline directories for metrics.json files
NAIVE_DIR = RESULTS_BASE / "hnts_real_baselines"
FOUNDATION_DIR = RESULTS_BASE / "hnts_foundation_baselines"

rows = []

def collect_from(base_dir, category):
    """Walk baseline subdirectories, read metrics.json + status.txt + runtime."""
    if not base_dir.exists():
        return
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name

        # Status
        status_file = d / "status.txt"
        status = status_file.read_text().strip() if status_file.exists() else "UNKNOWN"

        # Runtime
        runtime_file = d / "runtime.txt"
        runtime = runtime_file.read_text().strip() if runtime_file.exists() else "-"

        # Metrics
        agg_dsc = "--"
        gtvp_dsc = "--"
        gtvn_dsc = "--"
        hd95 = "--"

        mp = d / "metrics.json"
        if mp.exists() and mp.stat().st_size > 0:
            try:
                data = json.loads(mp.read_text())
                agg = data.get("aggregate", {}) if isinstance(data, dict) else {}

                if "agg_dsc_mean" in agg:
                    agg_dsc = f"{agg['agg_dsc_mean']:.4f}"
                elif "aggDSC" in agg:
                    agg_dsc = f"{agg['aggDSC']:.4f}"

                if "gtvp_dsc_mean" in agg:
                    gtvp_dsc = f"{agg['gtvp_dsc_mean']:.4f}"
                elif "GTVp_DSC" in agg:
                    gtvp_dsc = f"{agg['GTVp_DSC']:.4f}"

                if "gtvn_dsc_mean" in agg:
                    gtvn_dsc = f"{agg['gtvn_dsc_mean']:.4f}"
                elif "GTVn_DSC" in agg:
                    gtvn_dsc = f"{agg['GTVn_DSC']:.4f}"

                if "hd95_mean" in agg:
                    hd95 = f"{agg['hd95_mean']:.2f}"
                elif "HD95" in agg:
                    hd95 = f"{agg['HD95']:.2f}"

                if status == "UNKNOWN" and agg:
                    status = "PASS"
            except Exception:
                status = status if status != "UNKNOWN" else "PARSE_ERROR"

        rows.append({
            "baseline": name,
            "category": category,
            "aggDSC": agg_dsc,
            "gtvp_dsc": gtvp_dsc,
            "gtvn_dsc": gtvn_dsc,
            "hd95": hd95,
            "runtime": runtime,
            "status": status,
        })

collect_from(NAIVE_DIR, "Naive / Sanity")
collect_from(FOUNDATION_DIR, "Foundation Zero-Shot")

print("# CSM-SAM Experiment Results\n")
print(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
print(f"Total baselines evaluated: {len(rows)}\n")

if not rows:
    print("No baseline results found. Check logs under results/experiments/logs/.\n")
else:
    # Group by category
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    for cat in ["Naive / Sanity", "Foundation Zero-Shot"]:
        if cat not in by_cat:
            continue
        group = by_cat[cat]
        print(f"## {cat} Baselines\n")
        print("| baseline | aggDSC | GTVp DSC | GTVn DSC | HD95 | runtime | status |")
        print("|---|---|---|---|---|---|---|")
        for r in sorted(group, key=lambda x: x["baseline"]):
            print(f"| {r['baseline']} | {r['aggDSC']} | {r['gtvp_dsc']} | {r['gtvn_dsc']} | {r['hd95']} | {r['runtime']} | {r['status']} |")
        print()
PYAGG

# ---------------------------------------------------------------------------
# Print summary to stdout
# ---------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "  CSM-SAM EXPERIMENT SUMMARY"
echo "=============================================================="
cat "${SUMMARY}"
echo "=============================================================="
echo ""

# ---------------------------------------------------------------------------
# Final status
# ---------------------------------------------------------------------------
TOTAL_FAIL=$(( PHASE1_OK + PHASE2_OK + PHASE3_OK ))
if [ "${TOTAL_FAIL}" -eq 0 ]; then
    log "All 3 phases completed successfully."
else
    log "${TOTAL_FAIL} of 3 phases had errors (see logs for details)."
fi

log "Summary: ${SUMMARY}"
echo "Logs: ${RESULTS_BASE}/logs/*.log"
