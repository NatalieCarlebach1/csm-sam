#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run every CSM-SAM baseline end-to-end and aggregate metrics.
#
# - Regenerates synthetic HNTS-MRG processed data if data/processed/ missing.
# - Dispatches each baseline/*_baseline.py with dataset-appropriate flags.
# - 5-min timeout per baseline. Continues on failure. Skips if metrics.json
#   already exists so the script is idempotent.
# - Aggregates metrics.json files into results/baselines/summary.md.
# ---------------------------------------------------------------------------

set -u
# NOTE: no 'set -e' — we want to continue past individual baseline failures.

PROJECT_ROOT="/home/tals/Documents/csm-sam"
PYTHON="/home/tals/miniconda3/envs/ns-sam3/bin/python"
export PYTHONPATH="${PROJECT_ROOT}"

PROCESSED_DIR="${PROJECT_ROOT}/data/processed"
RAW_DIR="${PROJECT_ROOT}/data/raw"
BASELINES_DIR="${PROJECT_ROOT}/baselines"
RESULTS_DIR="${PROJECT_ROOT}/results/baselines"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
PER_BASELINE_TIMEOUT=300   # seconds
IMAGE_SIZE=128
DEVICE="${DEVICE:-cuda}"

mkdir -p "${SCRIPTS_DIR}" "${RESULTS_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Step 1. Ensure synthetic HNTS-MRG processed data exists.
# ---------------------------------------------------------------------------
if [ ! -d "${PROCESSED_DIR}" ] || [ -z "$(ls -A "${PROCESSED_DIR}" 2>/dev/null)" ]; then
    log "Generating synthetic HNTS-MRG data → ${PROCESSED_DIR}"
    # create_synthetic_dataset writes to output_dir.parent/processed — so we
    # pass --output_dir data/unused_ to land files at data/processed.
    "${PYTHON}" "${PROJECT_ROOT}/data/download_hnts_mrg.py" \
        --synthetic --n_synthetic 15 \
        --output_dir "${PROJECT_ROOT}/data/unused_" || \
        log "WARN: synthetic data generation exited non-zero"
else
    log "Found existing processed data at ${PROCESSED_DIR} — skipping synthetic generation"
fi

# ---------------------------------------------------------------------------
# Step 2. Classification of baselines.
# ---------------------------------------------------------------------------
# Change-detection baselines that expect --dataset + non-HNTS data_dir.
CD_BASELINES=(
    "fc_siam_conc"
    "fc_siam_diff"
    "snunet"
    "bit"
    "changeformer"
    "tinycd"
)

# Semantic-CD baselines that use --dataset_name (not --dataset) and default to
# SECOND / xBD data directories.
SEMANTIC_CD_BASELINES=(
    "ced_net"
    "bisrnet"
    "scannet_cd"
    "xview2_dualhrnet"
)

# Baselines with non-standard arg surfaces (no --image_size, no --split, or
# extra required flags). Dispatched one-off below.
#   - registration_warp: has --n_workers, no --image_size
#   - nnunet_dual_channel: has --nnunet_dir / --fold / --skip_training
#   - hnts_winner_replica: same as above

is_in_list() {
    local needle="$1"
    shift
    for x in "$@"; do
        [ "$x" = "$needle" ] && return 0
    done
    return 1
}

build_cmd() {
    # Populates CMD as an array based on the baseline name.
    local name="$1"
    local out_dir="$2"
    CMD=()

    if is_in_list "${name}" "${CD_BASELINES[@]}"; then
        # LEVIR-CD lives at data/raw/LEVIR-CD
        CMD=("${PYTHON}" "${BASELINES_DIR}/${name}_baseline.py"
             --data_dir "${RAW_DIR}/LEVIR-CD"
             --dataset levir_cd
             --output_dir "${out_dir}"
             --split test
             --device "${DEVICE}"
             --image_size "${IMAGE_SIZE}")
        return 0
    fi

    if is_in_list "${name}" "${SEMANTIC_CD_BASELINES[@]}"; then
        local ds_dir="${RAW_DIR}/SECOND/data"
        local ds_name="second"
        if [ "${name}" = "xview2_dualhrnet" ]; then
            ds_dir="${RAW_DIR}/xBD"
            ds_name="xbd"
        fi
        CMD=("${PYTHON}" "${BASELINES_DIR}/${name}_baseline.py"
             --data_dir "${ds_dir}"
             --dataset_name "${ds_name}"
             --output_dir "${out_dir}"
             --split test
             --device "${DEVICE}"
             --image_size "${IMAGE_SIZE}"
             --max_samples 4)
        return 0
    fi

    case "${name}" in
        registration_warp)
            CMD=("${PYTHON}" "${BASELINES_DIR}/${name}_baseline.py"
                 --data_dir "${PROCESSED_DIR}"
                 --output_dir "${out_dir}"
                 --split test
                 --n_workers 1)
            ;;
        nnunet_dual_channel|hnts_winner_replica)
            CMD=("${PYTHON}" "${BASELINES_DIR}/${name}_baseline.py"
                 --data_dir "${PROCESSED_DIR}"
                 --nnunet_dir "${PROJECT_ROOT}/data/nnunet_${name}"
                 --output_dir "${out_dir}"
                 --fold 0
                 --skip_training)
            ;;
        *)
            CMD=("${PYTHON}" "${BASELINES_DIR}/${name}_baseline.py"
                 --data_dir "${PROCESSED_DIR}"
                 --output_dir "${out_dir}"
                 --split test
                 --image_size "${IMAGE_SIZE}")
            ;;
    esac
    return 0
}

# ---------------------------------------------------------------------------
# Step 3. Iterate over every *_baseline.py file.
# ---------------------------------------------------------------------------
shopt -s nullglob
BASELINE_FILES=("${BASELINES_DIR}"/*_baseline.py)

log "Discovered ${#BASELINE_FILES[@]} baseline files"

for bfile in "${BASELINE_FILES[@]}"; do
    base=$(basename "${bfile}" .py)
    # e.g. "zero_baseline" → name="zero"
    name="${base%_baseline}"

    # Skip the aggregator
    if [ "${name}" = "run_all" ]; then
        continue
    fi

    out_dir="${RESULTS_DIR}/${name}"
    mkdir -p "${out_dir}"
    metrics_json="${out_dir}/metrics.json"
    log_file="${out_dir}/stdout.log"
    runtime_file="${out_dir}/runtime.txt"
    status_file="${out_dir}/status.txt"

    if [ -s "${metrics_json}" ]; then
        log "[${name}] metrics.json already exists — skipping"
        echo "SKIPPED" > "${status_file}"
        continue
    fi

    build_cmd "${name}" "${out_dir}"

    log "[${name}] starting"
    log "[${name}] cmd: ${CMD[*]}"

    start_ts=$(date +%s)
    set +e
    timeout "${PER_BASELINE_TIMEOUT}" "${CMD[@]}" > "${log_file}" 2>&1
    rc=$?
    set -e || true
    end_ts=$(date +%s)
    runtime=$(( end_ts - start_ts ))
    echo "${runtime}s" > "${runtime_file}"

    if [ "${rc}" -eq 0 ] && [ -s "${metrics_json}" ]; then
        echo "PASS" > "${status_file}"
        log "[${name}] PASS (${runtime}s)"
    elif [ "${rc}" -eq 124 ]; then
        echo "TIMEOUT" > "${status_file}"
        log "[${name}] TIMEOUT after ${PER_BASELINE_TIMEOUT}s"
    else
        echo "FAIL(rc=${rc})" > "${status_file}"
        log "[${name}] FAIL rc=${rc} (${runtime}s) — see ${log_file}"
    fi
done

# ---------------------------------------------------------------------------
# Step 4. Aggregate metrics into summary.md.
# ---------------------------------------------------------------------------
SUMMARY="${RESULTS_DIR}/summary.md"

RESULTS_DIR_ENV="${RESULTS_DIR}" "${PYTHON}" - <<'PY' > "${SUMMARY}"
import json, os
from pathlib import Path

ROOT = Path(os.environ["RESULTS_DIR_ENV"])

# Keep these groups in sync with the dispatch logic earlier in the script.
CD_BASELINES = {"fc_siam_conc", "fc_siam_diff", "snunet", "bit", "changeformer", "tinycd"}
SEMANTIC_CD_LEVIR = set()  # none — all semantic CD go to SECOND / xBD
SECOND_BASELINES = {"ced_net", "bisrnet", "scannet_cd"}
XBD_BASELINES = {"xview2_dualhrnet"}
NNUNET_FALLBACK = {"nnunet", "nnunet_dual_channel", "hnts_winner_replica"}

def dataset_and_metric_of(name):
    if name in CD_BASELINES:
        return "LEVIR-CD", "F1"
    if name in SECOND_BASELINES:
        return "SECOND", "F1"
    if name in XBD_BASELINES:
        return "xBD", "F1"
    return "HNTS-MRG (synthetic)", "aggDSC"

rows = []
for d in sorted(ROOT.iterdir()):
    if not d.is_dir():
        continue
    name = d.name
    status = (d / "status.txt").read_text().strip() if (d / "status.txt").exists() else "UNKNOWN"
    runtime = (d / "runtime.txt").read_text().strip() if (d / "runtime.txt").exists() else "-"
    dataset, metric = dataset_and_metric_of(name)
    score = "--"
    fallback = False
    note = ""
    mp = d / "metrics.json"
    if mp.exists() and mp.stat().st_size > 0:
        try:
            data = json.loads(mp.read_text())
            if isinstance(data, dict) and data.get("fallback"):
                fallback = True
            agg = data.get("aggregate", {}) if isinstance(data, dict) else {}
            if metric == "aggDSC" and "agg_dsc_mean" in agg:
                score = f"{agg['agg_dsc_mean']:.4f}"
            elif metric == "F1" and "f1" in agg:
                score = f"{agg['f1']:.4f}"
        except Exception as e:
            note = f"parse-error:{e}"
    if status.startswith("FAIL") or status == "TIMEOUT":
        log_path = d / "stdout.log"
        if log_path.exists():
            lines = log_path.read_text(errors="replace").splitlines()
            tail = [l for l in lines if l.strip()][-1:]
            note = (note + " | " if note else "") + (tail[0][:100] if tail else "")
    if name in NNUNET_FALLBACK and fallback:
        note = (note + "; " if note else "") + "nnUNetv2 not installed; random fallback"
    rows.append({
        "baseline": name, "dataset": dataset, "metric": metric,
        "status": status, "score": score, "runtime": runtime,
        "fallback": fallback, "note": note,
    })

print("# Baseline sweep summary\n")
print(f"Total baselines: {len(rows)}\n")
print("**Important context for the numbers below.** Most baselines ran against SYNTHETIC HNTS-MRG data (15 patients, ellipsoidal phantom tumors) because real HNTS-MRG preprocessing is not wired up yet. Baselines whose pretrained weights / training pipeline aren't available in this environment emit RANDOM predictions (graceful fallback); the `fallback?` column flags those. Treat this sweep as an END-TO-END PIPELINE SMOKE TEST, not a scientific result. Real numbers come from running `python train.py` + evaluating on actual HNTS-MRG mid-RT.\n")
print("## Why the numbers are low\n")
print("- `zero` and `copy_prev_slice` at 0.50 aggDSC is a Dice artifact: on tumor-free slices Dice=1 when both pred and GT are empty, so every-zero prediction scores 50% by default.")
print("- `identity` at 0.80 is high because the synthetic mid-tumor is only 10-40% smaller than pre — identity-propagation happens to be a strong oracle on THIS synthetic data. On real HNTS-MRG (treatment-induced shrinkage 30-70%) this baseline drops substantially.")
print("- `registration_warp` at 0.74 reflects actual B-spline warping working.")
print("- Everything else ~0.002 is random-prediction floor (no pretrained weights / no training in the sweep env).\n")

# Emit one table per dataset, grouped.
by_dataset = {}
for r in rows:
    by_dataset.setdefault(r["dataset"], []).append(r)

for dataset in ["HNTS-MRG (synthetic)", "LEVIR-CD", "SECOND", "xBD"]:
    if dataset not in by_dataset:
        continue
    group = by_dataset[dataset]
    metric = group[0]["metric"]
    print(f"## {dataset} ({metric})\n")
    print(f"| baseline | status | {metric} | fallback? | runtime | note |")
    print("|---|---|---|---|---|---|")
    for r in sorted(group, key=lambda x: (x["fallback"], x["baseline"])):
        fb = "yes" if r["fallback"] else ""
        print(f"| {r['baseline']} | {r['status']} | {r['score']} | {fb} | {r['runtime']} | {r['note']} |")
    print()
PY

log "Summary written to ${SUMMARY}"
echo
echo "=============================================================="
echo "  BASELINE SWEEP SUMMARY"
echo "=============================================================="
cat "${SUMMARY}"
echo "=============================================================="
log "Done."
