#!/bin/bash
# CSM-SAM — full pipeline: setup + train
#
# ── Quickstart ────────────────────────────────────────────────────────────────
#   bash run_all.sh --gpus 4 --cv
#
# ── AWS example ───────────────────────────────────────────────────────────────
#   bash run_all.sh --gpus 8 --cv \
#     --s3_data s3://my-bucket/hnts-mrg/raw \
#     --s3_checkpoint s3://my-bucket/checkpoints/sam2.1_hiera_large.pt \
#     --s3_output s3://my-bucket/csmsam/results
#
# ── AWS multi-node (run on EACH node) ─────────────────────────────────────────
#   NODE 0: bash run_all.sh --gpus 8 --nodes 2 --node_rank 0 --master_addr <IP> --cv ...
#   NODE 1: bash run_all.sh --gpus 8 --nodes 2 --node_rank 1 --master_addr <IP> --cv ...
#
# ── All options ───────────────────────────────────────────────────────────────
#   --gpus            GPUs per node                   (default: 1)
#   --cv              5-fold cross-validation
#   --n_folds         number of folds                 (default: 5)
#   --config          config yaml                     (default: configs/default.yaml)
#   --raw_dir         raw data directory              (default: data/raw)
#   --data_dir        preprocessed data directory     (default: data/processed)
#   --ckpt_dir        SAM2 checkpoint directory       (default: checkpoints/sam2)
#   --output_dir      training output directory       (default: checkpoints/csmsam_hnts)
#   --s3_data         s3://bucket/path  — raw data from S3
#   --s3_checkpoint   s3://bucket/file  — SAM2 checkpoint from S3
#   --s3_output       s3://bucket/path  — push results to S3
#   --synthetic       use synthetic data (no real dataset needed)
#   --n_synthetic     number of synthetic patients    (default: 15)
#   --nodes           total nodes for multi-node      (default: 1)
#   --node_rank       rank of this node               (default: 0)
#   --master_addr     IP of rank-0 node               (default: localhost)
#   --master_port     rendezvous port                 (default: 29500)
#   --skip_setup      skip setup (if already done)
#   --conda_env       conda environment name to use (e.g. owl-env)
#   --sam2_checkpoint full path to SAM2 checkpoint (overrides --ckpt_dir)

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
GPUS=1
CV=false
N_FOLDS=5
CONFIG="configs/default.yaml"
RAW_DIR="data/raw"
DATA_DIR="data/processed"
CKPT_DIR="checkpoints/sam2"
OUTPUT_DIR="checkpoints/csmsam_hnts"
S3_DATA=""
S3_CKPT=""
S3_OUTPUT=""
SYNTHETIC=false
N_SYNTHETIC=15
NODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29500
SKIP_SETUP=false
CONDA_ENV=""
SAM2_CKPT_OVERRIDE=""

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)           GPUS="$2";         shift 2 ;;
    --cv)             CV=true;           shift   ;;
    --n_folds)        N_FOLDS="$2";      shift 2 ;;
    --config)         CONFIG="$2";       shift 2 ;;
    --raw_dir)        RAW_DIR="$2";      shift 2 ;;
    --data_dir)       DATA_DIR="$2";     shift 2 ;;
    --ckpt_dir)       CKPT_DIR="$2";     shift 2 ;;
    --output_dir)     OUTPUT_DIR="$2";   shift 2 ;;
    --s3_data)        S3_DATA="$2";      shift 2 ;;
    --s3_checkpoint)  S3_CKPT="$2";      shift 2 ;;
    --s3_output)      S3_OUTPUT="$2";    shift 2 ;;
    --synthetic)      SYNTHETIC=true;    shift   ;;
    --n_synthetic)    N_SYNTHETIC="$2";  shift 2 ;;
    --nodes)          NODES="$2";        shift 2 ;;
    --node_rank)      NODE_RANK="$2";    shift 2 ;;
    --master_addr)    MASTER_ADDR="$2";  shift 2 ;;
    --master_port)    MASTER_PORT="$2";  shift 2 ;;
    --skip_setup)        SKIP_SETUP=true;          shift   ;;
    --conda_env)         CONDA_ENV="$2";           shift 2 ;;
    --sam2_checkpoint)   SAM2_CKPT_OVERRIDE="$2";  shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Resolve Python / torchrun ─────────────────────────────────────────────────
if [ -n "$CONDA_ENV" ]; then
  CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
  PY="$CONDA_BASE/envs/$CONDA_ENV/bin/python3"
  TORCHRUN="$CONDA_BASE/envs/$CONDA_ENV/bin/torchrun"
  PIP="$CONDA_BASE/envs/$CONDA_ENV/bin/pip"
  echo "Using conda env: $CONDA_ENV ($PY)"
else
  PY=$(which python3)
  TORCHRUN=$(which torchrun)
  PIP=$(which pip)
fi

if [ -n "$SAM2_CKPT_OVERRIDE" ]; then
  SAM2_CKPT="$SAM2_CKPT_OVERRIDE"
else
  SAM2_CKPT="$CKPT_DIR/sam2.1_hiera_large.pt"
fi

TOTAL_GPUS=$(( GPUS * NODES ))

echo "========================================================"
echo " CSM-SAM Full Pipeline"
echo " GPUs/node: $GPUS  |  Nodes: $NODES  |  Total: $TOTAL_GPUS GPU(s)"
[ "$CV" = true ] && echo " Mode: ${N_FOLDS}-fold CV" || echo " Mode: single run"
echo "========================================================"

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Install
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_SETUP" = false ]; then
  echo ""
  echo "[1/4] Installing csmsam package..."
  $PIP install -e ".[dev]" --quiet
  echo "      Done."
else
  echo "[1/4] Skipping setup."
fi

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — SAM2 checkpoint
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_SETUP" = false ]; then
  echo ""
  echo "[2/4] SAM2 checkpoint..."
  mkdir -p "$CKPT_DIR"
  if [ -f "$SAM2_CKPT" ]; then
    echo "      Already exists: $SAM2_CKPT"
  elif [ -n "$S3_CKPT" ]; then
    echo "      Downloading from $S3_CKPT"
    aws s3 cp "$S3_CKPT" "$SAM2_CKPT"
  else
    echo "      Downloading from HuggingFace..."
    $PIP install -q huggingface_hub
    $PY - <<PYEOF
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(repo_id="facebook/sam2.1-hiera-large", filename="sam2.1_hiera_large.pt")
os.makedirs("$CKPT_DIR", exist_ok=True)
shutil.copy(path, "$SAM2_CKPT")
print("      Saved to $SAM2_CKPT")
PYEOF
  fi
fi

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Dataset
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_SETUP" = false ]; then
  echo ""
  echo "[3/4] Dataset..."

  NEED_PREPROCESS=true
  if [ -d "$DATA_DIR" ] && [ "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo "      Preprocessed data already exists: $DATA_DIR"
    NEED_PREPROCESS=false
  fi

  if [ "$SYNTHETIC" = true ] && [ "$NEED_PREPROCESS" = true ]; then
    echo "      Generating $N_SYNTHETIC synthetic patients..."
    $PY data/download_hnts_mrg.py --synthetic --n_synthetic "$N_SYNTHETIC" --output_dir "$DATA_DIR"
    NEED_PREPROCESS=false
  elif [ -n "$S3_DATA" ] && [ "$NEED_PREPROCESS" = true ]; then
    echo "      Syncing raw data from $S3_DATA → $RAW_DIR"
    mkdir -p "$RAW_DIR"
    aws s3 sync "$S3_DATA" "$RAW_DIR" --no-progress
  fi

  if [ "$NEED_PREPROCESS" = true ]; then
    if [ ! -d "$RAW_DIR" ] || [ -z "$(ls -A $RAW_DIR 2>/dev/null)" ]; then
      echo ""
      echo "  ERROR: No data found at $RAW_DIR and --s3_data not provided."
      echo "  Options:"
      echo "    --synthetic                          use synthetic data for testing"
      echo "    --s3_data s3://bucket/hnts-mrg/raw   pull from S3"
      echo "    Download manually: https://zenodo.org/record/11829006"
      exit 1
    fi
    echo "      Preprocessing..."
    $PY data/preprocess.py --input_dir "$RAW_DIR" --output_dir "$DATA_DIR" --n_workers 4
  fi
fi

# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Train
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "[4/4] Training..."

if [ "$NODES" -gt 1 ]; then
  export NCCL_SOCKET_IFNAME=ens5
  export NCCL_IB_DISABLE=0
  export FI_PROVIDER=efa
  export NCCL_DEBUG=WARN
fi

run_fold() {
  local extra="$1"
  local base="--config $CONFIG --data_dir $DATA_DIR --sam2_checkpoint $SAM2_CKPT $extra"
  if [ "$TOTAL_GPUS" -gt 1 ]; then
    $TORCHRUN \
      --nproc_per_node="$GPUS" \
      --nnodes="$NODES" \
      --node_rank="$NODE_RANK" \
      --master_addr="$MASTER_ADDR" \
      --master_port="$MASTER_PORT" \
      train.py $base
  else
    $PY train.py $base
  fi
}

if [ "$CV" = true ]; then
  for fold in $(seq 0 $((N_FOLDS - 1))); do
    echo ""
    echo "  ── Fold ${fold} / ${N_FOLDS} ──"
    run_fold "--output_dir ${OUTPUT_DIR}/fold${fold} --fold ${fold} --n_folds ${N_FOLDS}"
  done
else
  run_fold "--output_dir ${OUTPUT_DIR}"
fi

# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Push results to S3 (optional)
# ════════════════════════════════════════════════════════════════════════════
if [ -n "$S3_OUTPUT" ]; then
  echo ""
  echo "Pushing results to $S3_OUTPUT..."
  aws s3 sync "$OUTPUT_DIR" "$S3_OUTPUT" --no-progress
fi

echo ""
echo "========================================================"
echo " All done. Results in $OUTPUT_DIR"
echo "========================================================"
