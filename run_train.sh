#!/bin/bash
# CSM-SAM training launcher — local or AWS
#
# ── Local examples ────────────────────────────────────────────────────────────
#   bash run_train.sh --gpus 1
#   bash run_train.sh --gpus 4
#   bash run_train.sh --gpus 4 --cv
#
# ── AWS single-node examples ──────────────────────────────────────────────────
#   bash run_train.sh --gpus 8 --cv \
#     --s3_data s3://my-bucket/csmsam/processed \
#     --s3_output s3://my-bucket/csmsam/results \
#     --s3_checkpoint s3://my-bucket/checkpoints/sam2.1_hiera_large.pt
#
# ── AWS multi-node example (run on EACH node) ─────────────────────────────────
#   NODE 0 (master):
#     bash run_train.sh --gpus 8 --nodes 2 --node_rank 0 --master_addr <NODE0_IP>
#   NODE 1:
#     bash run_train.sh --gpus 8 --nodes 2 --node_rank 1 --master_addr <NODE0_IP>
#
# ── All options ───────────────────────────────────────────────────────────────
#   --gpus          GPUs per node                    (default: 1)
#   --cv            run 5-fold cross-validation
#   --n_folds       number of folds                  (default: 5)
#   --data_dir      local path to preprocessed data  (default: data/processed)
#   --checkpoint    local path to SAM2 checkpoint    (default: checkpoints/sam2/sam2.1_hiera_large.pt)
#   --output_dir    local output directory           (default: checkpoints/csmsam_hnts)
#   --config        config yaml                      (default: configs/default.yaml)
#   --s3_data       s3://bucket/path  — sync data FROM S3 before training
#   --s3_checkpoint s3://bucket/file  — download SAM2 checkpoint from S3
#   --s3_output     s3://bucket/path  — sync results TO S3 after training
#   --nodes         total number of nodes            (default: 1)
#   --node_rank     rank of this node                (default: 0)
#   --master_addr   IP of rank-0 node                (default: localhost)
#   --master_port   rendezvous port                  (default: 29500)

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
GPUS=1
CV=false
N_FOLDS=5
DATA_DIR="data/processed"
SAM2_CKPT="checkpoints/sam2/sam2.1_hiera_large.pt"
OUTPUT_DIR="checkpoints/csmsam_hnts"
CONFIG="configs/default.yaml"
S3_DATA=""
S3_CKPT=""
S3_OUTPUT=""
NODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29500

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)          GPUS="$2";         shift 2 ;;
    --cv)            CV=true;           shift   ;;
    --n_folds)       N_FOLDS="$2";      shift 2 ;;
    --data_dir)      DATA_DIR="$2";     shift 2 ;;
    --checkpoint)    SAM2_CKPT="$2";    shift 2 ;;
    --output_dir)    OUTPUT_DIR="$2";   shift 2 ;;
    --config)        CONFIG="$2";       shift 2 ;;
    --s3_data)       S3_DATA="$2";      shift 2 ;;
    --s3_checkpoint) S3_CKPT="$2";      shift 2 ;;
    --s3_output)     S3_OUTPUT="$2";    shift 2 ;;
    --nodes)         NODES="$2";        shift 2 ;;
    --node_rank)     NODE_RANK="$2";    shift 2 ;;
    --master_addr)   MASTER_ADDR="$2";  shift 2 ;;
    --master_port)   MASTER_PORT="$2";  shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── S3: pull data ─────────────────────────────────────────────────────────────
if [ -n "$S3_DATA" ]; then
  echo "Syncing data from $S3_DATA → $DATA_DIR"
  aws s3 sync "$S3_DATA" "$DATA_DIR" --no-progress
fi

if [ -n "$S3_CKPT" ]; then
  mkdir -p "$(dirname "$SAM2_CKPT")"
  echo "Downloading checkpoint from $S3_CKPT → $SAM2_CKPT"
  aws s3 cp "$S3_CKPT" "$SAM2_CKPT"
fi

# ── AWS NCCL settings (ignored on local machines) ─────────────────────────────
if [ "$NODES" -gt 1 ]; then
  export NCCL_SOCKET_IFNAME=ens5       # AWS EFA/ENA interface name
  export NCCL_IB_DISABLE=0
  export FI_PROVIDER=efa               # use EFA if available
  export NCCL_DEBUG=WARN
fi

# ── Launcher helper ───────────────────────────────────────────────────────────
TOTAL_GPUS=$(( GPUS * NODES ))

run() {
  local extra_args="$1"
  local base_args="--config $CONFIG --data_dir $DATA_DIR --sam2_checkpoint $SAM2_CKPT $extra_args"

  if [ "$TOTAL_GPUS" -gt 1 ]; then
    torchrun \
      --nproc_per_node="$GPUS" \
      --nnodes="$NODES" \
      --node_rank="$NODE_RANK" \
      --master_addr="$MASTER_ADDR" \
      --master_port="$MASTER_PORT" \
      train.py $base_args
  else
    python train.py $base_args
  fi
}

# ── Run ───────────────────────────────────────────────────────────────────────
echo "========================================================"
echo " GPUs/node: $GPUS  |  Nodes: $NODES  |  Total GPUs: $TOTAL_GPUS"
[ "$CV" = true ] && echo " Mode: ${N_FOLDS}-fold CV" || echo " Mode: single run"
echo " Data:       $DATA_DIR"
echo " Checkpoint: $SAM2_CKPT"
echo " Output:     $OUTPUT_DIR"
echo "========================================================"

if [ "$CV" = true ]; then
  for fold in $(seq 0 $((N_FOLDS - 1))); do
    echo ""
    echo "======== Fold ${fold} / ${N_FOLDS} ========"
    run "--output_dir ${OUTPUT_DIR}/fold${fold} --fold ${fold} --n_folds ${N_FOLDS}"
  done
else
  run "--output_dir ${OUTPUT_DIR}"
fi

# ── S3: push results ──────────────────────────────────────────────────────────
if [ -n "$S3_OUTPUT" ]; then
  echo ""
  echo "Syncing results $OUTPUT_DIR → $S3_OUTPUT"
  aws s3 sync "$OUTPUT_DIR" "$S3_OUTPUT" --no-progress
fi

echo ""
echo "Done. Results in $OUTPUT_DIR"
