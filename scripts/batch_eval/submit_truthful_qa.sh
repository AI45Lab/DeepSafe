#!/bin/bash
# NOTE: internal storage paths, registry and cluster mounts were replaced with placeholders for the public release. Fill in your own values before running.

CONFIG_PATH="$1"
STAGE="${2:-all}"
NUM_GPUS="${3:-1}"
shift 3 || true
OVERRIDES=("$@")
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path> [gen|eval|all] [num_gpus] [overrides...]" >&2
  exit 1
fi

CONFIG_PATH=$(readlink -f "$CONFIG_PATH")
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

STAGE=$(echo "$STAGE" | tr '[:upper:]' '[:lower:]')
if [[ "$STAGE" != "gen" && "$STAGE" != "eval" && "$STAGE" != "all" ]]; then
  echo "ERROR: invalid stage '$STAGE' (expected gen|eval|all)" >&2
  exit 1
fi

echo "Submitting TruthfulQA task: $CONFIG_PATH (stage=$STAGE)"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi

RUN_SCRIPT="/path/to/MBEF/scripts/run_truthful_qa.sh"
if [[ "$STAGE" == "gen" ]]; then
  RUN_SCRIPT="/path/to/MBEF/scripts/run_truthful_qa_gen.sh"
elif [[ "$STAGE" == "eval" ]]; then
  RUN_SCRIPT="/path/to/MBEF/scripts/run_truthful_qa_eval.sh"
fi

TIMESTAMP=$(date +"%Y%m%d%H%M%S")
LOG_DIR="/path/to/MBEF/logs/truthful_qa/${STAGE}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
echo "Submitting TruthfulQA task to $LOG_FILE"

if [[ "$STAGE" == "gen" ]]; then
  if [ "$NUM_GPUS" -le 4 ]; then
    MEMORY=32768
    CPU=16
  else
    MEMORY=327680
    CPU=160
  fi
  echo "rjob submit (charged-group=ai4good2_gpu, profile=gpu)"
  rjob submit --charged-group="ai4good2_gpu" \
    --private-machine=group \
    --image=registry.example.com/your-image:tag \
    --gpu "$NUM_GPUS" \
    --memory "$MEMORY" \
    --cpu "$CPU" \
    --mount=gpfs://YOUR_CLUSTER/your-storage:/path/to \
    --mount=gpfs://gpfs1/ai4good2-share:/mnt/shared-storage-user/ai4good2-share \
    -- bash "$RUN_SCRIPT" "$CONFIG_PATH" "${OVERRIDES[@]}" > "$LOG_FILE" 2>&1
elif [[ "$STAGE" == "eval" ]]; then
  echo "rjob submit (charged-group=ai4good2_cpu_task, profile=cpu_task)"
  rjob submit --charged-group="ai4good2_cpu_task" \
    --image=registry.example.com/your-image:tag \
    --memory 8192 \
    --cpu 4 \
    --mount=gpfs://YOUR_CLUSTER/your-storage:/path/to \
    --mount=gpfs://gpfs1/ai4good2-share:/mnt/shared-storage-user/ai4good2-share \
    -- bash "$RUN_SCRIPT" "$CONFIG_PATH" "${OVERRIDES[@]}" > "$LOG_FILE" 2>&1
else
  if [ "$NUM_GPUS" -le 4 ]; then
    MEMORY=32768
    CPU=16
  else
    MEMORY=327680
    CPU=160
  fi
  echo "rjob submit (charged-group=ai4good2_gpu, profile=gpu) [stage=all]"
  rjob submit --charged-group="ai4good2_gpu" \
    --private-machine=group \
    --image=registry.example.com/your-image:tag \
    --gpu "$NUM_GPUS" \
    --memory "$MEMORY" \
    --cpu "$CPU" \
    --mount=gpfs://YOUR_CLUSTER/your-storage:/path/to \
    --mount=gpfs://gpfs1/ai4good2-share:/mnt/shared-storage-user/ai4good2-share \
    -- bash "$RUN_SCRIPT" "$CONFIG_PATH" "${OVERRIDES[@]}" > "$LOG_FILE" 2>&1
fi

