#!/bin/bash
# NOTE: internal storage paths, registry and cluster mounts were replaced with placeholders for the public release. Fill in your own values before running.

set -eo pipefail

CONFIG_PATH="$1"
NUM_GPUS="${2:-1}"
shift 2 || true
OVERRIDES=("$@")

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path> [num_gpus] [overrides...]" >&2
  exit 1
fi

CONFIG_PATH="$(readlink -f "$CONFIG_PATH")"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

echo "Submitting WMDP task: $CONFIG_PATH"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi

if [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] && [ "$NUM_GPUS" -le 4 ]; then
  MEMORY=32768
  CPU=16
else
  MEMORY=327680
  CPU=160
fi

RUN_SCRIPT="/path/to/MBEF/scripts/run_wmdp.sh"

TIMESTAMP=$(date +"%Y%m%d%H%M%S")
LOG_DIR="/path/to/MBEF/logs/wmdp/all"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
echo "Submitting WMDP to $LOG_FILE"

SUBMIT_PROFILE="${SUBMIT_PROFILE:-gpu}"

submit_one () {
  local charged_group="$1"
  echo "rjob submit (charged-group=$charged_group, profile=$SUBMIT_PROFILE)"

  if [[ "$charged_group" == "ai4good2_cpu_task" ]]; then
    rjob submit --charged-group="$charged_group" \
      --image=registry.example.com/your-image:tag \
      --memory 8192 \
      --cpu 4 \
      --mount=gpfs://YOUR_CLUSTER/your-storage:/path/to \
      --mount=gpfs://gpfs1/ai4good2-share:/mnt/shared-storage-user/ai4good2-share \
      -- bash "$RUN_SCRIPT" "$CONFIG_PATH" "${OVERRIDES[@]}" > "$LOG_FILE" 2>&1
  else
    rjob submit --charged-group="$charged_group" \
      --private-machine=group \
      --image=registry.example.com/your-image:tag \
      --gpu "$NUM_GPUS" \
      --memory "$MEMORY" \
      --cpu "$CPU" \
      --mount=gpfs://YOUR_CLUSTER/your-storage:/path/to \
      --mount=gpfs://gpfs1/ai4good2-share:/mnt/shared-storage-user/ai4good2-share \
      -- bash "$RUN_SCRIPT" "$CONFIG_PATH" "${OVERRIDES[@]}" > "$LOG_FILE" 2>&1
  fi
}

case "$SUBMIT_PROFILE" in
  gpu)
    submit_one "ai4good2_gpu"
    ;;
  cpu_task|cpu)
    submit_one "ai4good2_cpu_task"
    ;;
  *)
    echo "ERROR: unknown SUBMIT_PROFILE='$SUBMIT_PROFILE' (expected gpu|cpu_task)" >&2
    exit 1
    ;;
esac

