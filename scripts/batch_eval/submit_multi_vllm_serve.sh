#!/bin/bash
# NOTE: internal storage paths, registry and cluster mounts were replaced with placeholders for the public release. Fill in your own values before running.

set -eo pipefail

NUM_GPUS="${1:-2}"

if [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] && [ "$NUM_GPUS" -le 4 ]; then
  MEMORY=32768
  CPU=16
else
  MEMORY=327680
  CPU=160
fi

RUN_SCRIPT="/path/to/MBEF/scripts/run_multi_vllm_serve.sh"

TIMESTAMP=$(date +"%Y%m%d%H%M%S")
LOG_DIR="/path/to/MBEF/logs/multi_vllm/serve"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
echo "Submitting multi-vLLM serve job. Log: $LOG_FILE"

rjob submit --charged-group="ai4good2_gpu" \
  --private-machine=group \
  --image=registry.example.com/your-image:tag \
  --gpu "$NUM_GPUS" \
  --memory "$MEMORY" \
  --cpu "$CPU" \
  --mount=gpfs://YOUR_CLUSTER/your-storage:/path/to \
  --mount=gpfs://gpfs1/ai4good2-share:/mnt/shared-storage-user/ai4good2-share \
  -- bash "$RUN_SCRIPT" > "$LOG_FILE" 2>&1

