#!/bin/bash

set -eo pipefail

NUM_GPUS="${1:-2}"

if [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] && [ "$NUM_GPUS" -le 4 ]; then
  MEMORY=32768
  CPU=16
else
  MEMORY=327680
  CPU=160
fi

RUN_SCRIPT="/mnt/shared-storage-user/zhangbo1/MBEF/scripts/run_multi_vllm_serve.sh"

TIMESTAMP=$(date +"%Y%m%d%H%M%S")
LOG_DIR="/mnt/shared-storage-user/zhangbo1/MBEF/logs/multi_vllm/serve"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
echo "Submitting multi-vLLM serve job. Log: $LOG_FILE"

rjob submit --charged-group="ai4good2_gpu" \
  --private-machine=group \
  --image=registry.h.pjlab.org.cn/ailab-ai4good2-ai4good2_gpu/zhangbo1:v0130-x86 \
  --gpu "$NUM_GPUS" \
  --memory "$MEMORY" \
  --cpu "$CPU" \
  --mount=gpfs://gpfs1/zhangbo1:/mnt/shared-storage-user/zhangbo1 \
  --mount=gpfs://gpfs1/ai4good2-share:/mnt/shared-storage-user/ai4good2-share \
  -- bash "$RUN_SCRIPT" > "$LOG_FILE" 2>&1

