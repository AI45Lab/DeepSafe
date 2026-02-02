#!/bin/bash

CONFIG_PATH="$1"
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path> [num_gpus] [overrides...]"
  exit 1
fi

if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
  NUM_GPUS="$2"
  shift 2
else
  NUM_GPUS=1
  shift
fi
OVERRIDES=("$@")

CONFIG_PATH=$(readlink -f "$CONFIG_PATH")
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH"
  exit 1
fi

if [ $NUM_GPUS -le 4 ]; then
  MEMORY=$((NUM_GPUS * 65536))
  CPU=$((NUM_GPUS * 16))
else
  MEMORY=327680
  CPU=160
fi

echo "Submitting Ch3Ef GEN task: $CONFIG_PATH (num_gpus=$NUM_GPUS)"
echo "  Memory: ${MEMORY}MB, CPU: ${CPU}"

if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi

MAX_WAIT_DURATION="${RLAUNCH_MAX_WAIT_DURATION:-2h0m0s}"

rlaunch --charged-group=ai4good2_gpu \
  --private-machine=yes \
  --max-wait-duration "$MAX_WAIT_DURATION" \
  --enable-sshd=false \
  --image=registry.h.pjlab.org.cn/ailab-ai4good2-ai4good2_gpu/zhangbo1:v0130-x86 \
  --private-machine=group \
  --gpu "$NUM_GPUS" \
  --memory "$MEMORY" \
  --cpu "$CPU" \
  --entrypoint='' \
  --mount=gpfs://gpfs1/dutianyi:/mnt/shared-storage-user/dutianyi \
  --mount=gpfs://gpfs1/ai4good2-share:/mnt/shared-storage-user/ai4good2-share \
  -- bash "/mnt/shared-storage-user/dutianyi/Multi-Level-Bionic-Evaluation-Framework/scripts/run_ch3ef_gen.sh" "$CONFIG_PATH" "${OVERRIDES[@]}"
