#!/bin/bash

CONFIG_PATH="$1"
STAGE="${2:-all}"
NUM_GPUS="${3:-1}"
shift 3 || true
OVERRIDES=("$@")
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path> [gen|eval|all]"
  exit 1
fi

CONFIG_PATH=$(readlink -f "$CONFIG_PATH")
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH"
  exit 1
fi

STAGE=$(echo "$STAGE" | tr '[:upper:]' '[:lower:]')
if [[ "$STAGE" != "gen" && "$STAGE" != "eval" && "$STAGE" != "all" ]]; then
  echo "ERROR: invalid stage '$STAGE' (expected gen|eval|all)"
  exit 1
fi

echo "Submitting Salad-Bench task: $CONFIG_PATH (stage=$STAGE)"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi

if [ $NUM_GPUS -le 4 ]; then
  MEMORY=32768
  CPU=16
else
  MEMORY=327680
  CPU=160
fi

RUN_SCRIPT="/mnt/shared-storage-user/zhangbo1/MBEF/scripts/run_salad.sh"
if [[ "$STAGE" == "gen" ]]; then
  RUN_SCRIPT="/mnt/shared-storage-user/guoshaoxiong/Multi-Level-Bionic-Evaluation-Framework/scripts/run_argus_gen.sh"
elif [[ "$STAGE" == "eval" ]]; then
  RUN_SCRIPT="/mnt/shared-storage-user/guoshaoxiong/Multi-Level-Bionic-Evaluation-Framework/scripts/run_argus_eval.sh"
fi

echo "rlaunch $RUN_SCRIPT with $CONFIG_PATH"
rlaunch --charged-group=ai4good2_gpu \
  --private-machine=yes \
  --image=registry.h.pjlab.org.cn/ailab-ai4good2-ai4good2_gpu/zhangbo1:v0130-x86 \
  --max-wait-duration 3600s \
  --enable-sshd=false \
  --private-machine=group \
  --gpu "$NUM_GPUS" \
  --memory "$MEMORY" \
  --cpu "$CPU" \
  --entrypoint='' \
  --mount=gpfs://gpfs1/guoshaoxiong:/mnt/shared-storage-user/guoshaoxiong \
  --mount=gpfs://gpfs1/ai4good2-share:/mnt/shared-storage-user/ai4good2-share \
  -- bash "$RUN_SCRIPT" "$CONFIG_PATH" "${OVERRIDES[@]}"
