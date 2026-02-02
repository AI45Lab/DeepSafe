#!/bin/bash

CONFIG_PATH="$1"
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path>"
  exit 1
fi

CONFIG_PATH=$(readlink -f "$CONFIG_PATH")
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH"
  exit 1
fi

echo "Submitting Salad-Bench EVAL task: $CONFIG_PATH"

rlaunch --charged-group=ai4good2_cpu_task \
  --image=registry.h.pjlab.org.cn/ailab-ai4good2-ai4good2_gpu/zhangbo1:v0130-x86 \
  --max-wait-duration 3600s \
  --enable-sshd=false \
  --memory 32768 \
  --cpu 16 \
  --entrypoint='' \
  --mount=gpfs://gpfs1/dutianyi:/mnt/shared-storage-user/dutianyi \
  --mount=gpfs://gpfs1/ai4good2-share:/mnt/shared-storage-user/ai4good2-share \
  -- bash /mnt/shared-storage-user/dutianyi/Multi-Level-Bionic-Evaluation-Framework/scripts/run_mossbench_eval.sh "$CONFIG_PATH"

