#!/bin/bash
# NOTE: internal storage paths, registry and cluster mounts were replaced with placeholders for the public release. Fill in your own values before running.

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
  --image=registry.example.com/your-image:tag \
  --max-wait-duration 3600s \
  --enable-sshd=false \
  --memory 32768 \
  --cpu 16 \
  --entrypoint='' \
  --mount=gpfs://gpfs1/dutianyi:/mnt/shared-storage-user/dutianyi \
  --mount=gpfs://gpfs1/ai4good2-share:/mnt/shared-storage-user/ai4good2-share \
  -- bash /mnt/shared-storage-user/dutianyi/Multi-Level-Bionic-Evaluation-Framework/scripts/run_mossbench_eval.sh "$CONFIG_PATH"

