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

rlaunch --charged-group=ai4good2_gpu \
  --private-machine=yes \
  --image=registry.example.com/your-image:tag \
  --max-wait-duration 3600s \
  --enable-sshd=false \
  --private-machine=group \
  --gpu 1 \
  --memory 32768 \
  --cpu 16 \
  --entrypoint='' \
  --mount=gpfs://YOUR_CLUSTER/your-storage:/path/to \
  -- bash /path/to/MBEF/scripts/run_salad_eval.sh "$CONFIG_PATH"

