#!/bin/bash

CONFIG_PATH="$1"
shift || true
OVERRIDES=("$@")
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path> [overrides...]"
  exit 1
fi

CONFIG_PATH=$(readlink -f "$CONFIG_PATH")
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH"
  exit 1
fi

echo "Submitting Ch3Ef EVAL task: $CONFIG_PATH"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi

bash /mnt/shared-storage-user/dutianyi/Multi-Level-Bionic-Evaluation-Framework/scripts/run_ch3ef_eval.sh "$CONFIG_PATH" "${OVERRIDES[@]}"