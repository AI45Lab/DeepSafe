#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_PATH="$1"
NUM_GPUS="${2:-1}"
shift 2 || true
OVERRIDES=("$@")

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path> [num_gpus] [overrides...]"
  exit 1
fi

CONFIG_PATH=$(readlink -f "$CONFIG_PATH")
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH"
  exit 1
fi

echo "Submitting BeHonest GEN task: $CONFIG_PATH ($NUM_GPUS GPUs)"
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

SUBMIT_PROFILE="${SUBMIT_PROFILE:-gpu}"

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

submit_one () {
  local charged_group="$1"
  echo "rjob submit (charged-group=$charged_group, profile=$SUBMIT_PROFILE)"

  if [[ "$charged_group" == "ai4good2_cpu_task" ]]; then
    rjob submit --charged-group="$charged_group" \
      --namespace=ailab-ai4good2 \
      --image=registry.h.pjlab.org.cn/ailab-ai4good2-ai4good2_gpu/linqihao:v0116-x86 \
      --memory 8192 \
      --cpu 4 \
      --mount=gpfs://gpfs1/linqihao:/mnt/shared-storage-user/linqihao \
      --mount=gpfs://gpfs1/ai4good2-share:/mnt/shared-storage-user/ai4good2-share \
      -- bash "$MBEF_ROOT/scripts/run_behonest_gen.sh" "$CONFIG_PATH" "${OVERRIDES[@]}"
  else
    rjob submit --charged-group="$charged_group" \
      --private-machine=group \
      --image=registry.h.pjlab.org.cn/ailab-ai4good2-ai4good2_gpu/linqihao:v0116-x86 \
      --gpu "$NUM_GPUS" \
      --memory "$MEMORY" \
      --cpu "$CPU" \
      --mount=gpfs://gpfs1/linqihao:/mnt/shared-storage-user/linqihao \
      --mount=gpfs://gpfs1/ai4good2-share:/mnt/shared-storage-user/ai4good2-share \
      --host-network=true \
      -e DISTRIBUTED_JOB=true \
      -- bash "$MBEF_ROOT/scripts/run_behonest_gen.sh" "$CONFIG_PATH" "${OVERRIDES[@]}"
  fi
}

case "$SUBMIT_PROFILE" in
  gpu)
    submit_one "ai4good1_gpu"
    ;;
  cpu_task|cpu)
    submit_one "ai4good2_cpu_task"
    ;;
  *)
    echo "ERROR: unknown SUBMIT_PROFILE='$SUBMIT_PROFILE' (expected gpu|cpu_task)" >&2
    exit 1
    ;;
esac
