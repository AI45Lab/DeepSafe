#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

set -euo pipefail

CONFIG_PATH="$1"
STAGE="${2:-all}"
NUM_GPUS="${3:-1}"
shift 3 || true
OVERRIDES=("$@")

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

if [[ -z "${CONFIG_PATH:-}" ]]; then
  echo "Usage: $0 <config_path> [gen|eval|all] [num_gpus] [overrides...]" >&2
  exit 1
fi

CONFIG_PATH="$(readlink -f "$CONFIG_PATH")"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

STAGE="$(echo "$STAGE" | tr '[:upper:]' '[:lower:]')"
if [[ "$STAGE" != "gen" && "$STAGE" != "eval" && "$STAGE" != "all" ]]; then
  echo "ERROR: invalid stage '$STAGE' (expected gen|eval|all)" >&2
  exit 1
fi

RUN_SCRIPT="$MBEF_ROOT/scripts/run_do_not_answer_gen.sh"
if [[ "$STAGE" == "eval" ]]; then
  RUN_SCRIPT="$MBEF_ROOT/scripts/run_do_not_answer_eval.sh"
elif [[ "$STAGE" == "all" ]]; then
  RUN_SCRIPT="$MBEF_ROOT/scripts/run_do_not_answer_all.sh"
fi

TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
LOG_DIR="$MBEF_ROOT/logs/do_not_answer/${STAGE}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
SUBMIT_LOG_FILE="${LOG_DIR}/${TIMESTAMP}.submit.log"

if [ "$NUM_GPUS" -le 4 ]; then
  MEMORY=32768
  CPU=16
else
  MEMORY=327680
  CPU=160
fi

SUBMIT_PROFILE="${SUBMIT_PROFILE:-gpu}"

submit_one () {
  local charged_group="$1"
  echo "========================================================"
  echo "Submit Do-Not-Answer"
  echo "Stage:         $STAGE"
  echo "Profile:       $SUBMIT_PROFILE"
  echo "Charged group: $charged_group"
  echo "Config:        $CONFIG_PATH"
  echo "Run script:    $RUN_SCRIPT"
  echo "Log file:      $LOG_FILE"
  if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
    echo "Overrides:     ${OVERRIDES[*]}"
  fi
  echo "--------------------------------------------------------"
  echo "You can follow logs after the job starts writing:"
  echo "  tail -f '$LOG_FILE'"
  echo "--------------------------------------------------------"
  echo "Submitting..."

  if [[ "$charged_group" == "ai4good2_cpu_task" ]]; then
    rjob submit --charged-group="$charged_group" \
      --image=registry.h.pjlab.org.cn/ailab-ai4good2-ai4good2_gpu/zhangbo1:v0130-x86 \
      --memory 8192 \
      --cpu 4 \
      --mount=gpfs://gpfs1/zhangbo1:/mnt/shared-storage-user/zhangbo1 \
      --mount=gpfs://gpfs1/ai4good2-share:/mnt/shared-storage-user/ai4good2-share \
      -- bash -lc "set -euo pipefail; bash '$RUN_SCRIPT' '$CONFIG_PATH' ${OVERRIDES[*]} 2>&1 | tee '$LOG_FILE'" \
      2>&1 | tee "$SUBMIT_LOG_FILE"
  else
    rjob submit --charged-group="$charged_group" \
      --private-machine=group \
      --image=registry.h.pjlab.org.cn/ailab-ai4good2-ai4good2_gpu/zhangbo1:v0130-x86 \
      --gpu "$NUM_GPUS" \
      --memory "$MEMORY" \
      --cpu "$CPU" \
      --mount=gpfs://gpfs1/zhangbo1:/mnt/shared-storage-user/zhangbo1 \
      --mount=gpfs://gpfs1/ai4good2-share:/mnt/shared-storage-user/ai4good2-share \
      -- bash -lc "set -euo pipefail; bash '$RUN_SCRIPT' '$CONFIG_PATH' ${OVERRIDES[*]} 2>&1 | tee '$LOG_FILE'" \
      2>&1 | tee "$SUBMIT_LOG_FILE"
  fi

  echo "--------------------------------------------------------"
  echo "Submit receipt saved to: $SUBMIT_LOG_FILE"
  echo "Runtime log path:        $LOG_FILE"
  echo "Follow:                 tail -f '$LOG_FILE'"
  echo "========================================================"
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

