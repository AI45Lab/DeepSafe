#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

BASE_CONFIG="${1:-configs/eval_tasks/uncontrolled_aird_exp2_saferlhf_simple_v01.yaml}"
STAGE="${2:-all}"

if [[ "$BASE_CONFIG" != /* ]]; then
  BASE_CONFIG="$MBEF_ROOT/$BASE_CONFIG"
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "ERROR: base config not found: $BASE_CONFIG" >&2
  exit 1
fi
BASE_CONFIG="$(readlink -f "$BASE_CONFIG")"

STAGE="$(echo "$STAGE" | tr '[:upper:]' '[:lower:]')"
if [[ "$STAGE" != "gen" && "$STAGE" != "eval" && "$STAGE" != "all" ]]; then
  echo "ERROR: invalid stage '$STAGE' (expected gen|eval|all)"
  exit 1
fi

echo "Base config: $BASE_CONFIG"
echo "Stage: $STAGE"
echo "MBEF: $MBEF_ROOT"

OPEN_MODELS=(
  "Qwen1.5-0.5B-Chat|/mnt/shared-storage-user/zhangbo1/models/Qwen/Qwen1.5-0.5B-Chat|1|0.85|2048"
)

submit_gen () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  SUBMIT_PROFILE="gpu" bash "$MBEF_ROOT/scripts/submit_uncontrolled_aird.sh" "$BASE_CONFIG" gen "$num_gpus" \
    --runner.output_dir "results/uncontrolled_aird_exp2_batch/${tag}" \
    "${overrides[@]}"
}

submit_eval () {
  local tag="$1"
  shift 1
  local overrides=("$@")

  SUBMIT_PROFILE="cpu_task" bash "$MBEF_ROOT/scripts/submit_uncontrolled_aird.sh" "$BASE_CONFIG" eval 1 \
    --runner.output_dir "results/uncontrolled_aird_exp2_batch/${tag}" \
    "${overrides[@]}"
}

for spec in "${OPEN_MODELS[@]}"; do
  IFS="|" read -r tag model_path tp gpu_util max_tokens <<< "$spec"
  if [[ -z "${max_tokens:-}" ]]; then
    max_tokens="2048"
  fi

  if [[ ! -e "$model_path" ]]; then
    echo "ERROR: model_path not found for tag='$tag': $model_path" >&2
    exit 1
  fi

  COMMON_OVERRIDES=(
    --model.type APIModel
    --model.model_name "$model_path"
    --model.api_base http://localhost:21111/v1
    --model.api_key EMPTY
    --model.mode chat
    --model.concurrency 16
    --model.tensor_parallel_size "$tp"
    --model.gpu_memory_utilization "$gpu_util"
    --model.temperature 0.0
    --model.max_tokens "$max_tokens"
  )

  echo "==================== AIRD EXP2 MODEL: $tag ===================="
  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    submit_gen "$tag" "$tp" "${COMMON_OVERRIDES[@]}"
    sleep 5
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    submit_eval "$tag"
    sleep 5
  fi
done

echo "All submissions done."

