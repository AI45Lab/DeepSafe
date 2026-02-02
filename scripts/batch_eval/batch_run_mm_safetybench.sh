#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -eo pipefail

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
BASE_CONFIG="${1:-configs/eval_tasks/mm_safetybench_v01.yaml}"
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
  echo "ERROR: invalid stage '$STAGE' (expected gen|eval|all)" >&2
  exit 1
fi

echo "Base config: $BASE_CONFIG"
echo "Stage: $STAGE"
echo "MBEF: $MBEF_ROOT"

JUDGE_API_BASE="${MM_SAFETYBENCH_JUDGE_API_BASE:-http://35.220.164.252:3888/v1}"
JUDGE_API_BASE="${JUDGE_API_BASE%/}"
JUDGE_API_KEY_MODE="${MM_SAFETYBENCH_JUDGE_API_KEY_MODE:-OPENAI_API_KEY}"
JUDGE_CONCURRENCY="${MM_SAFETYBENCH_JUDGE_CONCURRENCY:-8}"

JUDGE_OVERRIDES=(
  --evaluator.judge_model_cfg.type APIModel
  --evaluator.judge_model_cfg.model_name gpt-4
  --evaluator.judge_model_cfg.api_base "$JUDGE_API_BASE"
  --evaluator.judge_model_cfg.api_key "$JUDGE_API_KEY_MODE"
  --evaluator.judge_model_cfg.mode chat
  --evaluator.judge_model_cfg.concurrency "$JUDGE_CONCURRENCY"
  --evaluator.judge_model_cfg.temperature 0.0
  --evaluator.judge_model_cfg.max_tokens 256
)

OPEN_MODELS=(
  "llava-1.5-7b-hf|/mnt/shared-storage-user/zhangbo1/models/llava-hf/models--llava-hf--llava-1.5-7b-hf|1|0.9"

)

CLOSED_MODELS=(
)

get_eval_num_gpus () {
  echo "0"
}

get_eval_submit_profile () {
  echo "cpu_task"
}

submit_gen () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/submit_mm_safetybench.sh" "$BASE_CONFIG" gen "$num_gpus" \
    --runner.output_dir "results/mm_safetybench_batch/${tag}" \
    "${overrides[@]}"
}

submit_eval () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/submit_mm_safetybench.sh" "$BASE_CONFIG" eval "$num_gpus" \
    --runner.output_dir "results/mm_safetybench_batch/${tag}" \
    "${overrides[@]}" \
    "${JUDGE_OVERRIDES[@]}"
}

PORT_BASE="${MM_SAFETYBENCH_PORT_BASE:-21115}"
PORT_COUNTER=0

for spec in "${OPEN_MODELS[@]}"; do
  IFS="|" read -r tag model_path tp gpu_util <<< "$spec"

  if [[ ! -e "$model_path" ]]; then
    echo "ERROR: model_path not found for tag='$tag': $model_path" >&2
    exit 1
  fi

  MODEL_PORT=$((PORT_BASE + PORT_COUNTER))
  PORT_COUNTER=$((PORT_COUNTER + 1))

  COMMON_OVERRIDES=(
    --model.type APIModel
    --model.model_name "$model_path"
    --model.api_base "http://localhost:${MODEL_PORT}/v1"
    --model.api_key EMPTY
    --model.mode chat
    --model.concurrency 8
    --model.tensor_parallel_size "$tp"
    --model.gpu_memory_utilization "$gpu_util"
    --model.temperature 0.0
    --model.max_tokens 512
  )

  echo "==================== OPEN TARGET: $tag ===================="
  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    submit_gen "$tag" "$tp" "${COMMON_OVERRIDES[@]}"
    sleep 10
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    eval_num_gpus=$(get_eval_num_gpus)
    eval_profile=$(get_eval_submit_profile)
    SUBMIT_PROFILE="$eval_profile" submit_eval "$tag" "$eval_num_gpus"
    sleep 10
  fi
done

for spec in "${CLOSED_MODELS[@]}"; do
  IFS="|" read -r tag model_name api_base api_key_mode conc <<< "$spec"

  COMMON_OVERRIDES=(
    --model.type APIModel
    --model.model_name "$model_name"
    --model.api_base "$api_base"
    --model.api_key "$api_key_mode"
    --model.concurrency "$conc"
    --model.mode chat
    --model.temperature 0.0
    --model.max_tokens 512
  )

  echo "==================== CLOSED TARGET: $tag ===================="
  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    SUBMIT_PROFILE="cpu_task" submit_gen "$tag" 0 "${COMMON_OVERRIDES[@]}"
    sleep 10
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    eval_num_gpus=$(get_eval_num_gpus)
    eval_profile=$(get_eval_submit_profile)
    SUBMIT_PROFILE="$eval_profile" submit_eval "$tag" "$eval_num_gpus"
    sleep 10
  fi
done

echo "All submissions done."

