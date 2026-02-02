#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -eo pipefail

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

BASE_CONFIG="${1:-configs/eval_tasks/fake_alignment_llama_3_8b.yaml}"
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
  --evaluator.judge_model_cfg.model_name gpt-4o
  --evaluator.judge_model_cfg.api_base "$JUDGE_API_BASE"
  --evaluator.judge_model_cfg.api_key "$JUDGE_API_KEY_MODE"
  --evaluator.judge_model_cfg.concurrency "$JUDGE_CONCURRENCY"
  --evaluator.judge_model_cfg.mode chat
  --evaluator.judge_model_cfg.temperature 0.0
  --evaluator.judge_model_cfg.max_tokens 64
)

OPEN_MODELS=(

)

CLOSED_MODELS=(
  "gpt-5.2|gpt-5.2|http://35.220.164.252:3888/v1|OPENAI_API_KEY|8"
  "deepseek-r1-250528|deepseek-r1-250528|http://35.220.164.252:3888/v1|OPENAI_API_KEY|8"
)

get_eval_num_gpus () {
  local judge_type=""
  local judge_tp="1"

  local i=0
  while [[ $i -lt ${#JUDGE_OVERRIDES[@]} ]]; do
    if [[ "${JUDGE_OVERRIDES[$i]}" == "--evaluator.judge_model_cfg.type" ]]; then
      judge_type="${JUDGE_OVERRIDES[$((i+1))]}"
    elif [[ "${JUDGE_OVERRIDES[$i]}" == "--evaluator.judge_model_cfg.tensor_parallel_size" ]]; then
      judge_tp="${JUDGE_OVERRIDES[$((i+1))]}"
    fi
    ((i++))
  done

  if [[ "$judge_type" == "VLLMLocalModel" ]]; then
    echo "$judge_tp"
  else
    echo "1"
  fi
}

submit_gen () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/submit_fake_alignment.sh" "$BASE_CONFIG" gen "$num_gpus" \
    --runner.output_dir "results/fake_alignment_batch/${tag}" \
    --runner.use_evaluator_gen true \
    "${overrides[@]}"
}

submit_eval () {
  local tag="$1"
  local num_gpus="$2"
  shift 2 || true

  bash "$MBEF_ROOT/scripts/submit_fake_alignment.sh" "$BASE_CONFIG" eval "$num_gpus" \
    --runner.output_dir "results/fake_alignment_batch/${tag}" \
    "${JUDGE_OVERRIDES[@]}"
}

for spec in "${OPEN_MODELS[@]}"; do
  IFS="|" read -r tag model_path tp gpu_util <<< "$spec"

  if [[ ! -e "$model_path" ]]; then
    echo "ERROR: model_path not found for tag='$tag': $model_path" >&2
    exit 1
  fi

  num_gpus="${tp}"

  COMMON_OVERRIDES=(
    --model.type APIModel
    --model.model_name "$model_path"
    --model.api_base http://localhost:21111/v1
    --model.api_key EMPTY
    --model.mode chat
    --model.concurrency 32
    --model.tensor_parallel_size "$tp"
    --model.gpu_memory_utilization "$gpu_util"
    --model.temperature 0.0
    --model.max_tokens 512
  )

  echo "==================== OPEN MODEL: $tag ===================="
  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    submit_gen "$tag" "$num_gpus" "${COMMON_OVERRIDES[@]}"
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    eval_num_gpus=$(get_eval_num_gpus)
    submit_eval "$tag" "$eval_num_gpus" "${COMMON_OVERRIDES[@]}"
  fi
done

for spec in "${CLOSED_MODELS[@]}"; do
  IFS="|" read -r tag api_model api_base api_key_mode conc <<< "$spec"

  COMMON_OVERRIDES=(
    --model.type APIModel
    --model.model_name "$api_model"
    --model.api_base "$api_base"
    --model.api_key "$api_key_mode"
    --model.concurrency "$conc"
    --model.mode chat
    --model.temperature 0.0
    --model.max_tokens 512
  )

  echo "==================== CLOSED MODEL: $tag ===================="
  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    SUBMIT_PROFILE="cpu_task" submit_gen "$tag" 1 "${COMMON_OVERRIDES[@]}"
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    eval_num_gpus=$(get_eval_num_gpus)
    SUBMIT_PROFILE="cpu_task" submit_eval "$tag" "$eval_num_gpus" "${COMMON_OVERRIDES[@]}"
  fi
done

echo "All submissions done."

