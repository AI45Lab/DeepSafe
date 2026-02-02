#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -eo pipefail

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

BASE_CONFIG="${1:-configs/eval_tasks/uncontrolled_aird_exp1_actor_v01.yaml}"
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

JUDGE_API_BASE="${AIRD_JUDGE_API_BASE:-http://35.220.164.252:3888/v1}"
JUDGE_API_BASE="${JUDGE_API_BASE%/}"

JUDGE_API_KEY="${AIRD_JUDGE_API_KEY:-OPENAI_API_KEY}"

JUDGE_OVERRIDES=(
  --evaluator.judge_model_cfg.type APIModel
  --evaluator.judge_model_cfg.model_name gpt-4o
  --evaluator.judge_model_cfg.api_base "$JUDGE_API_BASE"
  --evaluator.judge_model_cfg.api_key "$JUDGE_API_KEY"
  --evaluator.judge_model_cfg.concurrency 16
  --evaluator.judge_model_cfg.mode chat
  --evaluator.judge_model_cfg.timeout 60
  --evaluator.judge_gen_kwargs.temperature 0.0
  --evaluator.judge_gen_kwargs.max_tokens 256
)

if [[ "${AIRD_DEBUG_JUDGE:-}" == "1" || "${AIRD_DEBUG_JUDGE:-}" == "true" ]]; then
  JUDGE_OVERRIDES+=(--evaluator.debug_judge true)
  JUDGE_OVERRIDES+=(--evaluator.debug_judge_max_chars "${AIRD_DEBUG_JUDGE_MAX_CHARS:-800}")
fi

get_eval_num_gpus () {
  echo "1"
}

get_eval_submit_profile () {
  echo "cpu_task"
}

OPEN_MODELS=(

  "Qwen2.5-72B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen2.5-72B-Instruct|2|0.85|1024"
  "Llama-3.3-70B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/meta-llama/Llama-3.3-70B-Instruct|2|0.85|1024"
  
  "InternLM3-8B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/internlm/internlm3-8b-instruct|1|0.9|256"
  "Gemma-3-27B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/google/gemma-3-27b-it|1|0.85|1024"
  "Mistral-Small-24B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/mistralai/Mistral-Small-24B-Instruct-2501|1|0.85|1024"

  "GLM-4.5-Air|/mnt/shared-storage-user/ai4good2-share/models/zai-org/GLM-4.5-Air|4|0.9|1024"
  "Qwen3-30B-A3B-Thinking-2507|/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen3-30B-A3B-Instruct-2507|4|0.9|1024"
)

CLOSED_MODELS=(
  "gpt-4o|gpt-4o|http://35.220.164.252:3888/v1|OPENAI_API_KEY|8"
  "gemini-3-flash-preview|gemini-3-flash-preview|http://35.220.164.252:3888/v1|OPENAI_API_KEY|8"
  "doubao-seed-1-6-flash-250828|doubao-seed-1-6-flash-250828|http://35.220.164.252:3888/v1|OPENAI_API_KEY|8"
  "deepseek-r1-250528|deepseek-r1-250528|http://35.220.164.252:3888/v1|OPENAI_API_KEY|8"
)

submit_gen () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/submit_uncontrolled_aird.sh" "$BASE_CONFIG" gen "$num_gpus" \
    --runner.output_dir "results/uncontrolled_aird_exp1_batch/${tag}" \
    "${overrides[@]}"
}

submit_eval () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/submit_uncontrolled_aird.sh" "$BASE_CONFIG" eval "$num_gpus" \
    --runner.output_dir "results/uncontrolled_aird_exp1_batch/${tag}" \
    "${JUDGE_OVERRIDES[@]}" \
    "${overrides[@]}"
}

for spec in "${OPEN_MODELS[@]}"; do
  IFS="|" read -r tag model_path tp gpu_util max_tokens <<< "$spec"
  if [[ -z "${max_tokens:-}" ]]; then
    max_tokens="512"
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
    --model.concurrency 32
    --model.tensor_parallel_size "$tp"
    --model.gpu_memory_utilization "$gpu_util"
    --model.temperature 0.0
    --model.max_tokens "$max_tokens"
  )

  echo "==================== AIRD EXP1 MODEL: $tag ===================="
  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    SUBMIT_PROFILE="gpu" submit_gen "$tag" "$tp" "${COMMON_OVERRIDES[@]}"
    sleep 5
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    eval_num_gpus=$(get_eval_num_gpus)
    eval_profile=$(get_eval_submit_profile)
    SUBMIT_PROFILE="$eval_profile" submit_eval "$tag" "$eval_num_gpus"
    sleep 5
  fi
done

for spec in "${CLOSED_MODELS[@]}"; do
  IFS="|" read -r tag model_name api_base api_key_mode concurrency <<< "$spec"
  if [[ -z "${concurrency:-}" ]]; then concurrency="8"; fi
  closed_max_tokens="512"
  if [[ "$model_name" == "gpt-5.2" || "$model_name" == gpt-5.2-* || "$tag" == "gpt-5.2"  || "$tag" == "kimi-k2-thinking-turbo" ]]; then
    closed_max_tokens="2048"
  fi

  COMMON_OVERRIDES=(
    --model.type APIModel
    --model.model_name "$model_name"
    --model.api_base "$api_base"
    --model.api_key "$api_key_mode"
    --model.mode chat
    --model.concurrency "$concurrency"
    --model.temperature 0.0
    --model.max_tokens "$closed_max_tokens"
  )

  echo "==================== AIRD EXP1 CLOSED MODEL: $tag ===================="
  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    SUBMIT_PROFILE="cpu_task" submit_gen "$tag" 1 "${COMMON_OVERRIDES[@]}"
    sleep 5
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    eval_num_gpus=$(get_eval_num_gpus)
    eval_profile=$(get_eval_submit_profile)
    SUBMIT_PROFILE="$eval_profile" submit_eval "$tag" "$eval_num_gpus" "${COMMON_OVERRIDES[@]}"
    sleep 5
  fi
done

echo "All submissions done."

