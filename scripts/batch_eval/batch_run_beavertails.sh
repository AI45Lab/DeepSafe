#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -eo pipefail

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

BASE_CONFIG="${1:-$MBEF_ROOT/configs/eval_tasks/beavertails_proguard_judge_v01.yaml}"
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

JUDGE_OVERRIDES=(
  --evaluator.judge_model_cfg.type APIModel
  --evaluator.judge_model_cfg.model_name /mnt/shared-storage-user/ai4good2-share/models/ProGuard-7B
  --evaluator.judge_model_cfg.api_base http://localhost:21112/v1
  --evaluator.judge_model_cfg.api_key EMPTY
  --evaluator.judge_model_cfg.mode chat
  --evaluator.judge_model_cfg.temperature 0.0
  --evaluator.judge_model_cfg.max_tokens 256
  --evaluator.judge_model_cfg.concurrency 16
)

OPEN_MODELS=(
  "Qwen2.5-72B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen2.5-72B-Instruct|2|0.85"
  "Llama-3.3-70B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/meta-llama/Llama-3.3-70B-Instruct|2|0.85"
  
  "InternLM3-8B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/internlm/internlm3-8b-instruct|1|0.9"
  "Gemma-3-27B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/google/gemma-3-27b-it|1|0.85"
  "Mistral-Small-24B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/mistralai/Mistral-Small-24B-Instruct-2501|1|0.85"

  "GLM-4.5-Air|/mnt/shared-storage-user/ai4good2-share/models/zai-org/GLM-4.5-Air|4|0.9"
  "Qwen3-30B-A3B-Thinking-2507|/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen3-30B-A3B-Instruct-2507|4|0.9"
  
)

CLOSED_MODELS=(
  "gpt-4o-mini|gpt-4o-mini|http://35.220.164.252:3888/v1|OPENAI_API_KEY|8"
  "gpt-4o|gpt-4o|http://35.220.164.252:3888/v1|OPENAI_API_KEY|8"
  "gpt-5.2|gpt-5.2|http://35.220.164.252:3888/v1|OPENAI_API_KEY|8"
  "gemini-3-flash-preview|gemini-3-flash-preview|http://35.220.164.252:3888/v1|OPENAI_API_KEY|8"
  "claude-sonnet-4-5-20250929|claude-sonnet-4-5-20250929|http://35.220.164.252:3888/v1|OPENAI_API_KEY|8"

  "doubao-seed-1-6-flash-250828|doubao-seed-1-6-flash-250828|http://35.220.164.252:3888/v1|OPENAI_API_KEY|8"
  "deepseek-r1-250528|deepseek-r1-250528|http://35.220.164.252:3888/v1|OPENAI_API_KEY|8"

)

submit_gen () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/submit_beavertails.sh" "$BASE_CONFIG" gen "$num_gpus" \
    --runner.output_dir "results/beavertails_batch/${tag}" \
    "${overrides[@]}"
}

submit_eval () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/submit_beavertails.sh" "$BASE_CONFIG" eval "$num_gpus" \
    --runner.output_dir "results/beavertails_batch/${tag}" \
    "${JUDGE_OVERRIDES[@]}" \
    "${overrides[@]}"
}

for spec in "${OPEN_MODELS[@]}"; do
  IFS="|" read -r tag model_path tp gpu_util <<< "$spec"

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
    --model.max_tokens 512
  )

  echo "==================== OPEN MODEL: $tag ===================="
  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    SUBMIT_PROFILE="gpu" submit_gen "$tag" "$tp" "${COMMON_OVERRIDES[@]}"
    sleep 3
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    SUBMIT_PROFILE="gpu" submit_eval "$tag" 1
    sleep 3
  fi
done

for spec in "${CLOSED_MODELS[@]}"; do
  IFS="|" read -r tag api_model api_base api_key_mode conc <<< "$spec"
  if [[ -z "${conc:-}" ]]; then conc="4"; fi

  COMMON_OVERRIDES=(
    --model.type APIModel
    --model.model_name "$api_model"
    --model.api_base "$api_base"
    --model.api_key "$api_key_mode"
    --model.mode chat
    --model.concurrency "$conc"
    --model.temperature 0.0
    --model.max_tokens 512
  )

  echo "==================== CLOSED MODEL: $tag ===================="
  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    SUBMIT_PROFILE="cpu_task" submit_gen "$tag" 1 "${COMMON_OVERRIDES[@]}"
    sleep 3
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    SUBMIT_PROFILE="gpu" submit_eval "$tag" 1
    sleep 3
  fi
done

echo "Done submitting BeaverTails batch."

