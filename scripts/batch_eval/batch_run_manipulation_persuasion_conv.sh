#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

BASE_CONFIG="${1:-$MBEF_ROOT/configs/eval_tasks/manipulation_persuasion_conv_qwen.yaml}"
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

OPEN_MODELS=()

CLOSED_MODELS=(
  "doubao-seed-1-6-flash-250828|doubao-seed-1-6-flash-250828|http://35.220.164.252:3888/v1|OPENAI_API_KEY|8"
  "deepseek-r1-250528|deepseek-r1-250528|http://35.220.164.252:3888/v1|OPENAI_API_KEY|8"

)

PROTOCOL="${PROTOCOL:-author_score_v1}"

VOTER_MODEL_NAME="${VOTER_MODEL_NAME:-gpt-4o}"
VOTER_API_BASE="${VOTER_API_BASE:-http://35.220.164.252:3888/v1}"
VOTER_API_KEY_MODE="${VOTER_API_KEY_MODE:-OPENAI_API_KEY}"
VOTER_CONCURRENCY="${VOTER_CONCURRENCY:-8}"

submit_gen () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/submit_manipulation_persuasion_conv.sh" "$BASE_CONFIG" gen "$num_gpus" \
    --runner.output_dir "results/manipulation_persuasion_conv_batch/${tag}" \
    "${overrides[@]}"
}

submit_eval () {
  local tag="$1"
  shift 1
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/submit_manipulation_persuasion_conv.sh" "$BASE_CONFIG" eval 1 \
    --runner.output_dir "results/manipulation_persuasion_conv_batch/${tag}" \
    "${overrides[@]}"
}

for spec in "${OPEN_MODELS[@]}"; do
  echo "OPEN_MODELS is disabled in this script. Use bare_run_manipulation_persuasion_conv.sh for local-only models." >&2
  exit 1
done

for spec in "${CLOSED_MODELS[@]}"; do
  IFS="|" read -r tag api_model api_base api_key_mode conc <<< "$spec"
  if [[ -z "${conc:-}" ]]; then conc="4"; fi

  COMMON_OVERRIDES=(
    --evaluator.protocol "$PROTOCOL"
    --model.type APIModel
    --model.model_name "$api_model"
    --model.api_base "$api_base"
    --model.api_key "$api_key_mode"
    --model.mode chat
    --model.concurrency "$conc"
    --model.temperature 0.35
    --model.max_tokens 512

    --evaluator.voter_model_cfg.type APIModel
    --evaluator.voter_model_cfg.model_name "$VOTER_MODEL_NAME"
    --evaluator.voter_model_cfg.api_base "$VOTER_API_BASE"
    --evaluator.voter_model_cfg.api_key "$VOTER_API_KEY_MODE"
    --evaluator.voter_model_cfg.mode chat
    --evaluator.voter_model_cfg.concurrency "$VOTER_CONCURRENCY"
    --evaluator.voter_model_cfg.temperature 0.35
    --evaluator.voter_model_cfg.max_tokens 512
  )

  echo "==================== CLOSED MODEL: $tag ===================="
  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    SUBMIT_PROFILE="cpu_task" submit_gen "$tag" 1 "${COMMON_OVERRIDES[@]}"
    sleep 3
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    SUBMIT_PROFILE="cpu_task" submit_eval "$tag" --evaluator.protocol "$PROTOCOL"
    sleep 3
  fi
done

echo "Done submitting manipulation_persuasion_conv batch."

