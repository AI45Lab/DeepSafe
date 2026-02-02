#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Default to this DeepSafe repo root (two levels above scripts/batch_eval/).
MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
CFG="${1:-$MBEF_ROOT/configs/eval_tasks/manipulation_persuasion_conv_qwen.yaml}"
STAGE="${2:-all}"

if [[ "$CFG" != /* ]]; then
  CFG="$MBEF_ROOT/$CFG"
fi
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG" >&2
  exit 1
fi

STAGE="$(echo "$STAGE" | tr '[:upper:]' '[:lower:]')"
if [[ "$STAGE" != "all" && "$STAGE" != "gen" && "$STAGE" != "eval" ]]; then
  echo "ERROR: invalid stage '$STAGE' (expected all|gen|eval)" >&2
  exit 1
fi

echo "MBEF_ROOT: $MBEF_ROOT"
echo "Config:    $CFG"
echo "Stage:     $STAGE"

VOTER_MODEL_NAME="${VOTER_MODEL_NAME:-gpt-4o}"
VOTER_API_BASE="${VOTER_API_BASE:-http://35.220.164.252:3888/v1}"
VOTER_API_KEY_MODE="${VOTER_API_KEY_MODE:-ENV}"
VOTER_CONCURRENCY="${VOTER_CONCURRENCY:-8}"

API_MODELS=(
)

LOCAL_MODELS=(
)

set_key_for_model() {
  local mode="$1"
  if [[ "$mode" == "ENV" ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY:-}" == "EMPTY" ]]; then
      echo "ERROR: api_key_mode=ENV but OPENAI_API_KEY is empty. Export it first." >&2
      exit 1
    fi
    return 0
  fi
  if [[ "$mode" == ENV:* ]]; then
    local var="${mode#ENV:}"
    if [[ -z "${!var:-}" ]]; then
      echo "ERROR: api_key_mode=$mode but env var '$var' is empty. Export it first." >&2
      exit 1
    fi
    export OPENAI_API_KEY="${!var}"
    return 0
  fi
  export OPENAI_API_KEY="$mode"
}

run_one_api() {
  local tag="$1"
  local model_name="$2"
  local api_base="$3"
  local api_key_mode="$4"
  local conc="$5"

  set_key_for_model "$api_key_mode"
  set_key_for_model "$VOTER_API_KEY_MODE"

  local out_dir="results/manipulation_persuasion_conv_bare/${tag}"

  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    bash "$MBEF_ROOT/scripts/run_manipulation_persuasion_conv_gen.sh" "$CFG" \
      --runner.output_dir "$out_dir" \
      --evaluator.protocol author_score_v1 \
      --model.type APIModel \
      --model.model_name "$model_name" \
      --model.api_base "$api_base" \
      --model.api_key ENV \
      --model.mode chat \
      --model.concurrency "$conc" \
      --evaluator.voter_model_cfg.type APIModel \
      --evaluator.voter_model_cfg.model_name "$VOTER_MODEL_NAME" \
      --evaluator.voter_model_cfg.api_base "$VOTER_API_BASE" \
      --evaluator.voter_model_cfg.api_key "$VOTER_API_KEY_MODE" \
      --evaluator.voter_model_cfg.mode chat \
      --evaluator.voter_model_cfg.concurrency "$VOTER_CONCURRENCY"
  fi

  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    bash "$MBEF_ROOT/scripts/run_manipulation_persuasion_conv_eval.sh" "$CFG" \
      --runner.output_dir "$out_dir" \
      --evaluator.protocol author_score_v1
  fi
}

run_one_local() {
  local tag="$1"
  local model_path="$2"
  local tp="$3"
  local gpu_util="$4"

  if [[ ! -e "$model_path" ]]; then
    echo "ERROR: local model_path not found for tag='$tag': $model_path" >&2
    exit 1
  fi

  set_key_for_model "$VOTER_API_KEY_MODE"

  local out_dir="results/manipulation_persuasion_conv_bare/${tag}"

  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    bash "$MBEF_ROOT/scripts/run_manipulation_persuasion_conv_gen.sh" "$CFG" \
      --runner.output_dir "$out_dir" \
      --evaluator.protocol author_score_v1 \
      --model.type APIModel \
      --model.model_name "$model_path" \
      --model.api_base http://localhost:21111/v1 \
      --model.api_key EMPTY \
      --model.mode chat \
      --model.tensor_parallel_size "$tp" \
      --model.gpu_memory_utilization "$gpu_util" \
      --model.concurrency 32 \
      --evaluator.voter_model_cfg.type APIModel \
      --evaluator.voter_model_cfg.model_name "$VOTER_MODEL_NAME" \
      --evaluator.voter_model_cfg.api_base "$VOTER_API_BASE" \
      --evaluator.voter_model_cfg.api_key "$VOTER_API_KEY_MODE" \
      --evaluator.voter_model_cfg.mode chat \
      --evaluator.voter_model_cfg.concurrency "$VOTER_CONCURRENCY"
  fi

  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    bash "$MBEF_ROOT/scripts/run_manipulation_persuasion_conv_eval.sh" "$CFG" \
      --runner.output_dir "$out_dir" \
      --evaluator.protocol author_score_v1
  fi
}

if [[ ${#API_MODELS[@]} -eq 0 && ${#LOCAL_MODELS[@]} -eq 0 ]]; then
  echo "ERROR: both API_MODELS and LOCAL_MODELS are empty. Edit scripts/bare_run_manipulation_persuasion_conv.sh and add models." >&2
  exit 1
fi

for spec in "${API_MODELS[@]}"; do
  IFS="|" read -r tag model_name api_base api_key_mode conc <<< "$spec"
  if [[ -z "${conc:-}" ]]; then conc="4"; fi
  echo "==================== API MODEL: $tag ===================="
  run_one_api "$tag" "$model_name" "$api_base" "$api_key_mode" "$conc"
done

for spec in "${LOCAL_MODELS[@]}"; do
  IFS="|" read -r tag model_path tp gpu_util <<< "$spec"
  if [[ -z "${tp:-}" ]]; then tp="1"; fi
  if [[ -z "${gpu_util:-}" ]]; then gpu_util="0.85"; fi
  echo "==================== LOCAL MODEL: $tag ===================="
  run_one_local "$tag" "$model_path" "$tp" "$gpu_util"
done

echo "Done."

