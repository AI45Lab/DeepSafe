#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_CONFIG="${1:-$MBEF_ROOT/configs/eval_tasks/salad_judge_v01_qwen1.5-0.5b_vllm_local_proguard.yaml}"
STAGE="${2:-all}"
shift 2 || true
GLOBAL_OVERRIDES=("$@")

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
echo "MBEF:  $MBEF_ROOT"
if [[ ${#GLOBAL_OVERRIDES[@]} -gt 0 ]]; then
  echo "Global overrides: ${GLOBAL_OVERRIDES[*]}"
fi

JUDGE_OVERRIDES=(
  --evaluator.judge_model_cfg.type APIModel
  --evaluator.judge_model_cfg.model_name /path/to/your/judge/model
  --evaluator.judge_model_cfg.api_base http://localhost:21112/v1
  --evaluator.judge_model_cfg.api_key EMPTY
  --evaluator.judge_model_cfg.mode chat
  --evaluator.judge_model_cfg.temperature 0.0
  --evaluator.judge_model_cfg.max_tokens 256
  --evaluator.judge_model_cfg.tensor_parallel_size 1
  --evaluator.judge_model_cfg.gpu_memory_utilization 0.85
)

OPEN_MODELS=(
)

CLOSED_MODELS=(
)

run_gen () {
  local tag="$1"
  shift
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/run_salad_gen_local.sh" "$BASE_CONFIG" \
    --runner.output_dir "results/salad_batch/${tag}" \
    "${overrides[@]}" \
    "${GLOBAL_OVERRIDES[@]}"
}

run_eval () {
  local tag="$1"
  shift
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/run_salad_eval_local.sh" "$BASE_CONFIG" \
    --runner.output_dir "results/salad_batch/${tag}" \
    "${JUDGE_OVERRIDES[@]}" \
    "${overrides[@]}" \
    "${GLOBAL_OVERRIDES[@]}"
}

should_run_tag () {
  local tag="$1"
  if [[ -z "${ONLY_TAG:-}" ]]; then
    return 0
  fi
  [[ "$tag" == "$ONLY_TAG" ]]
}

for spec in "${OPEN_MODELS[@]}"; do
  IFS="|" read -r tag model_path tp gpu_util <<< "$spec"
  if ! should_run_tag "$tag"; then
    continue
  fi

  if [[ -z "$tag" || -z "$model_path" ]]; then
    echo "ERROR: invalid OPEN_MODELS spec: '$spec'" >&2
    exit 1
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
    --model.tensor_parallel_size "${tp:-1}"
    --model.gpu_memory_utilization "${gpu_util:-0.85}"
    --model.temperature 0.0
    --model.max_tokens 512
  )

  echo "==================== OPEN MODEL: $tag ===================="
  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    run_gen "$tag" "${COMMON_OVERRIDES[@]}"
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    run_eval "$tag"
  fi
done

for spec in "${CLOSED_MODELS[@]}"; do
  IFS="|" read -r tag api_model api_base api_key_mode conc <<< "$spec"
  if ! should_run_tag "$tag"; then
    continue
  fi

  if [[ -z "$tag" || -z "$api_model" || -z "$api_base" ]]; then
    echo "ERROR: invalid CLOSED_MODELS spec: '$spec'" >&2
    exit 1
  fi

  if [[ -z "${conc:-}" ]]; then
    conc="4"
  fi

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
    run_gen "$tag" "${COMMON_OVERRIDES[@]}"
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    run_eval "$tag"
  fi
done

echo "All done."

