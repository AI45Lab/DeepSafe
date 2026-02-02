#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

BASE_CONFIG="${1:-configs/eval_tasks/medhallu_pqa_labeled_v01.yaml}"
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
  "Llama-3.1-8B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/meta-llama/Llama-3.1-8B-Instruct|1|0.85"
)

CLOSED_MODELS=(
)

submit_gen () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")
  bash "$MBEF_ROOT/scripts/submit_medhallu.sh" "$BASE_CONFIG" gen "$num_gpus" \
    --runner.output_dir "results/medhallu_batch/${tag}" \
    "${overrides[@]}"
}

submit_eval () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")
  bash "$MBEF_ROOT/scripts/submit_medhallu.sh" "$BASE_CONFIG" eval "$num_gpus" \
    --runner.output_dir "results/medhallu_batch/${tag}" \
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
    --model.api_base http://localhost:21151/v1
    --model.api_key EMPTY
    --model.mode chat
    --model.concurrency 32
    --model.tensor_parallel_size "$tp"
    --model.gpu_memory_utilization "$gpu_util"
    --model.temperature 0.3
    --model.top_p 0.95
    --model.max_tokens 512
  )

  echo "==================== OPEN MODEL: $tag ===================="
  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    SUBMIT_PROFILE="gpu" submit_gen "$tag" "$tp" "${COMMON_OVERRIDES[@]}"
    sleep 5
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    SUBMIT_PROFILE="cpu_task" submit_eval "$tag" 1
    sleep 5
  fi
done

for spec in "${CLOSED_MODELS[@]}"; do
  IFS="|" read -r tag api_model api_base api_key conc <<< "$spec"
  if [[ -z "${conc:-}" ]]; then conc="4"; fi

  COMMON_OVERRIDES=(
    --model.type APIModel
    --model.model_name "$api_model"
    --model.api_base "$api_base"
    --model.api_key "$api_key"
    --model.concurrency "$conc"
    --model.mode chat
    --model.temperature 0.3
    --model.top_p 0.95
    --model.max_tokens 512
  )

  echo "==================== CLOSED MODEL: $tag ===================="
  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    SUBMIT_PROFILE="cpu_task" submit_gen "$tag" 1 "${COMMON_OVERRIDES[@]}"
    sleep 5
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    SUBMIT_PROFILE="cpu_task" submit_eval "$tag" 1
    sleep 5
  fi
done

echo "All submissions done."

