#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -eo pipefail

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

BASE_CONFIG="${1:-configs/eval_tasks/do_not_answer_v01.yaml}"
STAGE="${2:-all}"
LIMIT="${3:-0}"

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
echo "Limit: $LIMIT"

DNA_DATASET_CACHE="/mnt/shared-storage-user/ai4good2-share/datasets/do-not-answer"
if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
  if [[ ! -d "$DNA_DATASET_CACHE" ]]; then
    echo "ERROR: Do-Not-Answer dataset cache not found: $DNA_DATASET_CACHE" >&2
    echo "Please download it once on an egress-enabled node (cpu_task), e.g.:" >&2
    echo "  bash $MBEF_ROOT/scripts/download_do_not_answer_dataset.sh $DNA_DATASET_CACHE" >&2
    exit 1
  fi
fi

DATASET_OVERRIDES=()
if [[ -d "$DNA_DATASET_CACHE" ]]; then
  DATASET_OVERRIDES+=( --dataset.path "$DNA_DATASET_CACHE" )
fi
if [[ -n "${LIMIT:-}" && "${LIMIT}" != "0" ]]; then
  DATASET_OVERRIDES+=( --dataset.limit "$LIMIT" )
fi

OPEN_MODELS=(
  "Qwen1.5-0.5B-Chat|/mnt/shared-storage-user/zhangbo1/models/Qwen/Qwen1.5-0.5B-Chat|1|0.80"
  "Llama-2-7b-chat-hf|/mnt/shared-storage-user/ai4good2-share/models/meta-llama/Llama-2-7b-chat-hf|1|0.85"
)

CLOSED_MODELS=(
)

submit_gen () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/submit_do_not_answer.sh" "$BASE_CONFIG" gen "$num_gpus" \
    --runner.output_dir "results/do_not_answer_batch/${tag}" \
    "${overrides[@]}"
}

submit_eval () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/submit_do_not_answer.sh" "$BASE_CONFIG" eval "$num_gpus" \
    --runner.output_dir "results/do_not_answer_batch/${tag}" \
    "${overrides[@]}"
}

for spec in "${OPEN_MODELS[@]}"; do
  IFS="|" read -r tag model_path tp gpu_util <<< "$spec"

  if [[ ! -e "$model_path" ]]; then
    echo "ERROR: model_path not found for tag='$tag': $model_path" >&2
    exit 1
  fi
  if [[ -z "${tp:-}" ]]; then tp="1"; fi
  if [[ -z "${gpu_util:-}" ]]; then gpu_util="0.85"; fi

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
    SUBMIT_PROFILE="gpu" submit_gen "$tag" "$tp" "${DATASET_OVERRIDES[@]}" "${COMMON_OVERRIDES[@]}"
    sleep 10
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    SUBMIT_PROFILE="gpu" submit_eval "$tag" 1 "${DATASET_OVERRIDES[@]}"
    sleep 10
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
    --model.concurrency "$conc"
    --model.mode chat
    --model.temperature 0.0
    --model.max_tokens 512
  )

  echo "==================== CLOSED MODEL: $tag ===================="
  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    SUBMIT_PROFILE="cpu_task" submit_gen "$tag" 1 "${COMMON_OVERRIDES[@]}"
    sleep 10
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    SUBMIT_PROFILE="gpu" submit_eval "$tag" 1 "${DATASET_OVERRIDES[@]}"
    sleep 10
  fi
done

echo "All submissions done."

