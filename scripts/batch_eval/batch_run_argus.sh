#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -eo pipefail

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

BASE_CONFIG="${1:-configs/eval_tasks/argus.yaml}"
STAGE="${2:-eval}"

if [[ "$BASE_CONFIG" != /* ]]; then
  BASE_CONFIG="$MBEF_ROOT/$BASE_CONFIG"
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "ERROR: base config not found: $BASE_CONFIG" >&2
  echo "Hint: pass an absolute yaml path, e.g.:" >&2
  echo "  bash $MBEF_ROOT/scripts/batch_run_argus_batch.sh $MBEF_ROOT/configs/eval_tasks/xxx.yaml all" >&2
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
  "Qwen2.5-VL-32B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen2.5-VL-32B-Instruct/|1|0.9"
)

CLOSED_MODELS=(
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

  bash "$MBEF_ROOT/scripts/submit_argus.sh" "$BASE_CONFIG" gen "$num_gpus" \
    --runner.output_dir "results/argus_batch/${tag}" \
    "${overrides[@]}"
}

submit_eval () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/submit_argus.sh" "$BASE_CONFIG" eval "$num_gpus" \
    --runner.output_dir "results/argus_batch/${tag}" \
    "${overrides[@]}" \
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
    --model.type ArgusBase
    --model.model_name "$model_path"
    --model.api_base http://localhost:21111/v1
    --model.api_key EMPTY
    --model.concurrency 32
    --model.tensor_parallel_size "$tp"
    --model.gpu_memory_utilization "$gpu_util"
    --model.temperature 0.0
    --model.max_tokens 512
    --model.sprompt ""
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
    --model.type ArgusBase
    --model.model_name "$api_model"
    --model.api_base "$api_base"
    --model.api_key "$api_key_mode"
    --model.concurrency "$conc"
    --model.temperature 0.0
    --model.max_tokens 512
    --model.sprompt ""
  )

  echo "==================== CLOSED MODEL: $tag ===================="
  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    submit_gen "$tag" 1 "${COMMON_OVERRIDES[@]}"
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    eval_num_gpus=$(get_eval_num_gpus)
    submit_eval "$tag" "$eval_num_gpus" "${COMMON_OVERRIDES[@]}"
  fi
done

echo "All submissions done."

