#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -eo pipefail

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

BASE_CONFIG="${1:-configs/eval_tasks/salad_judge_v01_qwen1.5-0.5b_vllm_local.yaml}"
STAGE="${2:-all}"

if [[ "$BASE_CONFIG" != /* ]]; then
  BASE_CONFIG="$MBEF_ROOT/$BASE_CONFIG"
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "ERROR: base config not found: $BASE_CONFIG" >&2
  echo "Hint: pass an absolute yaml path, e.g.:" >&2
  echo "  bash $MBEF_ROOT/scripts/batch_run_salad_batch.sh $MBEF_ROOT/configs/eval_tasks/xxx.yaml all" >&2
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

JUDGE_OVERRIDES=(
  --evaluator.judge_model_cfg.type APIModel
  --evaluator.judge_model_cfg.model_name /mnt/shared-storage-user/ai4good2-share/models/ProGuard-7B
  --evaluator.judge_model_cfg.api_base http://localhost:21112/v1
  --evaluator.judge_model_cfg.api_key EMPTY
  --evaluator.judge_model_cfg.mode chat
  --evaluator.judge_model_cfg.temperature 0.0
  --evaluator.judge_model_cfg.max_tokens 1024
  --evaluator.judge_model_cfg.tensor_parallel_size 1
  --evaluator.judge_model_cfg.gpu_memory_utilization 0.85
)

OPEN_MODELS=(
  "Qwen2.5-72B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen2.5-72B-Instruct|2|0.85"
  "Llama-3.3-70B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/meta-llama/Llama-3.3-70B-Instruct|2|0.85"
  
  "Gemma-3-27B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/google/gemma-3-27b-it|1|0.85"

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

get_eval_submit_profile () {
  local judge_type=""
  local judge_api_base=""
  local i=0
  while [[ $i -lt ${#JUDGE_OVERRIDES[@]} ]]; do
    if [[ "${JUDGE_OVERRIDES[$i]}" == "--evaluator.judge_model_cfg.type" ]]; then
      judge_type="${JUDGE_OVERRIDES[$((i+1))]}"
    elif [[ "${JUDGE_OVERRIDES[$i]}" == "--evaluator.judge_model_cfg.api_base" ]]; then
      judge_api_base="${JUDGE_OVERRIDES[$((i+1))]}"
    fi
    ((i++))
  done

  if [[ "$judge_type" == "VLLMLocalModel" || -z "$judge_type" ]]; then
    echo "gpu"
    return 0
  fi

  if [[ "$judge_api_base" == http://localhost:* || "$judge_api_base" == http://127.0.0.1:* ]]; then
    echo "gpu"
  else
    echo "cpu_task"
  fi
}

submit_gen () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/submit_salad.sh" "$BASE_CONFIG" gen "$num_gpus" \
    --runner.output_dir "results/salad_batch/${tag}" \
    "${overrides[@]}"
}

submit_eval () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/batch_eval/submit_salad.sh" "$BASE_CONFIG" eval "$num_gpus" \
    --runner.output_dir "results/salad_batch/${tag}" \
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
  IFS="|" read -r tag api_model api_base api_key_mode conc <<< "$spec"
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
    SUBMIT_PROFILE="cpu_task" submit_gen "$tag" 1 "${COMMON_OVERRIDES[@]}"
    sleep 10
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    eval_num_gpus=$(get_eval_num_gpus)
    eval_profile=$(get_eval_submit_profile)
    SUBMIT_PROFILE="$eval_profile" submit_eval "$tag" "$eval_num_gpus" "${COMMON_OVERRIDES[@]}"
    sleep 10
  fi
done

echo "All submissions done."

