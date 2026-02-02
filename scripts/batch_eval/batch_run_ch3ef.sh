#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -eo pipefail

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

DEFAULT_CONFIG="$MBEF_ROOT/configs/eval_tasks/ch3ef.yaml"

if [[ "$1" == "gen" || "$1" == "eval" || "$1" == "all" ]]; then
  BASE_CONFIG="$DEFAULT_CONFIG"
  STAGE="$1"
else
  BASE_CONFIG="${1:-$DEFAULT_CONFIG}"
  STAGE="${2:-all}"
fi

if [[ "$BASE_CONFIG" != /* ]]; then
  BASE_CONFIG="$MBEF_ROOT/$BASE_CONFIG"
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "ERROR: base config not found: $BASE_CONFIG" >&2
  echo "Hint: pass an absolute yaml path, e.g.:" >&2
  echo "  bash $MBEF_ROOT/scripts/batch_run_ch3ef.sh $MBEF_ROOT/configs/eval_tasks/xxx.yaml all" >&2
  echo "  bash $MBEF_ROOT/scripts/batch_run_ch3ef.sh all
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

    "Kimi-VL-A3B-Thinking|/mnt/shared-storage-user/ai4good2-share/models/moonshotai/Kimi-VL-A3B-Thinking-2506|4|0.9"
    
)

CLOSED_MODELS=(
)

get_eval_num_gpus () {
  echo "1"
}

get_eval_submit_profile () {
  echo "cpu_task"
}

get_predictions_path() {
  local config="$1"
  local tag="$2"
  shift 2
  local overrides=("$@")
  
  local py_bin=$(which python3)
  local parsed_env="$("$py_bin" -m uni_eval.cli.parse_eval_config --config "$config" --mbef-root "$MBEF_ROOT" --format bash --strict --runner.output_dir "results/ch3ef_batch/${tag}" "${overrides[@]}" 2>/dev/null)"
  if [[ $? -eq 0 && -n "$parsed_env" ]]; then
    eval "$parsed_env"
    if [[ -n "$OUTPUT_DIR_REL" ]]; then
      echo "$MBEF_ROOT/$OUTPUT_DIR_REL/predictions.jsonl"
    elif [[ -n "$EXP_NAME" ]]; then
      echo "$MBEF_ROOT/results/$EXP_NAME/predictions.jsonl"
    else
      echo "$MBEF_ROOT/results/ch3ef_batch/${tag}/predictions.jsonl"
    fi
  else
    echo "$MBEF_ROOT/results/ch3ef_batch/${tag}/predictions.jsonl"
  fi
}

wait_for_predictions() {
  local predictions_path="$1"
  local tag="$2"
  
  if [[ -f "$predictions_path" ]]; then
    echo "  ✓ Predictions file already exists: $predictions_path"
    return 0
  fi
  
  echo "  Waiting for predictions file: $predictions_path"
  MAX_WAIT=7200
  WAIT_INTERVAL=30
  ELAPSED=0
  
  while [[ ! -f "$predictions_path" && $ELAPSED -lt $MAX_WAIT ]]; do
    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
    if [[ $((ELAPSED % 300)) -eq 0 ]]; then
      echo "    Still waiting for $tag... (${ELAPSED}s / ${MAX_WAIT}s elapsed)"
    fi
  done
  
  if [[ ! -f "$predictions_path" ]]; then
    echo "  ✗ ERROR: Predictions file not found after waiting ${MAX_WAIT}s"
    echo "    Expected: $predictions_path"
    return 1
  fi
  
  if [[ ! -s "$predictions_path" ]]; then
    echo "  ✗ ERROR: Predictions file exists but is empty"
    return 1
  fi
  
  PRED_LINES=$(wc -l < "$predictions_path" 2>/dev/null || echo "0")
  echo "  ✓ Predictions file ready: $predictions_path ($PRED_LINES lines)"
  return 0
}

submit_gen () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/submit_ch3ef.sh" "$BASE_CONFIG" "gen" "$num_gpus" \
    --runner.output_dir "results/ch3ef_batch/${tag}" \
    "${overrides[@]}"
}

submit_eval () {
  local tag="$1"
  local num_gpus="$2"

  bash "$MBEF_ROOT/scripts/submit_ch3ef.sh" "$BASE_CONFIG" "eval" 0 \
    --runner.output_dir "results/ch3ef_batch/${tag}"
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
  if [[ "$STAGE" == "gen" ]]; then
    submit_gen "$tag" "$num_gpus" "${COMMON_OVERRIDES[@]}"
    sleep 10
  elif [[ "$STAGE" == "eval" ]]; then
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
  if [[ "$STAGE" == "gen" ]]; then
    SUBMIT_PROFILE="cpu_task" submit_gen "$tag" 1 "${COMMON_OVERRIDES[@]}"
    sleep 10
  elif [[ "$STAGE" == "eval" ]]; then
    eval_num_gpus=$(get_eval_num_gpus)
    eval_profile=$(get_eval_submit_profile)
    SUBMIT_PROFILE="$eval_profile" submit_eval "$tag" "$eval_num_gpus"
    sleep 10
  fi
done

echo "All submissions done."