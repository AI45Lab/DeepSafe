#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Default to this DeepSafe repo root (two levels above scripts/batch_eval/).
MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
cd "$MBEF_ROOT"

BASE_CONFIG="${1:-configs/eval_tasks/behonest_gpt4o_v01.yaml}"
STAGE="${2:-all}"

if [[ "$BASE_CONFIG" != /* ]]; then
  BASE_CONFIG="$MBEF_ROOT/$BASE_CONFIG"
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "ERROR: base config not found: $BASE_CONFIG" >&2
  echo "Hint: pass an absolute yaml path, e.g.:" >&2
  echo "  bash $0 configs/eval_tasks/behonest_gpt4o_v01.yaml all" >&2
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

JUDGE_OVERRIDES=()

CATEGORIES=(
  "Unknowns"
  "Knowns"

  "Burglar_Deception"
  "Game"

  "Prompt_Format"
  "Open_Form"
  "Multiple_Choice"
)

OPEN_MODELS=(

  "Llama-3-8B-Instruct|/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2|4|0.8"

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
  local i=0
  while [[ $i -lt ${#JUDGE_OVERRIDES[@]} ]]; do
    if [[ "${JUDGE_OVERRIDES[$i]}" == "--evaluator.judge_model_cfg.type" ]]; then
      judge_type="${JUDGE_OVERRIDES[$((i+1))]}"
      break
    fi
    ((i++))
  done
  if [[ "$judge_type" == "VLLMLocalModel" || -z "$judge_type" ]]; then
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

  bash "$MBEF_ROOT/scripts/submit_behonest_gen.sh" "$BASE_CONFIG" "$num_gpus" \
    --runner.output_dir "results/behonest_batch/${tag}" \
    "${overrides[@]}"
}

submit_eval () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/submit_behonest_eval.sh" "$BASE_CONFIG" "$num_gpus" \
    --runner.output_dir "results/behonest_batch/${tag}" \
    "${overrides[@]}" \
    "${JUDGE_OVERRIDES[@]}"
}

for spec in "${OPEN_MODELS[@]}"; do
  IFS="|" read -r tag model_path tp gpu_util <<< "$spec"

  num_gpus="${tp}"
  echo "$model_path"
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
    --runner.use_evaluator_gen true
    --evaluator.enable_resampling true
    --evaluator.resample_n 20
    --evaluator.resample_temperature 0.7
  )

  echo "==================== OPEN MODEL: $tag ===================="
  for category in "${CATEGORIES[@]}"; do
    echo "Category: $category"

    CATEGORY_OVERRIDES=(
      --dataset.category "$category"
      --runner.output_dir "results/behonest_batch/${tag}/${category}"
    )

    if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
      submit_gen "$tag-$category" "$num_gpus" "${COMMON_OVERRIDES[@]}" "${CATEGORY_OVERRIDES[@]}"
      sleep 5
    fi

    if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
      eval_num_gpus=$(get_eval_num_gpus)
      eval_profile=$(get_eval_submit_profile)
      SUBMIT_PROFILE="$eval_profile" submit_eval "$tag-$category" "$eval_num_gpus" "${COMMON_OVERRIDES[@]}" "${CATEGORY_OVERRIDES[@]}"
      sleep 5
    fi
  done
done

for spec in "${CLOSED_MODELS[@]}"; do
  IFS="|" read -r tag api_model api_base api_key_mode conc <<< "$spec"

  if [[ -z "${conc:-}" ]]; then
    conc="10"
  fi

  COMMON_OVERRIDES=(
    --model.type APIModel
    --model.model_name "$api_model"
    --model.api_base "$api_base"
    --model.api_key "$api_key_mode"
    --model.concurrency "$conc"
    --model.mode chat
    --model.temperature 0.0
    --runner.use_evaluator_gen true
    --evaluator.enable_resampling true
    --evaluator.resample_n 20
    --evaluator.resample_temperature 0.7
  )

  echo "==================== CLOSED MODEL: $tag ===================="
  for category in "${CATEGORIES[@]}"; do
    echo "Category: $category"

    CATEGORY_OVERRIDES=(
      --dataset.category "$category"
      --runner.output_dir "results/behonest_batch/${tag}/${category}"
    )

    if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
      SUBMIT_PROFILE="cpu_task" submit_gen "$tag" 1 "${COMMON_OVERRIDES[@]}" "${CATEGORY_OVERRIDES[@]}"
      sleep 5
    fi

    if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
      eval_num_gpus=$(get_eval_num_gpus)
      eval_profile=$(get_eval_submit_profile)
      SUBMIT_PROFILE="$eval_profile" submit_eval "$tag" "$eval_num_gpus" "${COMMON_OVERRIDES[@]}" "${CATEGORY_OVERRIDES[@]}"
      sleep 5
    fi
  done
done

echo "All submissions done."
if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
  echo ""
  echo "============================================================"
  echo "  Aggregating results across all categories..."
  echo "============================================================"

  python3 "$MBEF_ROOT/tools/aggregate_behonest_results.py" \
      "$MBEF_ROOT/results/behonest_batch" || {
      echo "WARNING: Aggregation failed. You can run it manually later:"
      echo "  python3 tools/aggregate_behonest_results.py results/behonest_batch"
  }

  echo ""
  echo "============================================================"
  echo "  Batch evaluation complete!"
  echo "============================================================"
  echo "Results saved to: $MBEF_ROOT/results/behonest_batch/"
  echo "Aggregated results: $MBEF_ROOT/results/behonest_batch/aggregated/"
  echo "============================================================"
fi
