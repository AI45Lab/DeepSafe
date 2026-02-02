#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -eo pipefail




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
  "InternLM3-8B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/internlm/internlm3-8b-instruct|4|0.9"
  "Gemma-3-27B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/google/gemma-3-27b-it|4|0.85"
  "Mistral-Small-24B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/mistralai/Mistral-Small-24B-Instruct-2501|2|0.9"
  "GLM-4.5-Air|/mnt/shared-storage-user/ai4good2-share/models/zai-org/GLM-4.5-Air|4|0.9"

)

submit_gen () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  echo "Submitting GEN job: $tag ($num_gpus GPUs)"
  bash "$MBEF_ROOT/scripts/run_behonest_gen.sh" "$BASE_CONFIG" \
    --runner.output_dir "results/behonest_batch/${tag}" \
    "${overrides[@]}"
}

submit_eval () {
  local tag="$1"
  local num_gpus="$2"
  shift 2
  local overrides=("$@")

  echo "Submitting EVAL job: $tag ($num_gpus GPUs)"
  bash "$MBEF_ROOT/scripts/run_behonest_eval.sh" "$BASE_CONFIG" \
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
      eval_num_gpus=1
      if [[ ${#JUDGE_OVERRIDES[@]} -gt 0 ]]; then
        for override in "${JUDGE_OVERRIDES[@]}"; do
          if [[ "$override" == "--evaluator.judge_model_cfg.type" ]]; then
            next="${JUDGE_OVERRIDES[$((idx+1))]}"
            if [[ "$next" == "VLLMLocalModel" ]]; then
              eval_num_gpus=1
            fi
          fi
        done
      fi
      submit_eval "$tag-$category" "$eval_num_gpus" "${COMMON_OVERRIDES[@]}" "${CATEGORY_OVERRIDES[@]}"
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
      submit_gen "$tag-$category" 1 "${COMMON_OVERRIDES[@]}" "${CATEGORY_OVERRIDES[@]}"
      sleep 5
    fi

    if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
      eval_num_gpus=1
      submit_eval "$tag-$category" "$eval_num_gpus" "${COMMON_OVERRIDES[@]}" "${CATEGORY_OVERRIDES[@]}"
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
