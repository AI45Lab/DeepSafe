#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -eo pipefail

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

BASE_CONFIG="${1:-configs/eval_tasks/truthful_qa_demo_mc.yaml}"
STAGE="${2:-all}"
shift 2 || true
EXTRA_OVERRIDES=("$@")

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
if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  echo "Extra overrides: ${EXTRA_OVERRIDES[*]}"
fi

OPEN_MODELS=(
  "Qwen2.5-72B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen2.5-72B-Instruct|2|0.85"
  "Llama-3.3-70B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/meta-llama/Llama-3.3-70B-Instruct|2|0.85"
  
  "InternLM3-8B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/internlm/internlm3-8b-instruct|1|0.9"
  "Gemma-3-27B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/google/gemma-3-27b-it|1|0.85"
  "Mistral-Small-24B-Instruct|/mnt/shared-storage-user/ai4good2-share/models/mistralai/Mistral-Small-24B-Instruct-2501|1|0.85"

  "GLM-4.5-Air|/mnt/shared-storage-user/ai4good2-share/models/zai-org/GLM-4.5-Air|4|0.9"
  "Qwen3-30B-A3B-Thinking-2507|/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen3-30B-A3B-Instruct-2507|4|0.9"
)

COMMON_OVERRIDES=(
  --model.type APIModel
  --model.max_tokens 0
  --model.temperature 0.0
)

submit_stage () {
  local tag="$1"
  local stage="$2"
  local num_gpus="$3"
  shift 3 || true
  local overrides=("$@")

  local out_dir="results/truthful_qa_batch/${tag}"
  echo "==================== ${tag} ===================="
  echo "Submitting TruthfulQA task: $BASE_CONFIG (stage=$stage)"
  echo "Output: $out_dir"

  bash "$MBEF_ROOT/scripts/submit_truthful_qa.sh" \
    "$BASE_CONFIG" \
    "$stage" \
    "$num_gpus" \
    --runner.output_dir "$out_dir" \
    "${COMMON_OVERRIDES[@]}" \
    "${overrides[@]}" \
    "${EXTRA_OVERRIDES[@]}"
}

for spec in "${OPEN_MODELS[@]}"; do
  IFS="|" read -r TAG MODEL_NAME TP GPU_UTIL CONCURRENCY <<<"$spec"
  TP="${TP:-1}"
  GPU_UTIL="${GPU_UTIL:-0.8}"
  CONCURRENCY="${CONCURRENCY:-16}"

  API_BASE="http://127.0.0.1:21111/v1"

  MODEL_OVERRIDES=(
    --model.model_name "$MODEL_NAME"
    --model.api_base "$API_BASE"
    --model.api_key EMPTY
    --model.mode completion
    --model.concurrency "$CONCURRENCY"
    --model.tensor_parallel_size "$TP"
    --model.gpu_memory_utilization "$GPU_UTIL"
    --model.trust_remote_code true
  )

  case "$STAGE" in
    gen)
      submit_stage "$TAG" "gen" "$TP" "${MODEL_OVERRIDES[@]}"
      ;;
    eval)
      submit_stage "$TAG" "eval" "$TP" "${MODEL_OVERRIDES[@]}"
      ;;
    all)
      submit_stage "$TAG" "gen" "$TP" "${MODEL_OVERRIDES[@]}"
      submit_stage "$TAG" "eval" "$TP" "${MODEL_OVERRIDES[@]}"
      ;;
  esac
done

echo "All submissions done."

