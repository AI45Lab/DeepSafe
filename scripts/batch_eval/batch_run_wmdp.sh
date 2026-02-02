#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -eo pipefail

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

BASE_CONFIG="${1:-configs/eval_tasks/wmdp_v01_mixtral_8x7b_instruct_v01.yaml}"

if [[ "$BASE_CONFIG" != /* ]]; then
  BASE_CONFIG="$MBEF_ROOT/$BASE_CONFIG"
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "ERROR: base config not found: $BASE_CONFIG" >&2
  exit 1
fi
BASE_CONFIG="$(readlink -f "$BASE_CONFIG")"

echo "Base config: $BASE_CONFIG"
echo "MBEF: $MBEF_ROOT"

OPEN_MODELS=(

  "GLM-4.5-Air|/mnt/shared-storage-user/ai4good2-share/models/zai-org/GLM-4.5-Air|4|0.9"
  
)

CLOSED_MODELS=(

)

submit_open () {
  local tag="$1"
  local model_path="$2"
  local tp="$3"
  local gpu_util="$4"

  if [[ ! -e "$model_path" ]]; then
    echo "ERROR: model_path not found for tag='$tag': $model_path" >&2
    exit 1
  fi

  local overrides=(
    --model.type APIModel
    --model.model_name "$model_path"
    --model.api_base http://localhost:21111/v1
    --model.api_key EMPTY
    --model.mode chat
    --model.concurrency 32
    --model.tensor_parallel_size "$tp"
    --model.gpu_memory_utilization "$gpu_util"
    --model.temperature 0.0
    --model.max_tokens 4096
    --runner.output_dir "results/wmdp_batch/${tag}"
  )

  echo "--------------------------------------------------------"
  echo "[OPEN] tag=$tag"
  echo "[OPEN] model_path=$model_path"
  echo "[OPEN] tp=$tp gpu_util=$gpu_util"
  echo "--------------------------------------------------------"

  SUBMIT_PROFILE="gpu" bash "$MBEF_ROOT/scripts/submit_wmdp.sh" "$BASE_CONFIG" "$tp" "${overrides[@]}"
}

submit_closed () {
  local tag="$1"
  local model_name="$2"
  local api_base="$3"
  local api_key="$4"
  local concurrency="$5"

  local overrides=(
    --model.type APIModel
    --model.model_name "$model_name"
    --model.api_base "$api_base"
    --model.api_key "$api_key"
    --model.mode chat
    --model.concurrency "$concurrency"
    --model.temperature 0.0
    --model.max_tokens 512
    --runner.output_dir "results/wmdp_batch/${tag}"
  )

  echo "--------------------------------------------------------"
  echo "[CLOSED] tag=$tag"
  echo "[CLOSED] model_name=$model_name"
  echo "[CLOSED] api_base=$api_base"
  echo "[CLOSED] concurrency=$concurrency"
  echo "--------------------------------------------------------"

  SUBMIT_PROFILE="cpu_task" bash "$MBEF_ROOT/scripts/submit_wmdp.sh" "$BASE_CONFIG" 1 "${overrides[@]}"
}

for spec in "${OPEN_MODELS[@]}"; do
  IFS="|" read -r tag model_path tp gpu_util <<< "$spec"
  submit_open "$tag" "$model_path" "$tp" "$gpu_util"
done

for spec in "${CLOSED_MODELS[@]}"; do
  IFS="|" read -r tag model_name api_base api_key concurrency <<< "$spec"
  submit_closed "$tag" "$model_name" "$api_base" "$api_key" "$concurrency"
done

echo "Done submitting WMDP batch."

