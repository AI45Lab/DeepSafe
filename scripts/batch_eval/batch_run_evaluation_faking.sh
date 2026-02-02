#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

BASE_CONFIG="${1:-configs/eval_tasks/evaluation_faking_strongreject.yaml}"
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
  echo "ERROR: invalid stage '$STAGE' (expected gen|eval|all)" >&2
  exit 1
fi

echo "Base config: $BASE_CONFIG"
echo "Stage: $STAGE"
echo "MBEF: $MBEF_ROOT"

JUDGE_PORT="${JUDGE_PORT:-21118}"
BATCH_VLLM_PID=""

JUDGE_MODEL="/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen2.5-72B-Instruct"
JUDGE_API_BASE="http://localhost:21118/v1"
JUDGE_TP="${JUDGE_TP:-4}"
JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.85}"

batch_cleanup() {
  echo "Batch cleanup..."
  if [[ -n "$BATCH_VLLM_PID" ]]; then
    echo "Stopping shared vLLM (PID=$BATCH_VLLM_PID)..."
    kill "$BATCH_VLLM_PID" 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server.*--port $JUDGE_PORT" 2>/dev/null || true
  fi
}
trap batch_cleanup EXIT

wait_for_vllm() {
  local port=$1
  local retries=600
  echo "Waiting for vLLM on port $port..."
  for _ in $(seq 1 $retries); do
    code="$(curl --noproxy "*" -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/v1/models" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      echo "vLLM is ready!"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: vLLM failed to start within timeout."
  return 1
}

start_shared_judge_vllm() {
  existing_code="$(curl --noproxy "*" -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$JUDGE_PORT/v1/models" 2>/dev/null || true)"
  if [[ "$existing_code" == "200" ]]; then
    echo "Judge vLLM already running on port $JUDGE_PORT (reusing)"
    return 0
  fi

  echo "=========================================="
  echo "Starting shared Judge vLLM on port $JUDGE_PORT"
  echo "Model: $JUDGE_MODEL"
  echo "TP: $JUDGE_TP, GPU Util: $JUDGE_GPU_UTIL"
  echo "=========================================="

  local log_dir="$MBEF_ROOT/scripts/rlaunch_logs"
  mkdir -p "$log_dir"
  local vllm_log="$log_dir/vllm_judge_batch_$(date +%Y%m%d%H%M%S).log"
  echo "vLLM log: $vllm_log"

  python3 -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" \
    --served-model-name "$JUDGE_MODEL" \
    --trust-remote-code \
    --port "$JUDGE_PORT" \
    --tensor-parallel-size "$JUDGE_TP" \
    --gpu-memory-utilization "$JUDGE_GPU_UTIL" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" \
    > "$vllm_log" 2>&1 &
  BATCH_VLLM_PID=$!

  tail -f "$vllm_log" &
  local tail_pid=$!

  if ! wait_for_vllm "$JUDGE_PORT"; then
    echo "--- vLLM 启动失败，最后 100 行日志 ---"
    tail -n 100 "$vllm_log" || true
    kill $tail_pid 2>/dev/null || true
    exit 1
  fi

  kill $tail_pid 2>/dev/null || true
  echo "Shared Judge vLLM ready (PID=$BATCH_VLLM_PID)"
}

DATASET_OVERRIDES=(
  --dataset.path "$MBEF_ROOT/data/hf_datasets/StrongREJECT"
  --dataset.split train
  --dataset.limit 0
)

JUDGE_OVERRIDES=(
  --evaluator.judge_model_cfg.type APIModel
  --evaluator.judge_model_cfg.model_name /mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen2.5-72B-Instruct
  --evaluator.judge_model_cfg.api_base http://localhost:21118/v1
  --evaluator.judge_model_cfg.api_key EMPTY
  --evaluator.judge_model_cfg.concurrency 32
  --evaluator.judge_model_cfg.mode chat
  --evaluator.judge_model_cfg.temperature 0.0
  --evaluator.judge_model_cfg.max_tokens 2048
  --evaluator.judge_model_cfg.tensor_parallel_size 4
  --evaluator.judge_model_cfg.gpu_memory_utilization 0.85
)

OPEN_MODELS=(

  "GLM-4.5-Air|/mnt/shared-storage-user/ai4good2-share/models/zai-org/GLM-4.5-Air|4|0.9"
  "Qwen3-30B-A3B-Thinking-2507|/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen3-30B-A3B-Instruct-2507|4|0.9"
)

CLOSED_MODELS=(
)

run_gen () {
  local tag="$1"
  shift 1
  local overrides=("$@")

  bash "$MBEF_ROOT/scripts/run_evaluation_faking_gen.sh" "$BASE_CONFIG" \
    --runner.output_dir "results/eval_faking_batch/${tag}" \
    "${DATASET_OVERRIDES[@]}" \
    "${overrides[@]}"
}

run_eval () {
  local tag="$1"
  shift 1
  local overrides=("$@")

  local pred_path="$MBEF_ROOT/results/eval_faking_batch/${tag}/predictions.jsonl"

  bash "$MBEF_ROOT/scripts/run_evaluation_faking_eval.sh" "$BASE_CONFIG" \
    --runner.output_dir "results/eval_faking_batch/${tag}" \
    --dataset.type JsonlDataset \
    --dataset.path "$pred_path" \
    --dataset.question_key prompt \
    "${overrides[@]}" \
    "${JUDGE_OVERRIDES[@]}"
}

if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
  if [[ "$JUDGE_API_BASE" == http://localhost:* || "$JUDGE_API_BASE" == http://127.0.0.1:* ]]; then
    start_shared_judge_vllm
    export SKIP_VLLM_STARTUP=true
    export SKIP_VLLM_CLEANUP=true
  fi
fi

for spec in "${OPEN_MODELS[@]}"; do
  IFS="|" read -r tag model_path tp gpu_util <<< "$spec"

  if [[ ! -e "$model_path" ]]; then
    echo "ERROR: model_path not found for tag='$tag': $model_path" >&2
    exit 1
  fi

  COMMON_OVERRIDES=(
    --model.type APIModel
    --model.model_name "$model_path"
    --model.api_base http://localhost:21117/v1
    --model.api_key EMPTY
    --model.mode chat
    --model.concurrency 16
    --model.tensor_parallel_size "${tp:-1}"
    --model.gpu_memory_utilization "${gpu_util:-0.85}"
    --model.temperature 0.0
    --model.max_tokens 2048
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
  if [[ -z "${conc:-}" ]]; then
    conc="8"
  fi

  COMMON_OVERRIDES=(
    --model.type APIModel
    --model.model_name "$api_model"
    --model.api_base "$api_base"
    --model.api_key "$api_key_mode"
    --model.concurrency "$conc"
    --model.mode chat
    --model.temperature 0.0
    --model.max_tokens 2048
  )

  echo "==================== CLOSED MODEL: $tag ===================="
  if [[ "$STAGE" == "gen" || "$STAGE" == "all" ]]; then
    run_gen "$tag" "${COMMON_OVERRIDES[@]}"
  fi
  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    run_eval "$tag"
  fi
done

echo "All local batch runs done."

