#!/bin/bash
set -e

CONFIG_PATH="$1"
shift || true
OVERRIDES=("$@")
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path> [overrides...]" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH
cd "$MBEF_ROOT"

PY_BIN=$(which python3)
echo "Parsing config: $CONFIG_PATH"
PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config --config "$CONFIG_PATH" --mbef-root "$MBEF_ROOT" --format bash --strict "${OVERRIDES[@]}")" \
  || { echo "ERROR: failed to parse config: $CONFIG_PATH" >&2; exit 1; }
eval "$PARSED_ENV"

EXP_NAME="${EXP_NAME:-evaluation_faking_task}"
LOG_DIR="${LOG_DIR:-$MBEF_ROOT/scripts/rlaunch_logs/$EXP_NAME}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

if [[ -n "$OUTPUT_DIR_REL" ]]; then
  PREDICTIONS_PATH_DEFAULT="$MBEF_ROOT/$OUTPUT_DIR_REL/predictions.jsonl"
else
  PREDICTIONS_PATH_DEFAULT="$MBEF_ROOT/results/$EXP_NAME/predictions.jsonl"
fi
PREDICTIONS_PATH="${PREDICTIONS_PATH:-$PREDICTIONS_PATH_DEFAULT}"

echo "--------------------------------------------------------"
echo "Stage:  eval (evaluation_faking)"
echo "Task:   $EXP_NAME"
echo "Pred:   $PREDICTIONS_PATH"
if [[ -n "$JUDGE_MODEL" ]]; then
  echo "Judge:  $JUDGE_MODEL (api_base=$JUDGE_API_BASE port=${JUDGE_PORT:-})"
fi
echo "Logs:   $LOG_DIR"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi
echo "--------------------------------------------------------"

cleanup() {
  echo "Cleaning up..."
  pkill -P $$ || true
  if [[ -n "$JUDGE_PORT" && "${SKIP_VLLM_CLEANUP:-}" != "true" ]]; then
    echo "Stopping vLLM on port $JUDGE_PORT..."
    pkill -f "vllm.entrypoints.openai.api_server.*--port $JUDGE_PORT" || true
  fi
}
trap cleanup EXIT

wait_for_service() {
  local port=$1
  local name=$2
  local retries=600
  echo "Waiting for $name on port $port..."
  for _ in $(seq 1 $retries); do
    code="$(curl --noproxy "*" -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/v1/models" || true)"
    if [[ "$code" == "200" ]]; then
      echo "$name is ready!"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: $name failed to start within timeout."
  return 1
}

if [[ "$JUDGE_API_BASE" == http://localhost:* || "$JUDGE_API_BASE" == http://127.0.0.1:* ]]; then
  JUDGE_PORT="${JUDGE_PORT:-21118}"
  
  if [[ "${SKIP_VLLM_STARTUP:-}" == "true" ]]; then
    echo "Using shared Judge vLLM on port $JUDGE_PORT (managed by batch script)"
  elif existing_code="$(curl --noproxy "*" -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$JUDGE_PORT/v1/models" 2>/dev/null || true)" && [[ "$existing_code" == "200" ]]; then
    echo "Judge Model already running on port $JUDGE_PORT (reusing existing service)"
    SKIP_VLLM_CLEANUP=true
  else
  echo "Starting Judge Model (local vLLM API server) on port $JUDGE_PORT..."
    VLLM_LOG="$LOG_DIR/vllm_judge_$TIMESTAMP.log"
    echo "vLLM log: $VLLM_LOG"
    
    "$PY_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" \
    --served-model-name "$JUDGE_MODEL" \
    --trust-remote-code \
    --port "$JUDGE_PORT" \
    --tensor-parallel-size "${JUDGE_TENSOR_PARALLEL_SIZE:-1}" \
    --gpu-memory-utilization "${JUDGE_GPU_MEM_UTIL:-0.8}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" \
      > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    
    tail -f "$VLLM_LOG" &
    TAIL_PID=$!
    
    if ! wait_for_service "$JUDGE_PORT" "Judge Model"; then
      echo "--- vLLM 启动失败，最后 50 行日志 ---"
      tail -n 50 "$VLLM_LOG" || true
      kill $TAIL_PID 2>/dev/null || true
      exit 1
    fi
    
    kill $TAIL_PID 2>/dev/null || true
    echo "vLLM Judge Model ready (PID=$VLLM_PID)"
  fi
else
  if [[ -n "$JUDGE_API_BASE" ]]; then
    echo "Judge api_base is remote ($JUDGE_API_BASE). Skip starting local vLLM server."
  else
    echo "Judge api_base is empty (likely local in-process judge). Skip starting API server."
  fi
fi

RUN_LOG="$LOG_DIR/mbef_eval_$TIMESTAMP.log"

"$PY_BIN" -u tools/run.py "$CONFIG_PATH" \
  --runner.stage eval \
  --runner.predictions_path "$PREDICTIONS_PATH" \
  --runner.require_predictions true \
  --runner.prediction_field response \
  --evaluator.use_precomputed_predictions true \
  --evaluator.require_precomputed_predictions true \
  --model.type NoOpModel \
  "${OVERRIDES[@]}" \
  2>&1 | tee "$RUN_LOG"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
  echo "Success. Results at: $MBEF_ROOT/$OUTPUT_DIR_REL"
  echo "Report: $MBEF_ROOT/$OUTPUT_DIR_REL/report.md"
else
  echo "Failure (exit $EXIT_CODE). Check log: $RUN_LOG"
  tail -n 40 "$RUN_LOG" || true
fi
exit $EXIT_CODE

