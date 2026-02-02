#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -e

CONFIG_PATH="$1"
shift || true
OVERRIDES=("$@")

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path> [overrides...]"
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH"
  exit 1
fi



export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0,::1}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

DEFAULT_PROXY_URL="${VLSBENCH_DEFAULT_PROXY_URL:-"HTTP_PROXY"}"
export http_proxy="${http_proxy:-$DEFAULT_PROXY_URL}"
export https_proxy="${https_proxy:-$DEFAULT_PROXY_URL}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"

echo "--------------------------------------------------------"
echo "Transformer version:"
pip list | grep transformer
echo "GPU:" $CUDA_VISIBLE_DEVICES
echo "nvidia-smi:" $(nvidia-smi)
echo "flash-attention:" $(pip list | grep flash-attention)
echo "--------------------------------------------------------"

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH
cd "$MBEF_ROOT"

PY_BIN=$(which python3)
echo "Parsing config: $CONFIG_PATH"
PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config --config "$CONFIG_PATH" --mbef-root "$MBEF_ROOT" --format bash "${OVERRIDES[@]}")" \
  || { echo "ERROR: failed to parse config: $CONFIG_PATH"; exit 1; }
eval "$PARSED_ENV"

EXP_NAME="${EXP_NAME:-vlsbench_task}"
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
echo "Stage:  gen (vlsbench)"
echo "Task:   $EXP_NAME"
echo "Target: $TARGET_MODEL"
echo "API:    $TARGET_API_BASE"
echo "Pred:   $PREDICTIONS_PATH"
echo "Logs:   $LOG_DIR"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi
echo "--------------------------------------------------------"

cleanup() {
  echo "Cleaning up..."
  pkill -P $$ || true
  if [[ -n "$TARGET_PORT" ]]; then
    pkill -f "vllm.entrypoints.openai.api_server.*--port $TARGET_PORT" || true
  else
    pkill -f "vllm.entrypoints.openai.api_server" || true
  fi
}
trap cleanup EXIT

wait_for_service() {
  local port=$1
  local name=$2
  local retries=300
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

if [[ "$TARGET_API_BASE" == http://localhost:* || "$TARGET_API_BASE" == http://127.0.0.1:* ]]; then
  TARGET_PORT="${TARGET_PORT:-21113}"
  echo "Starting Target VLM (local vLLM API server) on port $TARGET_PORT..."
  nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$TARGET_MODEL" \
    --served-model-name "$TARGET_MODEL" \
    --trust-remote-code \
    --port "$TARGET_PORT" \
    --tensor-parallel-size "${TARGET_TENSOR_PARALLEL_SIZE:-1}" \
    --gpu-memory-utilization "${TARGET_GPU_MEM_UTIL:-0.8}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
    > "$LOG_DIR/vllm_target_$TIMESTAMP.log" 2>&1 &
  wait_for_service "$TARGET_PORT" "Target Model" || { tail -n 50 "$LOG_DIR/vllm_target_$TIMESTAMP.log" || true; exit 1; }
else
  echo "Target api_base is remote. Skip starting local vLLM."
fi

echo "Generating predictions..."
RUN_LOG="$LOG_DIR/mbef_gen_$TIMESTAMP.log"
"$PY_BIN" tools/run.py "$CONFIG_PATH" \
  --runner.stage gen \
  --runner.predictions_path "$PREDICTIONS_PATH" \
  "${OVERRIDES[@]}" \
  > "$RUN_LOG" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "Success. Predictions at: $PREDICTIONS_PATH"
else
  echo "Failure (exit $EXIT_CODE). Check log: $RUN_LOG"
  tail -n 40 "$RUN_LOG" || true
fi
exit $EXIT_CODE

