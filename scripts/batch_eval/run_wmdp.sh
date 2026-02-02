#!/bin/bash
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




export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
export http_proxy="HTTP_PROXY"
export https_proxy="HTTP_PROXY"
export HTTP_PROXY="HTTP_PROXY"
export HTTPS_PROXY="HTTP_PROXY"

export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
cd "$MBEF_ROOT"

PY_BIN=$(which python3)
echo "Parsing config: $CONFIG_PATH"

PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config \
  --config "$CONFIG_PATH" \
  --mbef-root "$MBEF_ROOT" \
  --format bash \
  --strict \
  "${OVERRIDES[@]}")" || { echo "ERROR: failed to parse config: $CONFIG_PATH"; exit 1; }
eval "$PARSED_ENV"

EXP_NAME="${EXP_NAME:-wmdp_task}"
LOG_DIR="${LOG_DIR:-$MBEF_ROOT/scripts/rlaunch_logs/$EXP_NAME}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

echo $LOG_DIR
echo "--------------------------------------------------------"
echo "Task:   $EXP_NAME (wmdp)"
echo "Target: $TARGET_MODEL"
echo "API:    $TARGET_API_BASE"
if [[ -n "$TARGET_PORT" ]]; then
  echo "Port:   $TARGET_PORT"
fi
echo "Out:    $MBEF_ROOT/$OUTPUT_DIR_REL"
echo "Logs:   $LOG_DIR"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi
echo "--------------------------------------------------------"

VLLM_PID=""
cleanup() {
    echo "Cleaning up..."
    if [[ -n "$VLLM_PID" ]]; then
      kill "$VLLM_PID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$TARGET_PORT" ]]; then
      pkill -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

wait_for_service() {
    local port=$1
    local name=$2
    local retries=240
    echo "Waiting for $name on port $port..."
    for i in $(seq 1 $retries); do
        if curl --noproxy "*" -s -o /dev/null -w "%{http_code}" \
            "http://127.0.0.1:$port/v1/models" | grep -q "200"; then
            echo "$name is ready!"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: $name failed to start."
    return 1
}

if [[ -n "$TARGET_PORT" ]]; then
  pkill -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || true
  sleep 2

  TARGET_TP="${TARGET_TENSOR_PARALLEL_SIZE:-1}"
  TARGET_GPU_UTIL="${TARGET_GPU_MEM_UTIL:-0.85}"
  VLLM_LOG="$LOG_DIR/vllm_target_$TIMESTAMP.log"
  echo "Starting WMDP Target Model via vLLM..."
  nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
      --model "$TARGET_MODEL" \
      --served-model-name "$TARGET_MODEL" \
      --trust-remote-code \
      --port "$TARGET_PORT" \
      --tensor-parallel-size "$TARGET_TP" \
      --gpu-memory-utilization "$TARGET_GPU_UTIL" \
      --max-model-len 4096 \
      > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!

  wait_for_service "$TARGET_PORT" "Target Model" || {
      tail -n 50 "$VLLM_LOG"
      exit 1
  }
else
  echo "Target api_base is remote; skipping local vLLM startup."
fi

echo "Running WMDP Evaluation..."
RUN_LOG="$LOG_DIR/mbef_run_$TIMESTAMP.log"

"$PY_BIN" tools/run.py "$CONFIG_PATH" "${OVERRIDES[@]}" > "$RUN_LOG" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Success! Results saved."
    echo "Report: $MBEF_ROOT/$OUTPUT_DIR_REL/report.md"
else
    echo "Failure (exit $EXIT_CODE). Check log: $RUN_LOG"
    tail -n 50 "$RUN_LOG"
fi

exit $EXIT_CODE
