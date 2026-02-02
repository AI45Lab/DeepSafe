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
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH

export http_proxy="HTTP_PROXY"
export https_proxy="HTTP_PROXY"
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
cd "$MBEF_ROOT"

PY_BIN="$(which python3)"
echo "Parsing config: $CONFIG_PATH"

PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config \
    --config "$CONFIG_PATH" \
    --mbef-root "$MBEF_ROOT" \
    --format bash \
    --strict "${OVERRIDES[@]}")" \
  || { echo "ERROR: failed to parse config"; exit 1; }
eval "$PARSED_ENV"

EXP_NAME="${EXP_NAME:-beavertails_task}"
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
echo "Stage:  GEN"
echo "Task:   $EXP_NAME"
echo "Target: $TARGET_MODEL (Port ${TARGET_PORT:-21111})"
echo "Output: $PREDICTIONS_PATH"
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
    fi
}
trap cleanup EXIT

if [[ -n "$TARGET_PORT" ]]; then
  pkill -f "vllm.entrypoints.openai.api_server.*--port $TARGET_PORT" || true
else
  pkill -f "vllm.entrypoints.openai.api_server.*--port 21111" || true
fi
sleep 3

wait_for_service() {
    local port=$1
    local name=$2
    local retries=1800
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

PORT="${TARGET_PORT:-21111}"
if [[ "$TARGET_API_BASE" == http://localhost:* || "$TARGET_API_BASE" == http://127.0.0.1:* ]]; then
    echo "Starting Target Model (local)..."
    TARGET_TP="${TARGET_TENSOR_PARALLEL_SIZE:-1}"
    TARGET_GPU_UTIL="${TARGET_GPU_MEM_UTIL:-0.85}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "Target TP=${TARGET_TP} (vLLM will require >= TP GPUs visible)"
    nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
        --model "$TARGET_MODEL" \
        --served-model-name "$TARGET_MODEL" \
        --trust-remote-code \
        --port "$PORT" \
        --tensor-parallel-size "$TARGET_TP" \
        --gpu-memory-utilization "$TARGET_GPU_UTIL" \
        --max-model-len 2048 \
        > "$LOG_DIR/vllm_target_$TIMESTAMP.log" 2>&1 &

    wait_for_service "$PORT" "Target Model" || {
        tail -n 50 "$LOG_DIR/vllm_target_$TIMESTAMP.log"
        exit 1
    }
else
    echo "Using remote Target API: $TARGET_API_BASE"
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
    echo "Success! Predictions saved to: $PREDICTIONS_PATH"
else
    echo "Failure (exit $EXIT_CODE). Check log: $RUN_LOG"
    tail -n 50 "$RUN_LOG"
fi

exit $EXIT_CODE