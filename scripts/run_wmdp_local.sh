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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
cd "$MBEF_ROOT"

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"

PY_BIN="$(which python3)"
echo "Parsing config: $CONFIG_PATH"

PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config \
    --config "$CONFIG_PATH" \
    --mbef-root "$MBEF_ROOT" \
    --format bash \
    "${OVERRIDES[@]}")" \
  || { echo "ERROR: failed to parse config"; exit 1; }
eval "$PARSED_ENV"

EXP_NAME="${EXP_NAME:-wmdp_task}"
if [[ -n "$MBEF_LOG_DIR" ]]; then
  LOG_DIR="$MBEF_LOG_DIR"
else
  LOG_DIR="$MBEF_ROOT/scripts/local_logs/$EXP_NAME"
fi
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

echo "--------------------------------------------------------"
echo "Task:   $EXP_NAME (WMDP)"
echo "Target: $TARGET_MODEL (Port ${TARGET_PORT:-21111})"
echo "Logs:   $LOG_DIR"
echo "--------------------------------------------------------"

cleanup() {
    echo "Cleaning up..."
    pkill -P $$ || true
    if [[ -n "$TARGET_PORT" ]]; then
        pkill -f "vllm.entrypoints.openai.api_server.*--port $TARGET_PORT" || true
    fi
}
trap cleanup EXIT

wait_for_service() {
    local port=$1
    local name=$2
    local retries=600
    echo "Waiting for $name on port $port..."
    for i in $(seq 1 $retries); do
        if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/v1/models" | grep -q "200"; then
            echo "$name is ready!"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: $name failed to start."
    return 1
}

if [[ "$TARGET_API_BASE" == http://localhost:* || "$TARGET_API_BASE" == http://127.0.0.1:* ]]; then
    echo "Starting Target Model (local vLLM)..."
    TARGET_TP="${TARGET_TENSOR_PARALLEL_SIZE:-1}"
    TARGET_GPU_UTIL="${TARGET_GPU_MEM_UTIL:-0.85}"
    
    nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
        --model "$TARGET_MODEL" \
        --served-model-name "$TARGET_MODEL" \
        --trust-remote-code \
        --port "${TARGET_PORT:-21111}" \
        --tensor-parallel-size "$TARGET_TP" \
        --gpu-memory-utilization "$TARGET_GPU_UTIL" \
        --max-model-len 4096 \
        > "$LOG_DIR/vllm_target_$TIMESTAMP.log" 2>&1 &
    wait_for_service "${TARGET_PORT:-21111}" "Target Model" || exit 1
fi

echo "Running WMDP Evaluation..."
RUN_LOG="$LOG_DIR/mbef_run_$TIMESTAMP.log"
"$PY_BIN" tools/run.py "$CONFIG_PATH" "${OVERRIDES[@]}" 2>&1 | tee "$RUN_LOG"

EXIT_CODE=${PIPESTATUS[0]}
exit $EXIT_CODE
