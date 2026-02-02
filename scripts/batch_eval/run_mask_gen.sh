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

source /mnt/shared-storage-user/yangjingyi/anaconda3/etc/profile.d/conda.sh
conda activate safeai

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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
EXP_NAME="${EXP_NAME:-mask_v01_qwen2.5-7b-instruct}"
LOG_DIR="$MBEF_ROOT/scripts/rlaunch_logs/$EXP_NAME"
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
    local retries=60
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
    CUDA_VISIBLE_DEVICES=0 nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
        --model "$TARGET_MODEL" \
        --served-model-name "$TARGET_MODEL" \
        --trust-remote-code \
        --port "$PORT" \
        --gpu-memory-utilization 0.85 \
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
  --runner.use_evaluator_gen true \
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