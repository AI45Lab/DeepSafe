#!/bin/bash
set -e

CONFIG_PATH="$1"
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path>"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH"
  exit 1
fi




SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
cd "$MBEF_ROOT"

PY_BIN=$(which python3)
echo "Parsing config: $CONFIG_PATH"
eval "$("$PY_BIN" -m uni_eval.cli.parse_eval_config --config "$CONFIG_PATH" --format bash)"

TARGET_PORT="${TARGET_PORT:-21111}"
JUDGE_PORT="${JUDGE_PORT:-}"

EXP_NAME="${EXP_NAME:-uncontrolled_aird_task}"
LOG_DIR="${LOG_DIR:-$MBEF_ROOT/scripts/rlaunch_logs/$EXP_NAME}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

echo "--------------------------------------------------------"
echo "Task:       $EXP_NAME"
echo "Target:     $TARGET_MODEL (port $TARGET_PORT)"
if [[ -n "$JUDGE_MODEL" ]]; then
  echo "Judge:      $JUDGE_MODEL (port ${JUDGE_PORT:-<same as target>})"
else
  echo "Judge:      (none)"
fi
echo "Logs dir:   $LOG_DIR"
echo "--------------------------------------------------------"

cleanup() {
  echo "Cleaning up..."
  pkill -P $$ || true
  pkill -f "vllm.entrypoints.openai.api_server" || true
}
trap cleanup EXIT

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 3

wait_for_service() {
  local port=$1
  local name=$2
  local retries=300
  echo "Waiting for $name on port $port..."
  for i in $(seq 1 $retries); do
    status_code=$(curl --noproxy "*" -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/v1/models" || true)
    if [[ "$status_code" == "200" ]]; then
      echo "$name is ready!"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: $name failed to start within timeout."
  return 1
}

START_JUDGE_SERVER=false
if [[ -n "$JUDGE_MODEL" && -n "$JUDGE_PORT" && "$JUDGE_PORT" != "$TARGET_PORT" ]]; then
  START_JUDGE_SERVER=true
fi

if [[ "$START_JUDGE_SERVER" == true ]]; then
  TARGET_MEM=0.45
  JUDGE_MEM=0.45
else
  TARGET_MEM=0.85
fi

echo "Starting Target LLM with mem_util=$TARGET_MEM on port $TARGET_PORT..."
CUDA_VISIBLE_DEVICES=0 nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$TARGET_MODEL" \
  --served-model-name "$TARGET_MODEL" \
  --trust-remote-code \
  --port "$TARGET_PORT" \
  --gpu-memory-utilization "$TARGET_MEM" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-16384}" \
  > "$LOG_DIR/vllm_target_$TIMESTAMP.log" 2>&1 &

wait_for_service "$TARGET_PORT" "Target Model" || { tail -n 40 "$LOG_DIR/vllm_target_$TIMESTAMP.log" || true; exit 1; }

if [[ "$START_JUDGE_SERVER" == true ]]; then
  echo "Starting Judge LLM with mem_util=$JUDGE_MEM on port $JUDGE_PORT..."
  CUDA_VISIBLE_DEVICES=0 nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" \
    --served-model-name "$JUDGE_MODEL" \
    --trust-remote-code \
    --port "$JUDGE_PORT" \
    --gpu-memory-utilization "$JUDGE_MEM" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-16384}" \
    > "$LOG_DIR/vllm_judge_$TIMESTAMP.log" 2>&1 &
  wait_for_service "$JUDGE_PORT" "Judge Model" || { tail -n 40 "$LOG_DIR/vllm_judge_$TIMESTAMP.log" || true; exit 1; }
else
  echo "No separate Judge vLLM started (same endpoint or no judge)."
fi

echo "Running Uncontrolled_AIRD evaluation via tools/run.py..."
RUN_LOG="$LOG_DIR/mbef_run_$TIMESTAMP.log"
"$PY_BIN" tools/run.py "$CONFIG_PATH" > "$RUN_LOG" 2>&1

EXIT_CODE=$?
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "Success! Results saved."
  echo "Report: $MBEF_ROOT/$OUTPUT_DIR_REL/report.md"
else
  echo "Failure (exit $EXIT_CODE). Check log: $RUN_LOG"
  tail -n 40 "$RUN_LOG" || true
fi
exit $EXIT_CODE

