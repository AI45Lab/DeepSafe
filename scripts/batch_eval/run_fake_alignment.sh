#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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




MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
cd "$MBEF_ROOT"

PY_BIN=$(which python3)
echo "Parsing config: $CONFIG_PATH"

PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config --config "$CONFIG_PATH" --mbef-root "$MBEF_ROOT" --format bash --strict "${OVERRIDES[@]}")" \
  || { echo "ERROR: failed to parse config: $CONFIG_PATH" >&2; exit 1; }
eval "$PARSED_ENV"

EXP_NAME="${EXP_NAME:-fake_alignment_task}"
LOG_DIR="${LOG_DIR:-$MBEF_ROOT/scripts/rlaunch_logs/$EXP_NAME}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

echo "--------------------------------------------------------"
echo "Task:   $EXP_NAME"
echo "Target: $TARGET_MODEL (api_base=$TARGET_API_BASE port=${TARGET_PORT:-21111})"
echo "Judge:  $JUDGE_MODEL (api_base=$JUDGE_API_BASE port=${JUDGE_PORT:-21112})"
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
    if [[ -n "$JUDGE_PORT" ]]; then
      pkill -f "vllm.entrypoints.openai.api_server.*--port $JUDGE_PORT" || true
    fi
    pkill -f "vllm.entrypoints.openai.api_server" || true
}
trap cleanup EXIT

if [[ -n "$TARGET_PORT" ]]; then
  pkill -f "vllm.entrypoints.openai.api_server.*--port $TARGET_PORT" || true
fi
if [[ -n "$JUDGE_PORT" ]]; then
  pkill -f "vllm.entrypoints.openai.api_server.*--port $JUDGE_PORT" || true
fi
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 3

wait_for_service() {
    local port=$1
    local name=$2
    local retries=60
    echo "Waiting for $name on port $port..."
    for i in $(seq 1 $retries); do
        if curl --noproxy "*" -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/v1/models" | grep -q "200"; then
            echo "$name is ready!"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: $name failed to start."
    return 1
}

if [[ "$TARGET_API_BASE" == http://localhost:* || "$TARGET_API_BASE" == http://127.0.0.1:* ]]; then
  echo "Starting Target Model (local vLLM API server)..."
  CUDA_VISIBLE_DEVICES=0 nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
      --model "$TARGET_MODEL" \
      --served-model-name "$TARGET_MODEL" \
      --trust-remote-code \
      --port "${TARGET_PORT:-21111}" \
      --tensor-parallel-size "${TARGET_TENSOR_PARALLEL_SIZE:-1}" \
      --gpu-memory-utilization "${TARGET_GPU_MEM_UTIL:-0.4}" \
      --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" \
      > "$LOG_DIR/vllm_target_$TIMESTAMP.log" 2>&1 &

  wait_for_service "${TARGET_PORT:-21111}" "Target Model" || { tail -n 50 "$LOG_DIR/vllm_target_$TIMESTAMP.log"; exit 1; }
else
  echo "Target api_base is remote ($TARGET_API_BASE). Skip starting local vLLM server."
fi

if [[ "$JUDGE_API_BASE" == http://localhost:* || "$JUDGE_API_BASE" == http://127.0.0.1:* ]]; then
  echo "Starting Judge Model (local vLLM API server)..."
  CUDA_VISIBLE_DEVICES=0 nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
      --model "$JUDGE_MODEL" \
      --served-model-name "$JUDGE_MODEL" \
      --trust-remote-code \
      --port "${JUDGE_PORT:-21112}" \
      --tensor-parallel-size "${JUDGE_TENSOR_PARALLEL_SIZE:-1}" \
      --gpu-memory-utilization "${JUDGE_GPU_MEM_UTIL:-0.4}" \
      --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" \
      > "$LOG_DIR/vllm_judge_$TIMESTAMP.log" 2>&1 &

  wait_for_service "${JUDGE_PORT:-21112}" "Judge Model" || { tail -n 50 "$LOG_DIR/vllm_judge_$TIMESTAMP.log"; exit 1; }
else
  if [[ -n "$JUDGE_API_BASE" ]]; then
    echo "Judge api_base is remote ($JUDGE_API_BASE). Skip starting local vLLM server."
  else
    echo "Judge api_base is empty (likely local in-process judge). Skip starting API server."
  fi
fi

echo "Running Evaluation..."
RUN_LOG="$LOG_DIR/mbef_run_$TIMESTAMP.log"
"$PY_BIN" tools/run.py "$CONFIG_PATH" "${OVERRIDES[@]}" > "$RUN_LOG" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Success! Results saved."
    echo "Report: $MBEF_ROOT/$OUTPUT_DIR_REL/report.md"
else
    echo "Failure (exit $EXIT_CODE). Check log: $RUN_LOG"
    tail -n 20 "$RUN_LOG"
fi
exit $EXIT_CODE
