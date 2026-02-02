#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -euo pipefail

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

CONFIG_PATH="${1:-}"
shift || true
OVERRIDES=("$@")
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path> [overrides...]" >&2
  exit 2
fi

CONFIG_PATH="$(readlink -f "$CONFIG_PATH")"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH" >&2
  exit 1
fi




PY_BIN="python3"
command -v "$PY_BIN" >/dev/null 2>&1 || PY_BIN="python"
command -v "$PY_BIN" >/dev/null 2>&1 || { echo "ERROR: python3/python not found" >&2; exit 1; }
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
cd "$MBEF_ROOT"

eval "$("$PY_BIN" -m uni_eval.cli.parse_eval_config \
  --config "$CONFIG_PATH" \
  --format bash \
  --strict \
  --mbef-root "$MBEF_ROOT" \
  "${OVERRIDES[@]}")"

mkdir -p "$LOG_DIR"
TS="$(date +"%Y%m%d%H%M%S")"

TARGET_LOG="$LOG_DIR/vllm_target_${TS}.log"
STAGE1_LOG="$LOG_DIR/mbef_stage1_${TS}.log"

STAGE_DIR="$RESULT_DIR/_stage1"
mkdir -p "$STAGE_DIR"
RESP_JSONL="${FLAMES_RESP_JSONL:-$STAGE_DIR/flames_target_responses.jsonl}"
CFG_STAGE1="$LOG_DIR/_cfg_stage1_${TS}.yaml"

TARGET_GPU_MEM_UTIL="${TARGET_GPU_MEM_UTIL:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
TARGET_CUDA_VISIBLE_DEVICES="${TARGET_CUDA_VISIBLE_DEVICES:-0}"
TARGET_API_BASE="${TARGET_API_BASE:-}"

cleanup() {
  echo "Cleaning up..."
  pkill -P $$ || true
  if [[ -n "${TARGET_PORT:-}" ]]; then
    pkill -f "vllm.entrypoints.openai.api_server.*--port $TARGET_PORT" || true
  fi
}
trap cleanup EXIT

wait_port() {
  local port="$1"
  local name="$2"
  local log_file="$3"
  echo "Waiting for $name on port $port..."
  for _ in $(seq 1 60); do
    if curl --noproxy "*" -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/v1/models" | grep -q "200"; then
      echo "$name is ready!"
      return 0
    fi
    sleep 5
  done
  echo "ERROR: $name failed to start (port $port)."
  tail -n 50 "$log_file" || true
  return 1
}

echo "--------------------------------------------------------"
echo "Stage:   gen"
echo "Config:   $CONFIG_PATH"
echo "ExpName:  $EXP_NAME"
echo "Result:   $RESULT_DIR"
echo "StageDir: $STAGE_DIR"
echo "RespJSONL:$RESP_JSONL"
echo "Target:   $TARGET_MODEL (api_base=$TARGET_API_BASE, port=$TARGET_PORT, CUDA=$TARGET_CUDA_VISIBLE_DEVICES)"
echo "Logs:     $LOG_DIR"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi
echo "--------------------------------------------------------"

STARTED_VLLM=0
if [[ -n "${TARGET_PORT:-}" ]]; then
  pkill -f "vllm.entrypoints.openai.api_server.*--port $TARGET_PORT" || true
  sleep 2

  echo "Starting Target Model via local vLLM ($TARGET_MODEL) ..."
  CUDA_VISIBLE_DEVICES="$TARGET_CUDA_VISIBLE_DEVICES" nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$TARGET_MODEL" \
    --served-model-name "$TARGET_MODEL" \
    --trust-remote-code \
    --port "$TARGET_PORT" \
    --gpu-memory-utilization "$TARGET_GPU_MEM_UTIL" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    > "$TARGET_LOG" 2>&1 &

  wait_port "$TARGET_PORT" "Target Model" "$TARGET_LOG" || exit 1
  STARTED_VLLM=1
else
  echo "Target api_base is remote (no localhost port parsed). Skip starting vLLM; stage-1 will call APIModel directly."
fi

echo "Generating stage-1 config (target responses)..."
"$PY_BIN" tools/flames_make_stage_config.py \
  --config "$CONFIG_PATH" \
  --stage 1 \
  --out "$CFG_STAGE1" \
  --resp-jsonl "$RESP_JSONL" \
  --stage-dir "$STAGE_DIR" \
  > "$LOG_DIR/flames_make_stage1_${TS}.log" 2>&1

echo "Stage-1: run target -> $RESP_JSONL"
"$PY_BIN" tools/run.py "$CFG_STAGE1" "${OVERRIDES[@]}" > "$STAGE1_LOG" 2>&1

EXIT_CODE=$?
if [[ "$STARTED_VLLM" == "1" ]]; then
  echo "Stopping vLLM target (port $TARGET_PORT)..."
  pkill -f "vllm.entrypoints.openai.api_server.*--port $TARGET_PORT" || true
  sleep 3
fi

if [ $EXIT_CODE -eq 0 ]; then
  echo "Success. Responses at: $RESP_JSONL"
else
  echo "Failure (exit $EXIT_CODE). See log: $STAGE1_LOG"
  tail -n 50 "$STAGE1_LOG" || true
fi
exit $EXIT_CODE

