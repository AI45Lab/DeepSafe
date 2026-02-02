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




export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH
export http_proxy="HTTP_PROXY"
export https_proxy="HTTP_PROXY"
export HTTP_PROXY="HTTP_PROXY"
export HTTPS_PROXY="HTTP_PROXY"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
cd "$MBEF_ROOT"

PY_BIN="$(which python3)"
echo "Parsing config: $CONFIG_PATH"
PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config --config "$CONFIG_PATH" --mbef-root "$MBEF_ROOT" --format bash --strict "${OVERRIDES[@]}")" \
  || { echo "ERROR: failed to parse config: $CONFIG_PATH"; exit 1; }
eval "$PARSED_ENV"

add_no_proxy_host() {
  local host="$1"
  if [[ -z "$host" ]]; then return 0; fi
  local cur="${no_proxy:-${NO_PROXY:-}}"
  if [[ -z "$cur" ]]; then
    cur="localhost,127.0.0.1,0.0.0.0,::1"
  fi
  if echo ",$cur," | grep -q ",$host,"; then
    export no_proxy="$cur"
    export NO_PROXY="$cur"
    return 0
  fi
  cur="${cur},${host}"
  export no_proxy="$cur"
  export NO_PROXY="$cur"
}

TARGET_API_HOST=""
if [[ -n "${TARGET_API_BASE:-}" ]]; then
  TARGET_API_HOST="$(python3 - "$TARGET_API_BASE" << 'PY'
import sys
from urllib.parse import urlparse
u = urlparse((sys.argv[1] if len(sys.argv) > 1 else "").strip())
print(u.hostname or "")
PY
)"
fi

SHOULD_BYPASS_PROXY="0"
if [[ -n "$TARGET_API_HOST" ]]; then
  SHOULD_BYPASS_PROXY="$(python3 - "$TARGET_API_HOST" << 'PY'
import sys, ipaddress
h = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
if h in ("localhost","127.0.0.1","0.0.0.0","::1"):
    print("1"); raise SystemExit
try:
    ip = ipaddress.ip_address(h)
except ValueError:
    print("0"); raise SystemExit

if ip.is_private:
    print("1"); raise SystemExit

if ip in ipaddress.ip_network("100.64.0.0/10"):
    print("1"); raise SystemExit

print("0")
PY
)"
fi

if [[ "$SHOULD_BYPASS_PROXY" == "1" ]]; then
  add_no_proxy_host "$TARGET_API_HOST"
  echo "NO_PROXY appended for in-cluster endpoint: $TARGET_API_HOST"
else
  echo "Keeping proxy for target endpoint (not detected as in-cluster): $TARGET_API_HOST"
fi

echo "Proxy enabled for remote APIs."
echo "TARGET_API_BASE=$TARGET_API_BASE"
echo "TARGET_API_HOST=$TARGET_API_HOST"
echo "NO_PROXY=$NO_PROXY"

EXP_NAME="${EXP_NAME:-manipulation_persuasion_conv}"
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
echo "Stage:  gen (manipulation_persuasion_conv)"
echo "Task:   $EXP_NAME"
echo "Target: $TARGET_MODEL (port ${TARGET_PORT:-<remote>})"
echo "Pred:   $PREDICTIONS_PATH"
echo "Logs:   $LOG_DIR"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi
echo "--------------------------------------------------------"

cleanup() {
  echo "Cleaning up..."
  pkill -P $$ >/dev/null 2>&1 || true
  if [[ -n "$TARGET_PORT" ]]; then
    pkill -f "vllm.entrypoints.openai.api_server.*--port $TARGET_PORT" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

wait_for_service() {
  local port=$1
  local name=$2
  local retries=1800
  echo "Waiting for $name on port $port..."
  for _ in $(seq 1 $retries); do
    local code
    code="$(curl --noproxy "*" -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/v1/models" || true)"
    if [[ "$code" == "200" ]]; then
      echo "$name is ready!"
      return 0
    fi
    sleep 5
  done
  echo "ERROR: $name failed to start."
  return 1
}

if [[ -n "$TARGET_PORT" ]]; then
  pkill -f "vllm.entrypoints.openai.api_server.*--port $TARGET_PORT" >/dev/null 2>&1 || true
  sleep 2
  TARGET_TP="${TARGET_TENSOR_PARALLEL_SIZE:-1}"
  TARGET_GPU_UTIL="${TARGET_GPU_MEM_UTIL:-0.85}"
  VLLM_LOG="$LOG_DIR/vllm_target_$TIMESTAMP.log"
  echo "Starting Target vLLM..."
  nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$TARGET_MODEL" \
    --served-model-name "$TARGET_MODEL" \
    --trust-remote-code \
    --port "$TARGET_PORT" \
    --tensor-parallel-size "$TARGET_TP" \
    --gpu-memory-utilization "$TARGET_GPU_UTIL" \
    --max-model-len 4096 \
    > "$VLLM_LOG" 2>&1 &
  wait_for_service "$TARGET_PORT" "Target Model" || { tail -n 50 "$VLLM_LOG"; exit 1; }
else
  echo "Target api_base is remote; skipping local vLLM startup."
fi

RUN_LOG="$LOG_DIR/mbef_gen_$TIMESTAMP.log"
echo "Generating predictions via evaluator.generate_predictions()..."
"$PY_BIN" tools/run.py "$CONFIG_PATH" \
  --runner.stage gen \
  --runner.predictions_path "$PREDICTIONS_PATH" \
  --runner.use_evaluator_gen true \
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

