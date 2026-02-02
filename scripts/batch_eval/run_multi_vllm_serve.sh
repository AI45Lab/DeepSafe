#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -euo pipefail

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export NO_PROXY="$no_proxy"

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
cd "$MBEF_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"

PY_BIN="$(which python3)"

pip install fastapi uvicorn httpx -i http://mirrors.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.h.pjlab.org.cn

ROUTER_PORT="${ROUTER_PORT:-21111}"
FIRST_PORT="${FIRST_PORT:-21112}"
SECOND_PORT="${SECOND_PORT:-21113}"

SECOND_ENABLED="${SECOND_ENABLED:-0}"
SECOND_NAME="${SECOND_NAME:-Llama-3.3-70B-Instruct}"
SECOND_PATH="${SECOND_PATH:-/mnt/shared-storage-user/ai4good2-share/models/meta-llama/Llama-3.3-70B-Instruct}"
SECOND_TP="${SECOND_TP:-2}"
SECOND_GPU_UTIL="${SECOND_GPU_UTIL:-0.85}"
SECOND_MAX_MODEL_LEN="${SECOND_MAX_MODEL_LEN:-8192}"

FIRST_NAME="gemma-3-27b-it"
FIRST_PATH="/mnt/shared-storage-user/ai4good2-share/models/google/gemma-3-27b-it"
FIRST_TP="1"
FIRST_GPU_UTIL="0.9"
FIRST_MAX_MODEL_LEN="8192"

LOG_DIR="${LOG_DIR:-$MBEF_ROOT/scripts/rlaunch_logs/multi_vllm}"
mkdir -p "$LOG_DIR"
TS="$(date +'%Y%m%d%H%M%S')"

cleanup() {
  echo "Cleaning up..."
  pkill -P $$ >/dev/null 2>&1 || true
  pkill -f "vllm.entrypoints.openai.api_server.*--port $FIRST_PORT" >/dev/null 2>&1 || true
  pkill -f "vllm.entrypoints.openai.api_server.*--port $SECOND_PORT" >/dev/null 2>&1 || true
  pkill -f "uvicorn .*multi_vllm_router:app.*--port $ROUTER_PORT" >/dev/null 2>&1 || true
}
trap cleanup EXIT

wait_for() {
  local port="$1"
  local name="$2"
  echo "Waiting for $name on :$port ..."
  local i=0
  while [ "$i" -lt 1800 ]; do
    i=$((i + 1))
    code="$(curl --noproxy "*" -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/v1/models" || true)"
    if [ "$code" = "200" ]; then
      echo "$name is ready."
      return 0
    fi
    sleep 2
  done
  echo "ERROR: $name failed to start on :$port"
  return 1
}

echo "Starting vLLM servers..."
FIRST_LOG="$LOG_DIR/vllm_${FIRST_NAME}_${TS}.log"
SECOND_LOG="$LOG_DIR/vllm_${SECOND_NAME}_${TS}.log"

CUDA_VISIBLE_DEVICES=0 nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$FIRST_PATH" \
  --served-model-name "$FIRST_NAME" \
  --trust-remote-code \
  --port "$FIRST_PORT" \
  --tensor-parallel-size "$FIRST_TP" \
  --gpu-memory-utilization "$FIRST_GPU_UTIL" \
  --max-model-len "$FIRST_MAX_MODEL_LEN" \
  >"$FIRST_LOG" 2>&1 &

if [[ "$SECOND_ENABLED" == "1" ]]; then
  CUDA_VISIBLE_DEVICES=1,2 nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$SECOND_PATH" \
    --served-model-name "$SECOND_NAME" \
    --trust-remote-code \
    --port "$SECOND_PORT" \
    --tensor-parallel-size "$SECOND_TP" \
    --gpu-memory-utilization "$SECOND_GPU_UTIL" \
    --max-model-len "$SECOND_MAX_MODEL_LEN" \
    >"$SECOND_LOG" 2>&1 &
fi

wait_for "$FIRST_PORT" "$FIRST_NAME" || { tail -n 80 "$FIRST_LOG"; exit 1; }
if [[ "$SECOND_ENABLED" == "1" ]]; then
  wait_for "$SECOND_PORT" "$SECOND_NAME" || { tail -n 80 "$SECOND_LOG"; exit 1; }
fi

echo "Starting router (one port for all models): :$ROUTER_PORT"
ROUTER_LOG="$LOG_DIR/router_${TS}.log"
if [[ "$SECOND_ENABLED" == "1" ]]; then
  export UPSTREAMS="$FIRST_NAME=http://127.0.0.1:$FIRST_PORT/v1,$SECOND_NAME=http://127.0.0.1:$SECOND_PORT/v1"
else
  export UPSTREAMS="$FIRST_NAME=http://127.0.0.1:$FIRST_PORT/v1"
fi
nohup "$PY_BIN" -m uvicorn scripts.multi_vllm_router:app --host 0.0.0.0 --port "$ROUTER_PORT" \
  >"$ROUTER_LOG" 2>&1 &

echo "Waiting for router..."
_i=0
while [ "$_i" -lt 120 ]; do
  _i=$((_i + 1))
  if curl --noproxy "*" -s "http://127.0.0.1:$ROUTER_PORT/health" | grep -q "\"ok\":true"; then
    echo "Router is ready."
    break
  fi
  sleep 1
done

echo "--------------------------------------------------------"
echo "Unified endpoint (ONE port): http://127.0.0.1:$ROUTER_PORT/v1"
echo "Models:"
echo "  - $FIRST_NAME"
if [[ "$SECOND_ENABLED" == "1" ]]; then
  echo "  - $SECOND_NAME"
fi
echo "Logs:"
echo "  - $FIRST_LOG"
if [[ "$SECOND_ENABLED" == "1" ]]; then
  echo "  - $SECOND_LOG"
fi
echo "  - $ROUTER_LOG"
echo "--------------------------------------------------------"

wait

