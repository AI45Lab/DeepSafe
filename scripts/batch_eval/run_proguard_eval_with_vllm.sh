#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -euo pipefail

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

CFG="${1:-}"
OUT="${2:-}"
PROGUARD_MODEL_PATH="${3:-}"
SERVED_NAME="${4:-ProGuard-7B}"
PORT="${5:-21122}"

if [[ -z "${CFG}" || -z "${OUT}" || -z "${PROGUARD_MODEL_PATH}" ]]; then
  echo "Usage: $0 <config_yaml> <output_dir> <proguard_model_path> [served_model_name] [port]" >&2
  exit 2
fi

if [[ "$CFG" != /* ]]; then
  CFG="$MBEF_ROOT/$CFG"
fi
CFG="$(readlink -f "$CFG")"
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG" >&2
  exit 1
fi

cd "$MBEF_ROOT"




export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"

PY_BIN="python3"
command -v "$PY_BIN" >/dev/null 2>&1 || PY_BIN="python"

VLLM_LOG="$MBEF_ROOT/scripts/rlaunch_logs/proguard/vllm_${PORT}_$(date +"%Y%m%d%H%M%S").log"
mkdir -p "$(dirname "$VLLM_LOG")"

cleanup() {
  pkill -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" || true
}
trap cleanup EXIT

wait_port() {
  local port="$1"
  echo "Waiting for ProGuard vLLM on port $port ..."
  for _ in $(seq 1 60); do
    if curl --noproxy "*" -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/v1/models" | grep -q "200"; then
      echo "ProGuard vLLM is ready."
      return 0
    fi
    sleep 2
  done
  echo "ERROR: ProGuard vLLM failed to start on port $port" >&2
  tail -n 50 "$VLLM_LOG" || true
  return 1
}

pkill -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" || true
sleep 2

echo "Starting ProGuard vLLM: model=$PROGUARD_MODEL_PATH served_name=$SERVED_NAME port=$PORT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$PROGUARD_MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --trust-remote-code \
  --port "$PORT" \
  --gpu-memory-utilization "${PROGUARD_GPU_MEM_UTIL:-0.90}" \
  --max-model-len "${PROGUARD_MAX_MODEL_LEN:-4096}" \
  > "$VLLM_LOG" 2>&1 &

wait_port "$PORT" || exit 1

bash "$MBEF_ROOT/scripts/run_proguard_eval.sh" "$CFG" "$OUT" "http://127.0.0.1:${PORT}/v1" "$SERVED_NAME"

