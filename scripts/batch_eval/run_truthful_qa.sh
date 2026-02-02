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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
cd "$MBEF_ROOT"

export HF_HOME="${HF_HOME:-/mnt/shared-storage-user/zhangbo1/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

pip install -q pandas datasets -i http://mirrors.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.h.pjlab.org.cn || true

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"

proxy_url="${PROXY_URL:-${DEFAULT_PROXY_URL:-}}"
if [[ -n "$proxy_url" ]]; then
  export http_proxy="$proxy_url"
  export https_proxy="$proxy_url"
  export HTTP_PROXY="$proxy_url"
  export HTTPS_PROXY="$proxy_url"
fi

PY_BIN="$(which python3)"
echo "Parsing config: $CONFIG_PATH"
PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config --config "$CONFIG_PATH" --mbef-root "$MBEF_ROOT" --format bash --strict "${OVERRIDES[@]}")" \
  || { echo "ERROR: failed to parse config: $CONFIG_PATH" >&2; exit 1; }
eval "$PARSED_ENV"

EXP_NAME="${EXP_NAME:-truthful_qa_task}"
LOG_DIR="${LOG_DIR:-$MBEF_ROOT/scripts/rlaunch_logs/$EXP_NAME}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

echo "--------------------------------------------------------"
echo "Task:   $EXP_NAME"
if [[ -n "$TARGET_MODEL" ]]; then
  echo "Model:  $TARGET_MODEL"
fi
echo "Logs:   $LOG_DIR"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi
echo "--------------------------------------------------------"

RUN_LOG="$LOG_DIR/mbef_truthfulqa_$TIMESTAMP.log"

cleanup() {
  echo "Cleaning up..."
  pkill -P $$ || true
  if [[ -n "$TARGET_PORT" ]]; then
    pkill -f "vllm.entrypoints.openai.api_server.*--port $TARGET_PORT" || true
  else
    pkill -f "vllm.entrypoints.openai.api_server" || true
  fi
}
trap cleanup EXIT

wait_for_service() {
  local port=$1
  local name=$2
  local retries=180
  echo "Waiting for $name on port $port..."
  for i in $(seq 1 $retries); do
    if curl --noproxy "*" -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/v1/models" | grep -q "200"; then
      echo "$name is ready!"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: $name failed to start."
  return 1
}

if [[ "$TARGET_API_BASE" == http://localhost:* || "$TARGET_API_BASE" == http://127.0.0.1:* ]]; then
  echo "Starting Target Model (local vLLM OpenAI server)..."

  if [[ -n "$TARGET_PORT" ]]; then
    pkill -f "vllm.entrypoints.openai.api_server.*--port $TARGET_PORT" || true
  else
    pkill -f "vllm.entrypoints.openai.api_server" || true
  fi
  sleep 2

  nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$TARGET_MODEL" \
    --served-model-name "$TARGET_MODEL" \
    --trust-remote-code \
    --port "${TARGET_PORT:-21111}" \
    --tensor-parallel-size "${TARGET_TENSOR_PARALLEL_SIZE:-1}" \
    --gpu-memory-utilization "${TARGET_GPU_MEM_UTIL:-0.8}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" \
    > "$LOG_DIR/vllm_target_$TIMESTAMP.log" 2>&1 &

  wait_for_service "${TARGET_PORT:-21111}" "Target Model" || { tail -n 50 "$LOG_DIR/vllm_target_$TIMESTAMP.log"; exit 1; }
else
  echo "Target api_base is remote ($TARGET_API_BASE). This bench only supports local open-source models."
  exit 2
fi

"$PY_BIN" tools/run.py "$CONFIG_PATH" \
  "${OVERRIDES[@]}" \
  --runner.stage all \
  > "$RUN_LOG" 2>&1

echo "Done. Log: $RUN_LOG"

