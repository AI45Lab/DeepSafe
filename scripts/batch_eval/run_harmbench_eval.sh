#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1,.pjlab.org.cn,.h.pjlab.org.cn"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1,.pjlab.org.cn,.h.pjlab.org.cn"

pip install spacy==3.8.11 datasketch -i http://mirrors.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.h.pjlab.org.cn
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
cd "$MBEF_ROOT"

echo "Installing spaCy English model..."
pip install $MBEF_ROOT/data/harmbench/en_core_web_sm-3.8.0-py3-none-any.whl -i http://mirrors.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.h.pjlab.org.cn

PY_BIN=$(which python3)
echo "Parsing config: $CONFIG_PATH"
PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config --config "$CONFIG_PATH" --mbef-root "$MBEF_ROOT" --format bash --strict "${OVERRIDES[@]}")" \
  || { echo "ERROR: failed to parse config: $CONFIG_PATH"; exit 1; }
eval "$PARSED_ENV"

EXP_NAME="${EXP_NAME:-harmbench_task}"
LOG_DIR="${LOG_DIR:-$MBEF_ROOT/logs/$EXP_NAME/eval}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

if [[ -n "$OUTPUT_DIR_REL" ]]; then
  PREDICTIONS_PATH_DEFAULT="$MBEF_ROOT/$OUTPUT_DIR_REL/predictions.jsonl"
else
  PREDICTIONS_PATH_DEFAULT="$MBEF_ROOT/results/$EXP_NAME/predictions.jsonl"
fi
PREDICTIONS_PATH="${PREDICTIONS_PATH:-$PREDICTIONS_PATH_DEFAULT}"

echo "--------------------------------------------------------"
echo "Stage:  eval"
echo "Task:   $EXP_NAME"
echo "Pred:   $PREDICTIONS_PATH"
if [[ -n "$JUDGE_MODEL" ]]; then
  echo "Judge:  $JUDGE_MODEL"
fi
echo "Logs:   $LOG_DIR"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi
echo "--------------------------------------------------------"

RUN_LOG="$LOG_DIR/mbef_eval_$TIMESTAMP.log"

cleanup() {
  echo "Cleaning up..."
  pkill -P $$ || true
  if [[ -n "$JUDGE_PORT" ]]; then
    pkill -f "vllm.entrypoints.openai.api_server.*--port $JUDGE_PORT" || true
  else
    pkill -f "vllm.entrypoints.openai.api_server" || true
  fi
}
trap cleanup EXIT

if [[ -n "$JUDGE_PORT" ]]; then
  pkill -f "vllm.entrypoints.openai.api_server.*--port $JUDGE_PORT" || true
else
  pkill -f "vllm.entrypoints.openai.api_server" || true
fi
sleep 3

wait_for_service() {
  local port=$1
  local name=$2
  local retries=10000
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

if [[ "$JUDGE_API_BASE" == http://localhost:* || "$JUDGE_API_BASE" == http://127.0.0.1:* ]]; then
  echo "Starting Judge Model $JUDGE_MODEL (local vLLM API server)..." 
  VLLM_LOGGING_LEVEL=DEBUG \
  VLLM_PRINT_PROMPT=1 \
  nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" \
    --served-model-name "$JUDGE_MODEL" \
    --trust-remote-code \
    --port "${JUDGE_PORT:-21111}" \
    --tensor-parallel-size "${JUDGE_TENSOR_PARALLEL_SIZE:-1}" \
    --gpu-memory-utilization "${JUDGE_GPU_MEM_UTIL:-0.8}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-2048}" \
    --chat-template /mnt/shared-storage-user/ai4good2-share/models/HarmBench-Llama-2-13b-cls/chat_template.jinja \
    --enable-log-requests \
    --enable-log-outputs \
    --uvicorn-log-level debug \
    > "$LOG_DIR/vllm_judge_$TIMESTAMP.log" 2>&1 &

  wait_for_service "${JUDGE_PORT:-21111}" "Judge Model" || { tail -n 50 "$LOG_DIR/vllm_judge_$TIMESTAMP.log"; exit 1; }
else
  echo "Judge api_base is remote ($JUDGE_API_BASE). Skip starting local vLLM server."
fi

"$PY_BIN tools/run.py" "$CONFIG_PATH" \
  --runner.stage eval \
  "${OVERRIDES[@]}" \
  > "$RUN_LOG" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "Success. Results at: $MBEF_ROOT/$OUTPUT_DIR_REL"
  echo "Report: $MBEF_ROOT/$OUTPUT_DIR_REL/report.md"
else
  echo "Failure exit $EXIT_CODE. Check log: $RUN_LOG"
  tail -n 30 "$RUN_LOG"
fi
exit $EXIT_CODE
