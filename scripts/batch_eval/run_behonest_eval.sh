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

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
cd "$MBEF_ROOT"

python3 -m pip install -q pandas datasets -i http://mirrors.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.h.pjlab.org.cn || true

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"

export HF_HOME="${HF_HOME:-/mnt/shared-storage-user/linqihao/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

PY_BIN="$(which python3)"
echo "Parsing config: $CONFIG_PATH"
PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config --config "$CONFIG_PATH" --mbef-root "$MBEF_ROOT" --format bash --strict "${OVERRIDES[@]}")" \
  || { echo "ERROR: failed to parse config: $CONFIG_PATH"; exit 1; }
eval "$PARSED_ENV"

EXP_NAME="${EXP_NAME:-behonest_task}"
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
echo "Stage:  eval"
echo "Task:   $EXP_NAME"
echo "Pred:   $PREDICTIONS_PATH"
if [[ -n "$JUDGE_MODEL" ]]; then
  echo "Judge:  $JUDGE_MODEL"
  if [[ -n "$JUDGE_PORT" ]]; then
    echo "Judge Port: $JUDGE_PORT"
  fi
fi
echo "Logs:   $LOG_DIR"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi
echo "--------------------------------------------------------"

cleanup() {
  echo "Cleaning up..."
  pkill -P $$ || true
  if [[ -n "$JUDGE_PORT" ]]; then
    pkill -f "vllm.entrypoints.openai.api_server.*--port $JUDGE_PORT" || true
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

if [[ -n "$JUDGE_MODEL" ]]; then
  if [[ "$JUDGE_API_BASE" == http://localhost:* || "$JUDGE_API_BASE" == http://127.0.0.1:* ]]; then
    echo "Starting Judge Model (local vLLM API server)..."

    if [[ -n "$JUDGE_PORT" ]]; then
      pkill -f "vllm.entrypoints.openai.api_server.*--port $JUDGE_PORT" || true
    fi
    sleep 2

    export PATH="/usr/bin:/bin:/usr/sbin:/sbin"

    nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
      --model "$JUDGE_MODEL" \
      --served-model-name "$JUDGE_MODEL" \
      --trust-remote-code \
      --port "${JUDGE_PORT:-22111}" \
      --tensor-parallel-size "${JUDGE_TENSOR_PARALLEL_SIZE:-1}" \
      --gpu-memory-utilization "${JUDGE_GPU_MEM_UTIL:-0.8}" \
      --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" \
      > "$LOG_DIR/vllm_judge_$TIMESTAMP.log" 2>&1 &

    wait_for_service "${JUDGE_PORT:-22111}" "Judge Model" || { tail -n 50 "$LOG_DIR/vllm_judge_$TIMESTAMP.log"; exit 1; }
  else
    echo "Judge api_base is remote ($JUDGE_API_BASE). Using remote judge."
  fi
else
  echo "No judge model configured. Self-Knowledge tasks (Unknowns/Knowns) do not need judge."
fi

echo "Evaluating predictions..."
RUN_LOG="$LOG_DIR/mbef_eval_$TIMESTAMP.log"
"$PY_BIN" tools/run.py "$CONFIG_PATH" \
  "${OVERRIDES[@]}" \
  --runner.stage eval \
  --runner.predictions_path "$PREDICTIONS_PATH" \
  --runner.require_predictions true \
  --evaluator.use_precomputed_predictions true \
  --evaluator.require_precomputed_predictions true \
  --evaluator.prediction_field prediction \
  --model.type NoOpModel \
  > "$RUN_LOG" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "Success. Results at: $MBEF_ROOT/$OUTPUT_DIR_REL"
  if [[ -f "$MBEF_ROOT/$OUTPUT_DIR_REL/report.md" ]]; then
    echo "Report: $MBEF_ROOT/$OUTPUT_DIR_REL/report.md"
  fi
  echo "Log: $RUN_LOG"
else
  echo "Failure (exit $EXIT_CODE). Check log: $RUN_LOG"
  tail -n 30 "$RUN_LOG"
fi
exit $EXIT_CODE
