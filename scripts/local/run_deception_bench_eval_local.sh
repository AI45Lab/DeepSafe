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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/cuda-12.9/compat"
cd "$MBEF_ROOT"

PY_BIN=$(which python3)
echo "Parsing config: $CONFIG_PATH"
PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config --config "$CONFIG_PATH" --mbef-root "$MBEF_ROOT" --format bash --strict "${OVERRIDES[@]}")" \
  || { echo "ERROR: failed to parse config: $CONFIG_PATH" >&2; exit 1; }
eval "$PARSED_ENV"

EXP_NAME="${EXP_NAME:-deception_bench_task}"
LOG_DIR="${LOG_DIR:-$MBEF_ROOT/scripts/local_logs/$EXP_NAME}"
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
  echo "Judge:  $JUDGE_MODEL (api_base=$JUDGE_API_BASE port=${JUDGE_PORT:-})"
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
  if [[ -n "$JUDGE_PORT" ]]; then
    pkill -f "vllm.entrypoints.openai.api_server.*--port $JUDGE_PORT" || true
  else
    pkill -f "vllm.entrypoints.openai.api_server" || true
  fi
  sleep 3

  if ! "$PY_BIN" -c "import vllm" 2>/dev/null; then
    echo "ERROR: vLLM is not installed in the current Python environment ($PY_BIN)" >&2
    echo "Please activate the correct conda/virtual environment with vLLM installed, or install vLLM:" >&2
    echo "  pip install vllm" >&2
    exit 1
  fi
  echo "Starting Judge Model (local vLLM API server)..."
  nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" \
    --served-model-name "$JUDGE_MODEL" \
    --trust-remote-code \
    --port "${JUDGE_PORT:-21112}" \
    --tensor-parallel-size "${JUDGE_TENSOR_PARALLEL_SIZE:-1}" \
    --gpu-memory-utilization "${JUDGE_GPU_MEM_UTIL:-0.8}" \
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

RUN_LOG="$LOG_DIR/mbef_eval_$TIMESTAMP.log"

"$PY_BIN" tools/run.py "$CONFIG_PATH" \
  --runner.stage eval \
  --runner.predictions_path "$PREDICTIONS_PATH" \
  --runner.require_predictions true \
  --runner.prediction_field prediction \
  --evaluator.use_precomputed_predictions true \
  --evaluator.require_precomputed_predictions true \
  --model.type NoOpModel \
  "${OVERRIDES[@]}" \
  > "$RUN_LOG" 2>&1

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
  echo "Success. Results at: $MBEF_ROOT/$OUTPUT_DIR_REL"
  echo "Report: $MBEF_ROOT/$OUTPUT_DIR_REL/report.md"
else
  echo "Failure (exit $EXIT_CODE). Check log: $RUN_LOG"
  tail -n 30 "$RUN_LOG"
fi
exit $EXIT_CODE
