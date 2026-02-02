#!/bin/bash
set -eo pipefail

CONFIG_PATH="${1:-}"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
cd "$MBEF_ROOT"

export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0,::1}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0,::1}"

PY_BIN="${PY_BIN:-python3}"

echo "Parsing config: $CONFIG_PATH"
PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config \
  --config "$CONFIG_PATH" \
  --mbef-root "$MBEF_ROOT" \
  --format bash \
  "${OVERRIDES[@]}")" || { echo "ERROR: failed to parse config: $CONFIG_PATH"; exit 1; }
eval "$PARSED_ENV"

EXP_NAME="${EXP_NAME:-$(basename "$CONFIG_PATH" | sed 's/\.ya\?ml$//')}"
LOG_DIR_DEFAULT="$MBEF_ROOT/scripts/local_logs/$EXP_NAME"
if [[ -n "${MBEF_LOG_DIR:-}" ]]; then
  LOG_DIR="$MBEF_LOG_DIR"
else
  LOG_DIR="$LOG_DIR_DEFAULT"
fi
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d%H%M%S")"

if [[ -n "${OUTPUT_DIR_REL:-}" ]]; then
  PREDICTIONS_PATH_DEFAULT="$MBEF_ROOT/$OUTPUT_DIR_REL/predictions.jsonl"
else
  PREDICTIONS_PATH_DEFAULT="$MBEF_ROOT/results/$EXP_NAME/predictions.jsonl"
fi
PREDICTIONS_PATH="${PREDICTIONS_PATH:-$PREDICTIONS_PATH_DEFAULT}"

echo "--------------------------------------------------------"
echo "Stage:  eval (local)"
echo "Task:   $EXP_NAME"
echo "Pred:   $PREDICTIONS_PATH"
if [[ -n "${JUDGE_MODEL:-}" ]]; then
  echo "Judge:  $JUDGE_MODEL"
fi
echo "Judge API: ${JUDGE_API_BASE:-<empty>}"
echo "Logs:   $LOG_DIR"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi
echo "--------------------------------------------------------"

wait_for_service() {
  local port="$1"
  local name="$2"
  local retries=600
  echo "Waiting for $name on port $port..."
  for _ in $(seq 1 "$retries"); do
    if curl --noproxy "*" -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/v1/models" | grep -q "200"; then
      echo "$name is ready!"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: $name failed to start."
  return 1
}

cleanup() {
  if [[ -n "${JUDGE_PORT:-}" ]]; then
    pkill -f "vllm.entrypoints.openai.api_server.*--port ${JUDGE_PORT}" >/dev/null 2>&1 || true
  else
    pkill -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ -n "${JUDGE_API_BASE:-}" && -n "${JUDGE_PORT:-}" ]]; then
  if [[ "$JUDGE_API_BASE" == http://localhost:* || "$JUDGE_API_BASE" == http://127.0.0.1:* ]]; then
    if [[ -n "${JUDGE_MODEL:-}" && -e "${JUDGE_MODEL}" ]]; then
      echo "Starting local judge vLLM server (APIModel + localhost api_base)..."
      pkill -f "vllm.entrypoints.openai.api_server.*--port ${JUDGE_PORT}" >/dev/null 2>&1 || true
      sleep 2
      nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
        --model "$JUDGE_MODEL" \
        --served-model-name "$JUDGE_MODEL" \
        --trust-remote-code \
        --port "${JUDGE_PORT}" \
        --tensor-parallel-size "${JUDGE_TENSOR_PARALLEL_SIZE:-1}" \
        --gpu-memory-utilization "${JUDGE_GPU_MEM_UTIL:-0.85}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN:-16384}" \
        > "$LOG_DIR/vllm_judge_${TIMESTAMP}.log" 2>&1 &
      wait_for_service "${JUDGE_PORT}" "Judge Model" || { tail -n 80 "$LOG_DIR/vllm_judge_${TIMESTAMP}.log" || true; exit 1; }
    else
      echo "NOTE: Judge api_base is localhost but JUDGE_MODEL is not a local path. Skipping judge server startup."
      echo "  JUDGE_MODEL=${JUDGE_MODEL:-<empty>}"
      echo "  JUDGE_API_BASE=${JUDGE_API_BASE:-<empty>}"
    fi
  fi
fi

REQUIRE_PREDICTIONS="${REQUIRE_PREDICTIONS:-true}"
REQUIRE_PRECOMPUTED="${REQUIRE_PRECOMPUTED:-false}"

echo "Evaluating..."
RUN_LOG="$LOG_DIR/mbef_eval_${TIMESTAMP}.log"
"$PY_BIN" tools/run.py "$CONFIG_PATH" \
  "${OVERRIDES[@]}" \
  --runner.stage eval \
  --runner.predictions_path "$PREDICTIONS_PATH" \
  --runner.require_predictions "$REQUIRE_PREDICTIONS" \
  --evaluator.use_precomputed_predictions true \
  --evaluator.require_precomputed_predictions "$REQUIRE_PRECOMPUTED" \
  --evaluator.prediction_field prediction \
  --model.type NoOpModel \
  > "$RUN_LOG" 2>&1

EXIT_CODE=$?
if [[ $EXIT_CODE -eq 0 ]]; then
  if [[ -n "${OUTPUT_DIR_REL:-}" ]]; then
    echo "Success. Results at: $MBEF_ROOT/$OUTPUT_DIR_REL"
    echo "Report: $MBEF_ROOT/$OUTPUT_DIR_REL/report.md"
  else
    echo "Success."
  fi
else
  echo "Failure (exit $EXIT_CODE). Check log: $RUN_LOG"
  tail -n 80 "$RUN_LOG" || true
fi
exit $EXIT_CODE

