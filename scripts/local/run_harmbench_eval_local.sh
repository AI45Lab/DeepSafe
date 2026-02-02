#!/bin/bash
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
cd "$MBEF_ROOT"

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"

PY_BIN="$(which python3)"
echo "Parsing config: $CONFIG_PATH"

PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config \
    --config "$CONFIG_PATH" \
    --mbef-root "$MBEF_ROOT" \
    --format bash \
    "${OVERRIDES[@]}")" \
  || { echo "ERROR: failed to parse config"; exit 1; }
eval "$PARSED_ENV"

EXP_NAME="${EXP_NAME:-harmbench_task}"
if [[ -n "$MBEF_LOG_DIR" ]]; then
  LOG_DIR="$MBEF_LOG_DIR"
else
  LOG_DIR="$MBEF_ROOT/scripts/local_logs/$EXP_NAME"
fi
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

if [[ -n "$OUTPUT_DIR_REL" ]]; then
  PREDICTIONS_PATH_DEFAULT="$MBEF_ROOT/$OUTPUT_DIR_REL/predictions.jsonl"
else
  PREDICTIONS_PATH_DEFAULT="$MBEF_ROOT/results/$EXP_NAME/predictions.jsonl"
fi
PREDICTIONS_PATH="${PREDICTIONS_PATH:-$PREDICTIONS_PATH_DEFAULT}"

echo "--------------------------------------------------------"
echo "Stage:  EVAL"
echo "Task:   $EXP_NAME"
echo "Judge:  $JUDGE_MODEL (Port ${JUDGE_PORT:-21111})"
echo "--------------------------------------------------------"

cleanup() {
    echo "Cleaning up..."
    pkill -P $$ || true
    if [[ -n "$JUDGE_PORT" ]]; then
        pkill -f "vllm.entrypoints.openai.api_server.*--port $JUDGE_PORT" || true
    fi
}
trap cleanup EXIT

PORT="${JUDGE_PORT:-21111}"
if [[ "$JUDGE_API_BASE" == http://localhost:* || "$JUDGE_API_BASE" == http://127.0.0.1:* ]]; then
    echo "Starting Judge Model $JUDGE_MODEL (local vLLM)..." 
    JUDGE_TP="${JUDGE_TENSOR_PARALLEL_SIZE:-1}"
    JUDGE_GPU_UTIL="${JUDGE_GPU_MEM_UTIL:-0.8}"
    
    nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
      --model "$JUDGE_MODEL" \
      --served-model-name "$JUDGE_MODEL" \
      --trust-remote-code \
      --port "${PORT}" \
      --tensor-parallel-size "${JUDGE_TP}" \
      --gpu-memory-utilization "${JUDGE_GPU_UTIL}" \
      --max-model-len "${VLLM_MAX_MODEL_LEN:-2048}" \
      --chat-template "$MBEF_ROOT/data/harmbench/chat_template.jinja" \
      > "$LOG_DIR/vllm_judge_$TIMESTAMP.log" 2>&1 &

    echo "Waiting for Judge Model on port $PORT..."
    READY=0
    for i in $(seq 1 600); do
        if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" | grep -q "200"; then
            echo "Judge Model is ready!"
            READY=1
            break
        fi
        sleep 5
    done
    if [[ $READY -eq 0 ]]; then
        echo "ERROR: Judge Model failed to start."
        tail -n 60 "$LOG_DIR/vllm_judge_$TIMESTAMP.log"
        exit 1
    fi
else
  echo "Judge api_base is remote ($JUDGE_API_BASE). Skip starting local vLLM server."
fi

echo "Running Evaluation..."
RUN_LOG="$LOG_DIR/mbef_eval_$TIMESTAMP.log"

"$PY_BIN" tools/run.py "$CONFIG_PATH" \
  --runner.stage eval \
  --runner.predictions_path "$PREDICTIONS_PATH" \
  --model.type NoOpModel \
  "${OVERRIDES[@]}" \
  2>&1 | tee "$RUN_LOG"

EXIT_CODE=${PIPESTATUS[0]}
exit $EXIT_CODE
