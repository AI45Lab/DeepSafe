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




SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
cd "$MBEF_ROOT"

PY_BIN=$(which python3)
echo "Parsing config: $CONFIG_PATH"
eval "$("$PY_BIN" -m uni_eval.cli.parse_eval_config --config "$CONFIG_PATH" --mbef-root "$MBEF_ROOT" --format bash "${OVERRIDES[@]}")"

TARGET_IS_LOCAL=false
if [[ "$TARGET_API_BASE" == http://localhost:* || "$TARGET_API_BASE" == http://127.0.0.1:* ]]; then
  TARGET_IS_LOCAL=true
fi
JUDGE_IS_LOCAL=false
if [[ -n "$JUDGE_API_BASE" ]] && ([[ "$JUDGE_API_BASE" == http://localhost:* || "$JUDGE_API_BASE" == http://127.0.0.1:* ]]); then
  JUDGE_IS_LOCAL=true
fi

if [[ "$TARGET_IS_LOCAL" == true ]]; then
  TARGET_PORT="${TARGET_PORT:-21111}"
else
  TARGET_PORT=""
fi
if [[ "$JUDGE_IS_LOCAL" == true ]]; then
  JUDGE_PORT="${JUDGE_PORT:-}"
else
  JUDGE_PORT=""
fi

EXP_NAME="${EXP_NAME:-vlsbench_task}"
LOG_DIR="${LOG_DIR:-$MBEF_ROOT/scripts/rlaunch_logs/$EXP_NAME}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

echo "--------------------------------------------------------"
echo "Task:       $EXP_NAME"
if [[ "$TARGET_IS_LOCAL" == true ]]; then
  echo "Target:     $TARGET_MODEL (local port $TARGET_PORT)"
else
  echo "Target:     $TARGET_MODEL (api_base $TARGET_API_BASE)"
fi
if [[ -n "$JUDGE_MODEL" ]]; then
  if [[ -n "$JUDGE_API_BASE" && -z "$JUDGE_PORT" ]]; then
    echo "Judge:      $JUDGE_MODEL (api_base $JUDGE_API_BASE)"
  else
    echo "Judge:      $JUDGE_MODEL (port ${JUDGE_PORT:-<same as target>})"
  fi
else
  echo "Judge:      External API (e.g. GPT-4o)"
fi
echo "Logs dir:   $LOG_DIR"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides:  ${OVERRIDES[*]}"
fi
echo "--------------------------------------------------------"

JUDGE_HTTP_PROXY=""
for ((i=0; i<${#OVERRIDES[@]}; i++)); do
  if [[ "${OVERRIDES[$i]}" == "--evaluator.judge_model_cfg.http_proxy" ]]; then
    JUDGE_HTTP_PROXY="${OVERRIDES[$((i+1))]:-}"
    break
  fi
done
if [[ -n "$JUDGE_API_BASE" && "$JUDGE_IS_LOCAL" == false ]]; then
  echo "[preflight] Judge remote api_base detected: $JUDGE_API_BASE"
  if [[ -n "$JUDGE_HTTP_PROXY" ]]; then
    echo "[preflight] Using http_proxy for judge: $JUDGE_HTTP_PROXY"
  else
    echo "[preflight] No judge http_proxy override found (will rely on direct network/env)."
  fi
  JUDGE_MODELS_URL="${JUDGE_API_BASE%/}/models"
  if [[ -n "$JUDGE_HTTP_PROXY" ]]; then
    code="$(curl -sS --max-time 10 -x "$JUDGE_HTTP_PROXY" -o /dev/null -w "%{http_code}" "$JUDGE_MODELS_URL" || true)"
  else
    code="$(curl -sS --max-time 10 -o /dev/null -w "%{http_code}" "$JUDGE_MODELS_URL" || true)"
  fi
  echo "[preflight] GET $JUDGE_MODELS_URL -> HTTP $code (reachable if not 000)"
fi

cleanup() {
    echo "Cleaning up..."
    pkill -P $$ || true
    pkill -f "vllm.entrypoints.openai.api_server" || true
}
trap cleanup EXIT

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 5

wait_for_service() {
    local port=$1
    local name=$2
    local retries=300
    echo "Waiting for $name on port $port..."
    for i in $(seq 1 $retries); do
        status_code=$(curl --noproxy "*" -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/v1/models" || true)
        if [[ "$status_code" == "200" ]]; then
            echo "$name is ready!"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: $name failed to start within timeout."
    return 1
}

START_JUDGE_SERVER=false
if [[ "$TARGET_IS_LOCAL" == true && "$JUDGE_IS_LOCAL" == true && -n "$JUDGE_MODEL" && -n "$JUDGE_PORT" && "$JUDGE_PORT" != "$TARGET_PORT" ]]; then
  START_JUDGE_SERVER=true
fi

if [[ "$TARGET_IS_LOCAL" == true ]]; then
  if [[ "$START_JUDGE_SERVER" == true ]]; then
    TARGET_MEM=0.4
    JUDGE_MEM=0.4
  else
    TARGET_MEM=0.8
  fi
fi

if [[ "$TARGET_IS_LOCAL" == true ]]; then
  echo "Starting Target VLM with mem_util=$TARGET_MEM on port $TARGET_PORT..."
  CUDA_VISIBLE_DEVICES=0 nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
      --model "$TARGET_MODEL" \
      --served-model-name "$TARGET_MODEL" \
      --trust-remote-code \
      --port "$TARGET_PORT" \
      --gpu-memory-utilization "$TARGET_MEM" \
      --max-model-len 8192 \
      > "$LOG_DIR/vllm_target_$TIMESTAMP.log" 2>&1 &

  wait_for_service "$TARGET_PORT" "Target Model" || { tail -n 40 "$LOG_DIR/vllm_target_$TIMESTAMP.log" || true; exit 1; }
else
  echo "Target is remote API; no local vLLM started."
fi

if [[ "$START_JUDGE_SERVER" == true ]]; then
  echo "Starting Judge VLM with mem_util=$JUDGE_MEM on port $JUDGE_PORT..."
  CUDA_VISIBLE_DEVICES=0 nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
      --model "$JUDGE_MODEL" \
      --served-model-name "$JUDGE_MODEL" \
      --trust-remote-code \
      --port "$JUDGE_PORT" \
      --gpu-memory-utilization "$JUDGE_MEM" \
      --max-model-len 8192 \
      > "$LOG_DIR/vllm_judge_$TIMESTAMP.log" 2>&1 &

  wait_for_service "$JUDGE_PORT" "Judge Model" || { tail -n 40 "$LOG_DIR/vllm_judge_$TIMESTAMP.log" || true; exit 1; }
else
  echo "Judge uses same endpoint as Target or external API; no separate Judge vLLM started."
fi

echo "Running VLSBench evaluation via tools/run.py..."
RUN_LOG="$LOG_DIR/mbef_run_$TIMESTAMP.log"
"$PY_BIN" tools/run.py "$CONFIG_PATH" "${OVERRIDES[@]}" > "$RUN_LOG" 2>&1

EXIT_CODE=$?
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Success! Results saved."
    echo "Report: $MBEF_ROOT/$OUTPUT_DIR_REL/report.md"
else
    echo "Failure (exit $EXIT_CODE). Check log: $RUN_LOG"
    tail -n 40 "$RUN_LOG" || true
fi
exit $EXIT_CODE
