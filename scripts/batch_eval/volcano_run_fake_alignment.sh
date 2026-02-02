#!/bin/bash
set -eo pipefail

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"

CONFIG_PATH="${1:-configs/eval_tasks/fake_alignment_qwen.yaml}"
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path> [overrides...]"
  exit 1
fi
shift || true

CONFIG_PATH=$(readlink -f "$CONFIG_PATH")
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH"
  exit 1
fi

MBFF_ROOT="${MBFF_ROOT:-/root/zhangbo1/MBFF-1222}"
if [[ ! -d "$MBFF_ROOT" ]]; then
  echo "ERROR: MBFF_ROOT does not exist: $MBFF_ROOT"
  exit 1
fi
export PYTHONPATH="${PYTHONPATH:-}:$MBFF_ROOT"
cd "$MBFF_ROOT"

PY_BIN="${PY_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
DEFAULT_GPU_UTIL=0.4

echo ">>> Parsing config: $CONFIG_PATH with overrides: $*"
eval "$("$PY_BIN" uni_eval/cli/parse_eval_config.py --config "$CONFIG_PATH" --format bash "$@")"

if [[ -z "$TARGET_MODEL" ]]; then
    echo "ERROR: TARGET_MODEL could not be parsed from config."
    exit 1
fi

EXP_NAME="fake_alignment_service"
LOG_DIR="$MBFF_ROOT/scripts/local_logs/$EXP_NAME"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +'%Y%m%d%H%M%S')"

OUT_DIR_RUN="$MBFF_ROOT/results/$EXP_NAME/run_$TIMESTAMP"
OUT_DIR_LATEST="$MBFF_ROOT/results/$EXP_NAME/latest"
mkdir -p "$OUT_DIR_RUN" "$OUT_DIR_LATEST"

RUN_LOG="$LOG_DIR/run_${EXP_NAME}_$TIMESTAMP.log"
VLLM_TARGET_LOG="$LOG_DIR/vllm_target_${EXP_NAME}_$TIMESTAMP.log"
VLLM_JUDGE_LOG="$LOG_DIR/vllm_judge_${EXP_NAME}_$TIMESTAMP.log"

echo "--------------------------------------------------------"
echo "Task:        $EXP_NAME"
echo "Target Model: $TARGET_MODEL"
echo "Target API:   $TARGET_API_BASE (Port: $TARGET_PORT)"
if [[ -n "$JUDGE_MODEL" ]]; then
    echo "Judge Model:  $JUDGE_MODEL"
    echo "Judge API:    $JUDGE_API_BASE (Port: $JUDGE_PORT)"
fi
echo "CUDA:        $CUDA_VISIBLE_DEVICES"
echo "Logs:        $LOG_DIR"
echo "--------------------------------------------------------"

cleanup() {
  echo "[cleanup] Stopping vLLM processes..."
  if [[ -n "$TARGET_PORT" ]]; then
      pkill -f "vllm.entrypoints.openai.api_server.*--port ${TARGET_PORT}" 2>/dev/null || true
  fi
  if [[ -n "$JUDGE_PORT" ]]; then
      pkill -f "vllm.entrypoints.openai.api_server.*--port ${JUDGE_PORT}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cleanup
sleep 1

wait_for_service() {
  local port="$1"
  local name="$2"
  local log_file="$3"
  local retries=300
  echo "Waiting for $name on port $port..."
  for _ in $(seq 1 "$retries"); do
    local code
    code="$(curl --noproxy "*" -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/v1/models" || true)"
    if [[ "$code" == "200" ]]; then
      echo "$name is ready!"
      return 0
    fi
    if ! pgrep -f "vllm.entrypoints.openai.api_server.*--port ${port}" > /dev/null; then
        echo "ERROR: $name process died unexpectedly."
        tail -n 20 "$log_file"
        return 1
    fi
    sleep 1
  done
  echo "ERROR: $name failed to start within timeout."
  tail -n 20 "$log_file"
  return 1
}

if [[ -n "$JUDGE_MODEL" ]] && [[ -n "$JUDGE_PORT" ]]; then
    TARGET_GPU_MEM_UTIL="${TARGET_GPU_MEM_UTIL:-0.45}"
    JUDGE_GPU_MEM_UTIL="${JUDGE_GPU_MEM_UTIL:-0.45}"
else
    TARGET_GPU_MEM_UTIL="${TARGET_GPU_MEM_UTIL:-$DEFAULT_GPU_UTIL}"
fi

if [[ -n "$TARGET_PORT" ]]; then
    echo "Starting Target vLLM ($TARGET_MODEL) on port $TARGET_PORT..."
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
      --model "$TARGET_MODEL" \
      --served-model-name "$TARGET_MODEL" \
      --trust-remote-code \
      --port "$TARGET_PORT" \
      --gpu-memory-utilization "$TARGET_GPU_MEM_UTIL" \
      --max-model-len "$VLLM_MAX_MODEL_LEN" \
      > "$VLLM_TARGET_LOG" 2>&1 &
    wait_for_service "$TARGET_PORT" "Target vLLM" "$VLLM_TARGET_LOG" || exit 1
fi

if [[ -n "$JUDGE_MODEL" ]] && [[ -n "$JUDGE_PORT" ]]; then
    if [[ "$TARGET_PORT" != "$JUDGE_PORT" ]]; then
        echo "Starting Judge vLLM ($JUDGE_MODEL) on port $JUDGE_PORT..."
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
          --model "$JUDGE_MODEL" \
          --served-model-name "$JUDGE_MODEL" \
          --trust-remote-code \
          --port "$JUDGE_PORT" \
          --gpu-memory-utilization "$JUDGE_GPU_MEM_UTIL" \
          --max-model-len "$VLLM_MAX_MODEL_LEN" \
          > "$VLLM_JUDGE_LOG" 2>&1 &
        wait_for_service "$JUDGE_PORT" "Judge vLLM" "$VLLM_JUDGE_LOG" || exit 1
    fi
fi

echo ">>> Running evaluation..."
"$PY_BIN" tools/run.py "$CONFIG_PATH" --runner.output_dir="$OUT_DIR_RUN" "$@" > "$RUN_LOG" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ">>> Success! Check logs at $RUN_LOG"
else
    echo ">>> Failure! (Exit code $EXIT_CODE)"
    tail -n 50 "$RUN_LOG"
fi

RESULT_RUN="$OUT_DIR_RUN/result.json"
REPORT_RUN="$OUT_DIR_RUN/report.md"
RESULT_LATEST="$OUT_DIR_LATEST/result.json"
REPORT_LATEST="$OUT_DIR_LATEST/report.md"

if [[ -f "$RESULT_RUN" ]]; then
  cp -f "$RESULT_RUN" "$RESULT_LATEST"
fi
if [[ -f "$REPORT_RUN" ]]; then
  cp -f "$REPORT_RUN" "$REPORT_LATEST"
fi

echo "RESULT_PATH=$RESULT_LATEST"
echo "REPORT_PATH=$REPORT_LATEST"
exit $EXIT_CODE
