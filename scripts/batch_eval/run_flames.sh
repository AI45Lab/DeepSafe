#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -euo pipefail

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
SCRIPT_PATH="$MBEF_ROOT/scripts/run_flames.sh"

usage() {
  echo "Usage:"
  echo "  bash $SCRIPT_PATH /abs/path/to/config.yaml"
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 2
fi

if [[ -f "/mnt/shared-storage-user/zhangbo1/miniconda3/etc/profile.d/conda.sh" ]]; then
  
  conda activate "${CONDA_ENV:-vllm_10_2}"
fi

PY_BIN="python3"
command -v "$PY_BIN" >/dev/null 2>&1 || PY_BIN="python"
command -v "$PY_BIN" >/dev/null 2>&1 || { echo "ERROR: python3/python not found" >&2; exit 1; }

export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
cd "$MBEF_ROOT"

MODE="submit"
if [[ "${1:-}" == "--worker" ]]; then
  MODE="run"
  shift
fi

CONFIG_PATH="$(readlink -f "$1")"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

eval "$("$PY_BIN" -m uni_eval.cli.parse_eval_config \
  --config "$CONFIG_PATH" \
  --format bash \
  --strict \
  --mbef-root "$MBEF_ROOT")"

if [[ "$MODE" == "submit" ]]; then
  command -v rlaunch >/dev/null 2>&1 || { echo "ERROR: rlaunch not found." >&2; exit 1; }

  echo "--------------------------------------------------------"
  echo "Submitting Flames Task:"
  echo "  Config:   $CONFIG_PATH"
  echo "  ExpName:  $EXP_NAME"
  echo "  Logs:     $LOG_DIR"
  echo "  Target:   $TARGET_MODEL (port $TARGET_PORT)"
  echo "  Evaluator:$EVALUATOR_TYPE (HAS_SCORER=$HAS_SCORER)"
  echo "--------------------------------------------------------"

  ONE_GPU="${ONE_GPU:-1}"
  if [[ "$ONE_GPU" == "1" ]]; then
    export RLAUNCH_GPU="${RLAUNCH_GPU:-1}"
  fi

  rlaunch --charged-group="${RLAUNCH_GROUP:-ai4good2_gpu}" \
    --private-machine=yes \
    --image="${RLAUNCH_IMAGE:-registry.h.pjlab.org.cn/ailab-trustworthyaitest-aitest_gpu/test_gpu:gptv0}" \
    --max-wait-duration "${RLAUNCH_MAX_WAIT:-3600s}" \
    --enable-sshd=false \
    --private-machine=group \
    --gpu "${RLAUNCH_GPU:-2}" \
    --memory "${RLAUNCH_MEMORY:-32768}" \
    --cpu "${RLAUNCH_CPU:-8}" \
    --entrypoint='' \
    --mount="${RLAUNCH_MOUNT:-gpfs://gpfs1/zhangbo1:/mnt/shared-storage-user/zhangbo1}" \
    -- bash "$SCRIPT_PATH" --worker "$CONFIG_PATH"

  exit $?
fi

mkdir -p "$LOG_DIR"
TS="$(date +"%Y%m%d%H%M%S")"
RUN_LOG="$LOG_DIR/mbef_run_${TS}.log"
TARGET_LOG="$LOG_DIR/vllm_target_${TS}.log"

TARGET_GPU_MEM_UTIL="${TARGET_GPU_MEM_UTIL:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"

cleanup() {
  echo "Cleaning up..."
  pkill -P $$ || true
  pkill -f "vllm.entrypoints.openai.api_server.*--port $TARGET_PORT" || true
}
trap cleanup EXIT

wait_port() {
  local port="$1"
  local name="$2"
  local log_file="$3"
  echo "Waiting for $name on port $port..."
  for _ in $(seq 1 60); do
    if curl --noproxy "*" -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/v1/models" | grep -q "200"; then
      echo "$name is ready!"
      return 0
    fi
    sleep 5
  done
  echo "ERROR: $name failed to start (port $port)."
  tail -n 50 "$log_file" || true
  return 1
}

pkill -f "vllm.entrypoints.openai.api_server.*--port $TARGET_PORT" || true
sleep 2

echo "Starting Target Model ($TARGET_MODEL) on GPU0..."
CUDA_VISIBLE_DEVICES="${TARGET_CUDA_VISIBLE_DEVICES:-0}" nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$TARGET_MODEL" \
  --served-model-name "$TARGET_MODEL" \
  --trust-remote-code \
  --port "$TARGET_PORT" \
  --gpu-memory-utilization "$TARGET_GPU_MEM_UTIL" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  > "$TARGET_LOG" 2>&1 &

wait_port "$TARGET_PORT" "Target Model" "$TARGET_LOG" || exit 1

ONE_GPU="${ONE_GPU:-1}"

echo "Running 2-stage Flames pipeline:"
echo "  Stage-1 (vLLM): generate target responses -> JSONL"
echo "  Stage-2 (Transformers): Flames scorer inference -> metrics/report"

  STAGE_DIR="$RESULT_DIR/_stage1"
  mkdir -p "$STAGE_DIR"
  RESP_JSONL="$STAGE_DIR/flames_target_responses.jsonl"
  CFG_STAGE1="$LOG_DIR/_cfg_stage1_${TS}.yaml"
  CFG_STAGE2="$LOG_DIR/_cfg_stage2_${TS}.yaml"
STAGE1_LOG="$LOG_DIR/mbef_stage1_${TS}.log"

TARGET_CUDA_VISIBLE_DEVICES="${TARGET_CUDA_VISIBLE_DEVICES:-0}"
SCORER_CUDA_VISIBLE_DEVICES="${SCORER_CUDA_VISIBLE_DEVICES:-1}"
if [[ "${ONE_GPU:-0}" == "1" ]]; then
  echo "ONE_GPU=1: running sequentially on a single GPU."
  SCORER_CUDA_VISIBLE_DEVICES="$TARGET_CUDA_VISIBLE_DEVICES"
fi

  echo "Generating stage-1 config (target responses)..."
"$PY_BIN" tools/flames_make_stage_config.py \
  --config "$CONFIG_PATH" \
  --stage 1 \
  --out "$CFG_STAGE1" \
  --resp-jsonl "$RESP_JSONL" \
  --stage-dir "$STAGE_DIR" \
  > "$LOG_DIR/flames_make_stage1_${TS}.log" 2>&1

echo "Stage-1: run target -> $RESP_JSONL"
CUDA_VISIBLE_DEVICES="$TARGET_CUDA_VISIBLE_DEVICES" "$PY_BIN" tools/run.py "$CFG_STAGE1" > "$STAGE1_LOG" 2>&1

echo "Stopping vLLM target (port $TARGET_PORT)..."
  pkill -f "vllm.entrypoints.openai.api_server.*--port $TARGET_PORT" || true
  sleep 5

  echo "Generating stage-2 config (scorer-only)..."
"$PY_BIN" tools/flames_make_stage_config.py \
  --config "$CONFIG_PATH" \
  --stage 2 \
  --out "$CFG_STAGE2" \
  --resp-jsonl "$RESP_JSONL" \
  --result-dir "$RESULT_DIR" \
  > "$LOG_DIR/flames_make_stage2_${TS}.log" 2>&1

echo "Stage-2: run scorer -> final metrics"
CUDA_VISIBLE_DEVICES="$SCORER_CUDA_VISIBLE_DEVICES" "$PY_BIN" tools/run.py "$CFG_STAGE2" > "$RUN_LOG" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "Success. Results at: $RESULT_DIR"
else
  echo "Failure (exit $EXIT_CODE). See log: $RUN_LOG"
  tail -n 50 "$RUN_LOG" || true
fi
exit $EXIT_CODE

