#!/bin/bash
set -euo pipefail

MBFF_ROOT="/root/zhangbo1/MBFF-1222"
MODEL_PATH="${MODEL_PATH:-/root/zhangbo1/models/Qwen2.5-VL-3B-Instruct}"

VLSBENCH_DATASET_PATH="${VLSBENCH_DATASET_PATH:-/path/to/vlsbench/data.json}"

VLLM_PORT="${VLLM_PORT:-21113}"
API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

PY_BIN="${PY_BIN:-python3}"

TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"

if [[ ! -d "$MBFF_ROOT" ]]; then
  echo "ERROR: MBFF_ROOT not found: $MBFF_ROOT"
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH not found: $MODEL_PATH"
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}:$MBFF_ROOT"
cd "$MBFF_ROOT"

EXP_NAME="vlsbench_service"
LOG_DIR="$MBFF_ROOT/scripts/local_logs/$EXP_NAME"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +'%Y%m%d%H%M%S')"

OUT_DIR_RUN="$MBFF_ROOT/results/$EXP_NAME/run_$TIMESTAMP"
OUT_DIR_LATEST="$MBFF_ROOT/results/$EXP_NAME/latest"
mkdir -p "$OUT_DIR_RUN" "$OUT_DIR_LATEST"

CONFIG_PATH="$LOG_DIR/_cfg_${EXP_NAME}_$TIMESTAMP.yaml"
RUN_LOG="$LOG_DIR/run_${EXP_NAME}_$TIMESTAMP.log"
VLLM_LOG="$LOG_DIR/vllm_${EXP_NAME}_$TIMESTAMP.log"

cleanup() {
  echo "[cleanup] stopping vLLM (port=$VLLM_PORT) ..."
  pkill -f "vllm.entrypoints.openai.api_server.*--port ${VLLM_PORT}" 2>/dev/null || true
}
trap cleanup EXIT

pkill -f "vllm.entrypoints.openai.api_server.*--port ${VLLM_PORT}" 2>/dev/null || true
sleep 1

cat > "$CONFIG_PATH" <<EOF
task_name: ${EXP_NAME}
output_dir: ${OUT_DIR_RUN}

dataset:
  type: VLSBenchDataset
  path: ${VLSBENCH_DATASET_PATH}

model:
  type: APIModel
  model_name: ${MODEL_PATH}
  api_base: ${API_BASE}
  api_key: EMPTY
  mode: chat
  temperature: ${TEMPERATURE}
  top_p: ${TOP_P}
  top_k: ${TOP_K}
  max_tokens: 512
  repetition_penalty: ${REPETITION_PENALTY}

evaluator:
  type: VLSBenchEvaluator
  batch_size: 1
  template_name: vlsbench
  judge_model_cfg:
    type: APIModel
    model_name: ${MODEL_PATH}
    api_base: ${API_BASE}
    api_key: EMPTY
    mode: chat
    temperature: ${TEMPERATURE}
    top_p: ${TOP_P}
    top_k: ${TOP_K}
    max_tokens: 256

metrics:
  - type: VLSBenchMetric

summarizer:
  type: StandardSummarizer

runner:
  type: LocalRunner
  output_dir: ${OUT_DIR_RUN}
EOF

wait_for_service() {
  local port="$1"
  local name="$2"
  local retries=300
  echo "Waiting for $name on port $port..."
  for _ in $(seq 1 "$retries"); do
    local code
    code="$(curl --noproxy "*" -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/v1/models" || true)"
    if [[ "$code" == "200" ]]; then
      echo "$name is ready!"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: $name failed to start within timeout."
  return 1
}

echo "--------------------------------------------------------"
echo "MBFF_ROOT:   $MBFF_ROOT"
echo "Model:       $MODEL_PATH"
echo "Dataset:     $VLSBENCH_DATASET_PATH"
echo "API Base:    $API_BASE"
echo "CUDA:        $CUDA_VISIBLE_DEVICES"
echo "Logs:        $LOG_DIR"
echo "Config:      $CONFIG_PATH"
echo "Output(run): $OUT_DIR_RUN"
echo "Output(lat): $OUT_DIR_LATEST"
echo "--------------------------------------------------------"

echo "Starting vLLM server on port $VLLM_PORT..."
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_PATH" \
  --trust-remote-code \
  --port "$VLLM_PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  > "$VLLM_LOG" 2>&1 &

wait_for_service "$VLLM_PORT" "vLLM" || { tail -n 80 "$VLLM_LOG" || true; exit 1; }

echo "Running evaluation..."
"$PY_BIN" tools/run.py "$CONFIG_PATH" > "$RUN_LOG" 2>&1

REPORT_RUN="$OUT_DIR_RUN/report.md"
REPORT_LATEST="$OUT_DIR_LATEST/report.md"

if [[ ! -f "$REPORT_RUN" ]]; then
  echo "ERROR: report not found at $REPORT_RUN"
  echo "Last 80 lines of run log:"
  tail -n 80 "$RUN_LOG" || true
  exit 1
fi

cp -f "$REPORT_RUN" "$REPORT_LATEST"
echo "REPORT_PATH=$REPORT_LATEST"
echo "Done."

