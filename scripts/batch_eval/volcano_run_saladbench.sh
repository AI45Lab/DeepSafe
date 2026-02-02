#!/bin/bash
set -eo pipefail

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"

CONFIG_PATH="${1:-configs/eval_tasks/salad_judge_v01_qwen1.5-0.5b_vllm_dual_gpu_volcano.yaml}"
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path> [overrides...]"
  exit 1
fi
shift || true

CONFIG_PATH="$(readlink -f "$CONFIG_PATH")"
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
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]] || [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  _TOTAL_GPUS=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    _TOTAL_GPUS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  fi
  if [[ "${_TOTAL_GPUS:-0}" -ge 2 ]]; then
    CUDA_VISIBLE_DEVICES="0,1"
  elif [[ "${_TOTAL_GPUS:-0}" -ge 1 ]]; then
    CUDA_VISIBLE_DEVICES="0"
  else
    CUDA_VISIBLE_DEVICES=""
  fi
fi
export CUDA_VISIBLE_DEVICES
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
DEFAULT_GPU_UTIL=0.6

count_visible_gpus () {
  local s="${1:-$CUDA_VISIBLE_DEVICES}"
  if [[ -z "$s" ]]; then
    echo 0
    return
  fi
  local -a parts=()
  IFS=',' read -r -a parts <<< "$s"
  local n=0
  for x in "${parts[@]}"; do
    if [[ -n "${x// /}" ]]; then
      n=$((n+1))
    fi
  done
  echo "$n"
}

TARGET_CUDA_VISIBLE_DEVICES="${TARGET_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
JUDGE_CUDA_VISIBLE_DEVICES="${JUDGE_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"

if [[ -z "${TARGET_CUDA_VISIBLE_DEVICES_SET:-}" ]] && [[ -z "${JUDGE_CUDA_VISIBLE_DEVICES_SET:-}" ]]; then
  if [[ -z "${TARGET_CUDA_VISIBLE_DEVICES:-}" ]] || [[ "$TARGET_CUDA_VISIBLE_DEVICES" == "$CUDA_VISIBLE_DEVICES" ]]; then
    if [[ "$CUDA_VISIBLE_DEVICES" == *","* ]]; then
      IFS=',' read -r _gpu0 _gpu1 _rest <<< "$CUDA_VISIBLE_DEVICES"
      if [[ -n "${_gpu0// /}" ]] && [[ -n "${_gpu1// /}" ]]; then
        TARGET_CUDA_VISIBLE_DEVICES="${_gpu0// /}"
        JUDGE_CUDA_VISIBLE_DEVICES="${_gpu1// /}"
      fi
    fi
  fi
fi

echo ">>> Parsing config: $CONFIG_PATH with overrides: $*"
eval "$("$PY_BIN" uni_eval/cli/parse_eval_config.py --config "$CONFIG_PATH" --format bash "$@")"

if [[ -z "${TARGET_MODEL:-}" ]]; then
  echo "ERROR: TARGET_MODEL could not be parsed from config."
  exit 1
fi

EXP_NAME="saladbench_service"
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
echo "Task:            $EXP_NAME"
echo "Config:          $CONFIG_PATH"
echo "CUDA(target):    $TARGET_CUDA_VISIBLE_DEVICES (n=$(count_visible_gpus "$TARGET_CUDA_VISIBLE_DEVICES"))"
echo "CUDA(judge):     $JUDGE_CUDA_VISIBLE_DEVICES (n=$(count_visible_gpus "$JUDGE_CUDA_VISIBLE_DEVICES"))"
echo "Target Model:    $TARGET_MODEL"
echo "Target API:      ${TARGET_API_BASE:-} (Port: ${TARGET_PORT:-})"
echo "Target TP:       ${TARGET_TENSOR_PARALLEL_SIZE:-}"
echo "Judge Model:     ${JUDGE_MODEL:-}"
echo "Judge API:       ${JUDGE_API_BASE:-} (Port: ${JUDGE_PORT:-})"
echo "Judge TP:        ${JUDGE_TENSOR_PARALLEL_SIZE:-}"
echo "Evaluator:       ${EVALUATOR_TYPE:-}"
echo "Template:        ${TEMPLATE_NAME:-}"
echo "Output(run):     $OUT_DIR_RUN"
echo "Output(latest):  $OUT_DIR_LATEST"
echo "Logs:            $LOG_DIR"
echo "--------------------------------------------------------"

cleanup() {
  echo "[cleanup] Stopping vLLM processes..."
  if [[ -n "${TARGET_PORT:-}" ]]; then
    pkill -f "vllm.entrypoints.openai.api_server.*--port ${TARGET_PORT}" 2>/dev/null || true
  fi
  if [[ -n "${JUDGE_PORT:-}" ]]; then
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
      tail -n 50 "$log_file" || true
      return 1
    fi
    sleep 1
  done
  echo "ERROR: $name failed to start within timeout."
  tail -n 50 "$log_file" || true
  return 1
}

NUM_VISIBLE_TARGET="$(count_visible_gpus "$TARGET_CUDA_VISIBLE_DEVICES")"
NUM_VISIBLE_JUDGE="$(count_visible_gpus "$JUDGE_CUDA_VISIBLE_DEVICES")"

TARGET_GPU_MEM_UTIL="${TARGET_GPU_MEM_UTIL:-$DEFAULT_GPU_UTIL}"
JUDGE_GPU_MEM_UTIL="${JUDGE_GPU_MEM_UTIL:-$DEFAULT_GPU_UTIL}"

TARGET_TP="${TARGET_TENSOR_PARALLEL_SIZE:-1}"
JUDGE_TP="${JUDGE_TENSOR_PARALLEL_SIZE:-1}"

if [[ -n "${TARGET_PORT:-}" ]]; then
  if [[ -d "$TARGET_MODEL" ]]; then
    true
  else
    echo "WARN: TARGET_MODEL is not a local directory: $TARGET_MODEL"
  fi

  if [[ "$NUM_VISIBLE_TARGET" -gt 0 ]] && [[ "$TARGET_TP" =~ ^[0-9]+$ ]] && [[ "$TARGET_TP" -gt "$NUM_VISIBLE_TARGET" ]]; then
    echo "ERROR: target tensor_parallel_size=$TARGET_TP > visible_gpus=$NUM_VISIBLE_TARGET (TARGET_CUDA_VISIBLE_DEVICES=$TARGET_CUDA_VISIBLE_DEVICES)"
    echo "Fix: set more GPUs in CUDA_VISIBLE_DEVICES, or override: --model.tensor_parallel_size <n>"
    exit 1
  fi

  echo "Starting Target vLLM ($TARGET_MODEL) on port $TARGET_PORT..."
  CUDA_VISIBLE_DEVICES="$TARGET_CUDA_VISIBLE_DEVICES" nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$TARGET_MODEL" \
    --served-model-name "$TARGET_MODEL" \
    --trust-remote-code \
    --port "$TARGET_PORT" \
    --gpu-memory-utilization "$TARGET_GPU_MEM_UTIL" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --tensor-parallel-size "$TARGET_TP" \
    > "$VLLM_TARGET_LOG" 2>&1 &

  wait_for_service "$TARGET_PORT" "Target vLLM" "$VLLM_TARGET_LOG" || exit 1
fi

if [[ -n "${JUDGE_PORT:-}" ]]; then
  if [[ -z "${JUDGE_MODEL:-}" ]]; then
    echo "ERROR: JUDGE_PORT is set but JUDGE_MODEL is empty."
    exit 1
  fi

  if [[ -d "$JUDGE_MODEL" ]]; then
    true
  else
    echo "WARN: JUDGE_MODEL is not a local directory: $JUDGE_MODEL"
  fi

  if [[ "$NUM_VISIBLE_JUDGE" -gt 0 ]] && [[ "$JUDGE_TP" =~ ^[0-9]+$ ]] && [[ "$JUDGE_TP" -gt "$NUM_VISIBLE_JUDGE" ]]; then
    echo "ERROR: judge tensor_parallel_size=$JUDGE_TP > visible_gpus=$NUM_VISIBLE_JUDGE (JUDGE_CUDA_VISIBLE_DEVICES=$JUDGE_CUDA_VISIBLE_DEVICES)"
    echo "Fix: set more GPUs in CUDA_VISIBLE_DEVICES, or override: --evaluator.judge_model_cfg.tensor_parallel_size <n>"
    exit 1
  fi

  echo "Starting Judge vLLM ($JUDGE_MODEL) on port $JUDGE_PORT..."
  CUDA_VISIBLE_DEVICES="$JUDGE_CUDA_VISIBLE_DEVICES" nohup "$PY_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" \
    --served-model-name "$JUDGE_MODEL" \
    --trust-remote-code \
    --port "$JUDGE_PORT" \
    --gpu-memory-utilization "$JUDGE_GPU_MEM_UTIL" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --tensor-parallel-size "$JUDGE_TP" \
    > "$VLLM_JUDGE_LOG" 2>&1 &

  wait_for_service "$JUDGE_PORT" "Judge vLLM" "$VLLM_JUDGE_LOG" || exit 1
fi

TWO_PHASE="${SALAD_TWO_PHASE:-0}"
if [[ "$TWO_PHASE" == "1" ]]; then
  echo ">>> Running Salad-Bench evaluation (two-phase, mutual exclusion: target then judge)..."

  echo ">>> Phase 1: Target Generation only..."
  "$PY_BIN" tools/run.py "$CONFIG_PATH" \
    --runner.output_dir="$OUT_DIR_RUN" \
    --evaluator.mode generate_only \
    "$@" > "$RUN_LOG" 2>&1
  EXIT_CODE=$?
  if [[ $EXIT_CODE -ne 0 ]]; then
    echo ">>> Failure in Phase 1! (Exit code $EXIT_CODE)"
    tail -n 120 "$RUN_LOG" || true
  else
    echo ">>> Phase 1 success."
  fi

  if [[ -n "${TARGET_PORT:-}" ]]; then
    echo ">>> Stopping Target vLLM before Phase 2..."
    pkill -f "vllm.entrypoints.openai.api_server.*--port ${TARGET_PORT}" 2>/dev/null || true
    sleep 2
  fi

  if [[ $EXIT_CODE -eq 0 ]]; then
    PRECOMP_PATH="$OUT_DIR_RUN/precomputed_predictions.json"
    echo ">>> Building precomputed dataset: $PRECOMP_PATH"
    OUT_DIR_RUN="$OUT_DIR_RUN" "$PY_BIN" - <<'PY'
import json, os, sys
out_dir = os.environ.get("OUT_DIR_RUN", "")
if not out_dir:
    print("OUT_DIR_RUN missing", file=sys.stderr)
    sys.exit(2)
src = os.path.join(out_dir, "result.json")
dst = os.path.join(out_dir, "precomputed_predictions.json")
with open(src, "r", encoding="utf-8") as f:
    res = json.load(f)
details = res.get("details") or []
out = []
for item in details:
    if not isinstance(item, dict):
        continue
    prompt = item.get("prompt")
    pred = item.get("prediction")
    if isinstance(prompt, str) and prompt.strip():
        out.append({"id": item.get("id"), "prompt": prompt, "prediction": pred, "meta": item.get("meta")})
with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"Wrote {len(out)} items to {dst}")
PY

    echo ">>> Phase 2: Judge Evaluation only..."
    PHASE2_LOG="$LOG_DIR/run_${EXP_NAME}_phase2_$TIMESTAMP.log"
    PHASE2_ARGS=()
    if [[ "$#" -gt 0 ]]; then
      args=("$@")
      i=0
      while [[ $i -lt ${#args[@]} ]]; do
        a="${args[$i]}"
        if [[ "$a" == --evaluator.judge_model_cfg.* ]]; then
          PHASE2_ARGS+=("$a")
          if [[ $((i+1)) -lt ${#args[@]} ]]; then
            b="${args[$((i+1))]}"
            if [[ "$b" != --* ]]; then
              PHASE2_ARGS+=("$b")
              i=$((i+1))
            fi
          fi
        elif [[ "$a" == --evaluator.batch_size ]]; then
          PHASE2_ARGS+=("$a")
          if [[ $((i+1)) -lt ${#args[@]} ]]; then
            PHASE2_ARGS+=("${args[$((i+1))]}")
            i=$((i+1))
          fi
        fi
        i=$((i+1))
      done
    fi

    OUT_DIR_RUN="$OUT_DIR_RUN" "$PY_BIN" tools/run.py "$CONFIG_PATH" \
      --runner.output_dir="$OUT_DIR_RUN" \
      --model.type NoOpModel \
      --dataset.type PrecomputedDataset \
      --dataset.path "$PRECOMP_PATH" \
      --evaluator.mode judge_only \
      --evaluator.use_precomputed_predictions true \
      --evaluator.require_precomputed_predictions true \
      "${PHASE2_ARGS[@]}" > "$PHASE2_LOG" 2>&1
    EXIT_CODE=$?
    if [[ $EXIT_CODE -ne 0 ]]; then
      echo ">>> Failure in Phase 2! (Exit code $EXIT_CODE)"
      tail -n 120 "$PHASE2_LOG" || true
    else
      echo ">>> Phase 2 success! Check logs at $PHASE2_LOG"
    fi
  fi
else
  echo ">>> Running Salad-Bench evaluation..."
  "$PY_BIN" tools/run.py "$CONFIG_PATH" --runner.output_dir="$OUT_DIR_RUN" "$@" > "$RUN_LOG" 2>&1
  EXIT_CODE=$?
fi

if [[ $EXIT_CODE -eq 0 ]]; then
  echo ">>> Success! Check logs at $RUN_LOG"
else
  echo ">>> Failure! (Exit code $EXIT_CODE)"
  tail -n 80 "$RUN_LOG" || true
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

