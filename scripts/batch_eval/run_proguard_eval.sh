#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -euo pipefail

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

if [[ "${1:-}" == "submit" ]]; then
  shift
  CFG_SUB="${1:-}"
  OUT_SUB="${2:-}"
  PROGUARD_MODEL_PATH="${3:-/mnt/shared-storage-user/ai4good2-share/models/ProGuard-7B}"
  SERVED_NAME="${4:-ProGuard-7B}"
  PORT="${5:-21122}"

  if [[ -z "${CFG_SUB}" || -z "${OUT_SUB}" ]]; then
    echo "Usage: $0 submit <config_yaml> <output_dir> [proguard_model_path] [served_model_name] [port]" >&2
    exit 2
  fi

  if [[ "$CFG_SUB" != /* ]]; then
    CFG_SUB="$MBEF_ROOT/$CFG_SUB"
  fi
  CFG_SUB="$(readlink -f "$CFG_SUB")"
  if [[ ! -f "$CFG_SUB" ]]; then
    echo "ERROR: config not found: $CFG_SUB" >&2
    exit 1
  fi

  if [[ "$OUT_SUB" == /* ]]; then
    OUT_REL_SUB="${OUT_SUB#${MBEF_ROOT}/}"
  else
    OUT_REL_SUB="$OUT_SUB"
  fi

  TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
  LOG_DIR="$MBEF_ROOT/logs/proguard_eval"
  mkdir -p "$LOG_DIR"
  LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"

  SUBMIT_PROFILE="${SUBMIT_PROFILE:-gpu}"
  if [[ "$SUBMIT_PROFILE" != "gpu" ]]; then
    echo "ERROR: ProGuard vLLM requires GPU; please use SUBMIT_PROFILE=gpu" >&2
    exit 1
  fi

  echo "Submitting ProGuard eval via rjob:"
  echo "  Config:   $CFG_SUB"
  echo "  Output:   $OUT_REL_SUB"
  echo "  ProGuard: $PROGUARD_MODEL_PATH (served_name=$SERVED_NAME, port=$PORT)"
  echo "  Log:      $LOG_FILE"

  rjob submit --charged-group="ai4good2_gpu" \
    --private-machine=group \
    --image=registry.h.pjlab.org.cn/ailab-ai4good2-ai4good2_gpu/zhangbo1:v0130-x86 \
    --gpu 1 \
    --memory 32768 \
    --cpu 16 \
    --mount=gpfs://gpfs1/zhangbo1:/mnt/shared-storage-user/zhangbo1 \
    --mount=gpfs://gpfs1/ai4good2-share:/mnt/shared-storage-user/ai4good2-share \
    -- bash -lc "set -euo pipefail; bash '$MBEF_ROOT/scripts/run_proguard_eval_with_vllm.sh' '$CFG_SUB' '$OUT_REL_SUB' '$PROGUARD_MODEL_PATH' '$SERVED_NAME' '$PORT' 2>&1 | tee '$LOG_FILE'"

  exit $?
fi

CFG="${1:-}"
OUT="${2:-}"
API_BASE="${3:-http://127.0.0.1:21122/v1}"
MODEL_NAME="${4:-ProGuard-7B}"

if [[ -z "${CFG}" || -z "${OUT}" ]]; then
  echo "Usage: $0 <config_yaml> <output_dir> [proguard_api_base] [proguard_model_name]" >&2
  exit 2
fi

if [[ "$CFG" != /* ]]; then
  CFG="$MBEF_ROOT/$CFG"
fi
CFG="$(readlink -f "$CFG")"
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG" >&2
  exit 1
fi

if [[ "$OUT" == /* ]]; then
  OUT_REL="${OUT#${MBEF_ROOT}/}"
else
  OUT_REL="$OUT"
fi

PRED="$MBEF_ROOT/$OUT_REL/predictions.jsonl"
if [[ ! -f "$PRED" ]]; then
  echo "ERROR: predictions.jsonl not found: $PRED" >&2
  exit 1
fi

cd "$MBEF_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"

echo "--------------------------------------------------------"
echo "ProGuard eval"
echo "Config:   $CFG"
echo "Output:   $OUT_REL"
echo "Pred:     $PRED"
echo "ProGuard: $MODEL_NAME @ $API_BASE"
echo "--------------------------------------------------------"

RUNNER_REQUIRE_PREDICTIONS="${RUNNER_REQUIRE_PREDICTIONS:-true}"
EVALUATOR_REQUIRE_PRECOMPUTED="${EVALUATOR_REQUIRE_PRECOMPUTED:-false}"

python3 tools/run.py "$CFG" \
  --runner.stage eval \
  --runner.output_dir "$OUT_REL" \
  --runner.predictions_path "$PRED" \
  --runner.require_predictions "$RUNNER_REQUIRE_PREDICTIONS" \
  --model.type NoOpModel \
  --evaluator.use_precomputed_predictions true \
  --evaluator.require_precomputed_predictions "$EVALUATOR_REQUIRE_PRECOMPUTED" \
  --evaluator.judge_model_cfg.api_base "$API_BASE" \
  --evaluator.judge_model_cfg.model_name "$MODEL_NAME"

