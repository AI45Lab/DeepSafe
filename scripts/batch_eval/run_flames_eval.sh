#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -euo pipefail

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"




CONFIG_PATH="${1:-}"
shift || true
OVERRIDES=("$@")
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path> [overrides...]" >&2
  exit 2
fi

CONFIG_PATH="$(readlink -f "$CONFIG_PATH")"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

PY_BIN="python3"
command -v "$PY_BIN" >/dev/null 2>&1 || PY_BIN="python"
command -v "$PY_BIN" >/dev/null 2>&1 || { echo "ERROR: python3/python not found" >&2; exit 1; }

export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH
cd "$MBEF_ROOT"

eval "$("$PY_BIN" -m uni_eval.cli.parse_eval_config \
  --config "$CONFIG_PATH" \
  --format bash \
  --mbef-root "$MBEF_ROOT" \
  "${OVERRIDES[@]}")"

mkdir -p "$LOG_DIR"
TS="$(date +"%Y%m%d%H%M%S")"
RUN_LOG="$LOG_DIR/mbef_stage2_${TS}.log"

STAGE_DIR="$RESULT_DIR/_stage1"
RESP_JSONL="${FLAMES_RESP_JSONL:-$STAGE_DIR/flames_target_responses.jsonl}"
CFG_STAGE2="$LOG_DIR/_cfg_stage2_${TS}.yaml"

if [[ ! -f "$RESP_JSONL" ]]; then
  FALLBACK_JSONL="$RESULT_DIR/flames_target_responses.jsonl"
  if [[ -f "$FALLBACK_JSONL" ]]; then
    echo "[warn] Flames responses JSONL not found at stage dir: $RESP_JSONL"
    echo "[warn] Found fallback at: $FALLBACK_JSONL"
    echo "[warn] Proceeding with fallback. (Recommended: keep stage-1 outputs under $STAGE_DIR/)"
    RESP_JSONL="$FALLBACK_JSONL"
  else
    echo "ERROR: Flames responses JSONL not found: $RESP_JSONL" >&2
    echo "Hint: run gen stage first, or set FLAMES_RESP_JSONL=/path/to/jsonl" >&2
    exit 1
  fi
fi

ONE_GPU="${ONE_GPU:-0}"
SCORER_CUDA_VISIBLE_DEVICES="${SCORER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
if [[ "$ONE_GPU" == "1" ]]; then
  SCORER_CUDA_VISIBLE_DEVICES="${TARGET_CUDA_VISIBLE_DEVICES:-0}"
fi

echo "--------------------------------------------------------"
echo "Stage:   eval"
echo "Config:   $CONFIG_PATH"
echo "ExpName:  $EXP_NAME"
echo "Result:   $RESULT_DIR"
echo "RespJSONL:$RESP_JSONL"
echo "ScorerCUDA: $SCORER_CUDA_VISIBLE_DEVICES"
echo "Logs:     $LOG_DIR"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi
echo "--------------------------------------------------------"

echo "Generating stage-2 config (scorer-only)..."
"$PY_BIN" tools/flames_make_stage_config.py \
  --config "$CONFIG_PATH" \
  --stage 2 \
  --out "$CFG_STAGE2" \
  --resp-jsonl "$RESP_JSONL" \
  --result-dir "$RESULT_DIR" \
  > "$LOG_DIR/flames_make_stage2_${TS}.log" 2>&1

echo "Stage-2: run scorer -> final metrics"
CUDA_VISIBLE_DEVICES="$SCORER_CUDA_VISIBLE_DEVICES" "$PY_BIN" tools/run.py "$CFG_STAGE2" "${OVERRIDES[@]}" > "$RUN_LOG" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "Success. Results at: $RESULT_DIR"
  echo "Report: $RESULT_DIR/report.md"
else
  echo "Failure (exit $EXIT_CODE). See log: $RUN_LOG"
  tail -n 50 "$RUN_LOG" || true
fi
exit $EXIT_CODE

