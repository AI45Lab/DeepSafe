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

EXP_NAME="${EXP_NAME:-flames_task}"
if [[ -n "$MBEF_LOG_DIR" ]]; then
  LOG_DIR="$MBEF_LOG_DIR"
else
  LOG_DIR="$MBEF_ROOT/scripts/local_logs/$EXP_NAME"
fi
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

STAGE_DIR="$RESULT_DIR/_stage1"
RESP_JSONL="${FLAMES_RESP_JSONL:-$STAGE_DIR/flames_target_responses.jsonl}"
CFG_STAGE2="$LOG_DIR/_cfg_stage2_$TIMESTAMP.yaml"

echo "--------------------------------------------------------"
echo "Stage:  EVAL"
echo "Task:   $EXP_NAME"
echo "Scorer: Transformers-based"
echo "Input:  $RESP_JSONL"
echo "Logs:   $LOG_DIR"
echo "--------------------------------------------------------"

echo "Generating stage-2 config..."
"$PY_BIN" tools/flames_make_stage_config.py \
  --config "$CONFIG_PATH" \
  --stage 2 \
  --out "$CFG_STAGE2" \
  --resp-jsonl "$RESP_JSONL" \
  --result-dir "$RESULT_DIR"

echo "Stage-2: Running scorer..."
RUN_LOG="$LOG_DIR/mbef_stage2_$TIMESTAMP.log"
"$PY_BIN" tools/run.py "$CFG_STAGE2" "${OVERRIDES[@]}" 2>&1 | tee "$RUN_LOG"

EXIT_CODE=${PIPESTATUS[0]}
exit $EXIT_CODE
