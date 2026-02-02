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

EXP_NAME="${EXP_NAME:-truthful_qa_task}"
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
echo "Output: $MBEF_ROOT/$OUTPUT_DIR_REL"
echo "--------------------------------------------------------"

echo "Running Evaluation..."
RUN_LOG="$LOG_DIR/mbef_eval_$TIMESTAMP.log"

"$PY_BIN" tools/run.py "$CONFIG_PATH" \
  --runner.stage eval \
  --runner.predictions_path "$PREDICTIONS_PATH" \
  --runner.require_predictions true \
  --evaluator.use_precomputed_predictions true \
  --evaluator.require_precomputed_predictions true \
  --model.type NoOpModel \
  "${OVERRIDES[@]}" \
  2>&1 | tee "$RUN_LOG"

EXIT_CODE=${PIPESTATUS[0]}
exit $EXIT_CODE
