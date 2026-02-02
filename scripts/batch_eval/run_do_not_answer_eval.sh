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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH
cd "$MBEF_ROOT"




PY_BIN="$(which python3)"
echo "Parsing config: $CONFIG_PATH"
PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config --config "$CONFIG_PATH" --mbef-root "$MBEF_ROOT" --format bash "${OVERRIDES[@]}")" \
  || { echo "ERROR: failed to parse config: $CONFIG_PATH"; exit 1; }
eval "$PARSED_ENV"

EXP_NAME="${EXP_NAME:-do_not_answer}"
LOG_DIR="${LOG_DIR:-$MBEF_ROOT/scripts/rlaunch_logs/$EXP_NAME}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

if [[ -n "$OUTPUT_DIR_REL" ]]; then
  PREDICTIONS_PATH_DEFAULT="$MBEF_ROOT/$OUTPUT_DIR_REL/predictions.jsonl"
else
  PREDICTIONS_PATH_DEFAULT="$MBEF_ROOT/results/$EXP_NAME/predictions.jsonl"
fi
PREDICTIONS_PATH="${PREDICTIONS_PATH:-$PREDICTIONS_PATH_DEFAULT}"

echo "--------------------------------------------------------"
echo "Stage:  eval(do-not-answer)"
echo "Task:   $EXP_NAME"
echo "Pred:   $PREDICTIONS_PATH"
echo "Logs:   $LOG_DIR"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi
echo "--------------------------------------------------------"

RUN_LOG="$LOG_DIR/mbef_eval_$TIMESTAMP.log"

"$PY_BIN" tools/run.py "$CONFIG_PATH" \
  "${OVERRIDES[@]}" \
  --runner.stage eval \
  --runner.predictions_path "$PREDICTIONS_PATH" \
  --runner.require_predictions true \
  --model.type NoOpModel \
  --evaluator.use_precomputed_predictions true \
  --evaluator.require_precomputed_predictions false \
  --evaluator.prediction_field prediction \
  --evaluator.mode score_only \
  > "$RUN_LOG" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "Success. Results at: $MBEF_ROOT/$OUTPUT_DIR_REL"
  echo "Report: $MBEF_ROOT/$OUTPUT_DIR_REL/report.md"
else
  echo "Failure (exit $EXIT_CODE). Check log: $RUN_LOG"
  tail -n 30 "$RUN_LOG"
fi
exit $EXIT_CODE

