#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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

export VLLM_USE_CPU=1
export CUDA_VISIBLE_DEVICES=""
export VLLM_LOGGING_LEVEL=ERROR



export no_proxy="localhost,127.0.0.1,0.0.0.0,::1,.pjlab.org.cn,.h.pjlab.org.cn"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1,.pjlab.org.cn,.h.pjlab.org.cn"

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
cd "$MBEF_ROOT"

PY_BIN=$(which python3)
echo "Parsing config: $CONFIG_PATH"
PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config --config "$CONFIG_PATH" --mbef-root "$MBEF_ROOT" --format bash --strict "${OVERRIDES[@]}")" \
  || { echo "ERROR: failed to parse config: $CONFIG_PATH"; exit 1; }
eval "$PARSED_ENV"

EXP_NAME="${EXP_NAME:-ch3ef_task}"
LOG_DIR="${LOG_DIR:-$MBEF_ROOT/logs/$EXP_NAME/eval}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

if [[ -n "$OUTPUT_DIR_REL" ]]; then
  PREDICTIONS_PATH_DEFAULT="$MBEF_ROOT/$OUTPUT_DIR_REL/predictions.jsonl"
else
  PREDICTIONS_PATH_DEFAULT="$MBEF_ROOT/results/$EXP_NAME/predictions.jsonl"
fi
PREDICTIONS_PATH="${PREDICTIONS_PATH:-$PREDICTIONS_PATH_DEFAULT}"

echo "--------------------------------------------------------"
echo "Stage:  eval"
echo "Task:   $EXP_NAME"
echo "Pred:   $PREDICTIONS_PATH"
if [[ -n "$JUDGE_MODEL" ]]; then
  echo "Judge:  $JUDGE_MODEL"
fi
echo "Logs:   $LOG_DIR"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi
echo "--------------------------------------------------------"

echo "Running evaluation..."
RUN_LOG="$LOG_DIR/mbef_eval_$TIMESTAMP.log"

"$PY_BIN" tools/run.py "$CONFIG_PATH" \
  --runner.stage eval \
  --runner.predictions_path "$PREDICTIONS_PATH" \
  --runner.require_predictions true \
  --evaluator.use_precomputed_predictions true \
  --evaluator.require_precomputed_predictions true \
  --evaluator.prediction_field prediction \
  --model.type NoOpModel \
  "${OVERRIDES[@]}" \
  > "$RUN_LOG" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "Success. Results at: $MBEF_ROOT/$OUTPUT_DIR_REL"
  echo "Report: $MBEF_ROOT/$OUTPUT_DIR_REL/report.md"
else
  echo "Failure exit $EXIT_CODE. Check log: $RUN_LOG"
  tail -n 30 "$RUN_LOG"
fi
exit $EXIT_CODE

