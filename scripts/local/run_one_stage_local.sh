#!/bin/bash
set -eo pipefail

CONFIG_PATH="${1:-}"
EXTRA_ARG="${2:-}"

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path>"
  exit 1
fi
if [[ -n "$EXTRA_ARG" ]]; then
  echo "ERROR: This script only accepts ONE argument: <config_path>" >&2
  exit 2
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
cd "$MBEF_ROOT"

PY_BIN="${PY_BIN:-python3}"
PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config \
  --config "$CONFIG_PATH" \
  --mbef-root "$MBEF_ROOT" \
  --format bash)" || { echo "ERROR: failed to parse config: $CONFIG_PATH"; exit 1; }
eval "$PARSED_ENV"

EXP_NAME="${EXP_NAME:-$(basename "$CONFIG_PATH" | sed 's/\.ya\?ml$//')}"
LOG_DIR="${MBEF_LOG_DIR:-$MBEF_ROOT/scripts/local_logs/$EXP_NAME}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d%H%M%S")"

echo "========================================================"
echo "MBEF one-shot (one-stage local)"
echo "Config: $CONFIG_PATH"
echo "MBEF:   $MBEF_ROOT"
echo "Logs:   $LOG_DIR"
echo "========================================================"

RUN_LOG="$LOG_DIR/mbef_run_${TIMESTAMP}.log"
"$PY_BIN" tools/run.py "$CONFIG_PATH" > "$RUN_LOG" 2>&1

EXIT_CODE=$?
if [[ $EXIT_CODE -eq 0 ]]; then
  if [[ -n "${OUTPUT_DIR_REL:-}" ]]; then
    echo "Success. Results at: $MBEF_ROOT/$OUTPUT_DIR_REL"
    echo "Report: $MBEF_ROOT/$OUTPUT_DIR_REL/report.md"
  else
    echo "Success."
  fi
else
  echo "Failure (exit $EXIT_CODE). Check log: $RUN_LOG"
  tail -n 80 "$RUN_LOG" || true
fi
exit $EXIT_CODE

