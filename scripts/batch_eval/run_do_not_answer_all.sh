#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -e

CONFIG_PATH="$1"
shift || true
OVERRIDES=("$@")

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path> [overrides...]" >&2
  exit 1
fi

MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
RUN_GEN="$MBEF_ROOT/scripts/run_do_not_answer_gen.sh"
RUN_EVAL="$MBEF_ROOT/scripts/run_do_not_answer_eval.sh"

echo "==================== Stage: gen(do-not-answer) ===================="
bash "$RUN_GEN" "$CONFIG_PATH" "${OVERRIDES[@]}"

echo "==================== Stage: eval(do-not-answer) ===================="
bash "$RUN_EVAL" "$CONFIG_PATH" "${OVERRIDES[@]}"

