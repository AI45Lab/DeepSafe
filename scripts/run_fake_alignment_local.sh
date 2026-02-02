#!/bin/bash
set -eo pipefail

CONFIG_PATH="${1:-}"
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"

PY_BIN="$(which python3)"
PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config \
    --config "$CONFIG_PATH" \
    --mbef-root "$MBEF_ROOT" \
    --format bash)"
eval "$PARSED_ENV"

EXP_NAME="${EXP_NAME:-fake_alignment_task}"
export MBEF_LOG_DIR="$MBEF_ROOT/scripts/local_logs/$EXP_NAME"
mkdir -p "$MBEF_LOG_DIR"

echo "==== Starting Two-Stage Local Run: $EXP_NAME ===="
bash "$SCRIPT_DIR/local/run_fake_alignment_gen_local.sh" "$CONFIG_PATH"
bash "$SCRIPT_DIR/local/run_fake_alignment_eval_local.sh" "$CONFIG_PATH"
echo "==== Finished Two-Stage Local Run: $EXP_NAME ===="
