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

echo "==================== FLAMES: all (gen -> eval) ===================="
echo "Config: $CONFIG_PATH"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  echo "Overrides: ${OVERRIDES[*]}"
fi

bash "$MBEF_ROOT/scripts/run_flames_gen.sh" "$CONFIG_PATH" "${OVERRIDES[@]}"
echo "-------------------- stage-1 done --------------------"
bash "$MBEF_ROOT/scripts/run_flames_eval.sh" "$CONFIG_PATH" "${OVERRIDES[@]}"
echo "-------------------- stage-2 done --------------------"

