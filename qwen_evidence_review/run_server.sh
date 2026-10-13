#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  TASK_PYTHON="$PYTHON_BIN"
elif [[ -x "$TASK_ROOT/.venv_server/bin/python" ]]; then
  TASK_PYTHON="$TASK_ROOT/.venv_server/bin/python"
else
  TASK_PYTHON=python
fi
export PYTHONUNBUFFERED=1
exec "$TASK_PYTHON" "$TASK_ROOT/run.py" "$@"
