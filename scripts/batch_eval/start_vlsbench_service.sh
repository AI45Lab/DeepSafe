#!/bin/bash
# NOTE: internal storage paths, registry and cluster mounts were replaced with placeholders for the public release. Fill in your own values before running.
set -euo pipefail
export TZ=Asia/Shanghai
MBFF_ROOT="${MBFF_ROOT:-/path/to/MBFF-1222}"
cd "$MBFF_ROOT"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

: "${MBFF_SERVICE_KEY:=}"
export MBFF_SERVICE_KEY

exec python3 -m uvicorn vlsbench_service:app --host "$HOST" --port "$PORT"

