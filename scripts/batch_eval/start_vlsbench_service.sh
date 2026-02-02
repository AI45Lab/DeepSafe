#!/bin/bash
set -euo pipefail
export TZ=Asia/Shanghai
MBFF_ROOT="/root/zhangbo1/MBFF-1222"
cd "$MBFF_ROOT"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

: "${MBFF_SERVICE_KEY:=}"
export MBFF_SERVICE_KEY

exec python3 -m uvicorn vlsbench_service:app --host "$HOST" --port "$PORT"

