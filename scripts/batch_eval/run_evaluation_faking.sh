#!/bin/bash

set -e

CONFIG_PATH="$1"
shift || true
OVERRIDES=("$@")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="${MBEF_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
LOG_DIR="${LOG_DIR:-$MBEF_ROOT/logs}"

if [[ -z "$CONFIG_PATH" ]]; then
  echo "使用方法: $0 <你的yaml配置文件路径> [可选覆盖参数...]"
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "错误: 配置文件不存在 → $CONFIG_PATH"
  exit 1
fi

mkdir -p "$LOG_DIR"
export PYTHONPATH="${PYTHONPATH}:$MBEF_ROOT"
cd "$MBEF_ROOT"
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

export API_MODEL_DEBUG="${API_MODEL_DEBUG:-}"

echo -e "\n====================================="
echo "开始执行评估任务 | 配置文件: $CONFIG_PATH"
echo "====================================="
RUN_LOG="$LOG_DIR/eval_run_$TIMESTAMP.log"
python3 tools/run.py $CONFIG_PATH "${OVERRIDES[@]}" 

