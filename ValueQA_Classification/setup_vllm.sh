#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PYTHON="${VLLM_SETUP_PYTHON:-python}"
# vLLM chooses its own torch/CUDA wheels; never install into the training environment.
"$TASK_PYTHON" -m venv "$TASK_ROOT/.venv_vllm"
"$TASK_ROOT/.venv_vllm/bin/python" -m pip install "vllm==0.31.0"
"$TASK_ROOT/.venv_vllm/bin/python" -m pip check
"$TASK_ROOT/.venv_vllm/bin/python" -c 'from vllm import LLM, SamplingParams; print("vLLM imports OK; GPU/model compatibility still requires a real trial")'
echo "Set VLLM_PYTHON=$TASK_ROOT/.venv_vllm/bin/python before using run_server.sh."
