#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PYTHON="${PYTHON_BIN:-python}"
"$TASK_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "Activate the working CUDA training environment first"; print(torch.__version__, torch.cuda.get_device_name(0))'
# Keep upgrades out of the training environment, but reuse its CUDA PyTorch.
"$TASK_PYTHON" -m venv --system-site-packages "$TASK_ROOT/.venv_server"
"$TASK_ROOT/.venv_server/bin/python" -m pip install -r "$TASK_ROOT/requirements.txt"
"$TASK_ROOT/.venv_server/bin/python" "$TASK_ROOT/check_env.py" --config "$TASK_ROOT/config.current.server.json" --dependencies-only
echo 'Server environment prepared. No model weights were downloaded. Configure the base and adapter paths, then run_server.sh.'
