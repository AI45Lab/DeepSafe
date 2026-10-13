param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$taskRoot = $PSScriptRoot
$taskVenv = Join-Path $taskRoot ".venv"
$taskPy = Join-Path $taskVenv "Scripts\python.exe"
# Reuse installed CUDA torch; keep new transformers/PEFT packages in this folder's environment.
if (-not (Test-Path -LiteralPath $taskPy)) {
    & $Python -m venv --system-site-packages $taskVenv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}
& $taskPy -c "import torch; assert torch.cuda.is_available(), 'Install CUDA PyTorch first'; print(torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "CUDA torch check failed; follow README.md" }
& $taskPy -m pip install --upgrade -r (Join-Path $taskRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Package installation failed" }
& $taskPy (Join-Path $taskRoot "check_env.py") --config (Join-Path $taskRoot "config.current.local.json") --dependencies-only
if ($LASTEXITCODE -ne 0) { throw "Environment check failed" }
Write-Host "Dependencies ready. Select config.current.local.json explicitly and configure model paths before inference. No model weights were downloaded."
