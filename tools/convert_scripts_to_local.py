import os
import re
from pathlib import Path

def convert_script(input_path, output_path, stage):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = re.sub(r'# Runner \(.*\) for (.*)', f'# Local Runner ({stage.upper()} ONLY) for \\1', content)

    env_block = """# 1. Environment Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MBEF_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$MBEF_ROOT"
cd "$MBEF_ROOT"

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
"""
                                                            
    content = re.sub(r'# 1\. Environment Setup.*?cd "\$MBEF_ROOT"', env_block, content, flags=re.DOTALL)

    content = re.sub(r'export http_proxy=.*?\n', '', content)
    content = re.sub(r'export https_proxy=.*?\n', '', content)
    content = re.sub(r'export HTTP_PROXY=.*?\n', '', content)
    content = re.sub(r'export HTTPS_PROXY=.*?\n', '', content)

    content = content.replace("# rjob will set it to the allocated GPU list", "# Set GPU visibility")
    content = content.replace("CUDA_VISIBLE_DEVICES=0 nohup", "nohup")                                

    content = content.replace('rlaunch_logs', 'local_logs')

    content = content.replace('local retries=60', 'local retries=600')

    if stage == 'gen':
        content = re.sub(r'"\$PY_BIN" tools/run\.py.*?> "\$RUN_LOG" 2>&1', 
                         r'"$PY_BIN" tools/run.py "$CONFIG_PATH" --runner.stage gen --runner.predictions_path "$PREDICTIONS_PATH" "${OVERRIDES[@]}" 2>&1 | tee "$RUN_LOG"', 
                         content)
    else:
                                            
        content = re.sub(r'"\$PY_BIN" tools/run\.py.*?> "\$RUN_LOG" 2>&1', 
                         r'"$PY_BIN" tools/run.py "$CONFIG_PATH" --runner.stage eval --runner.predictions_path "$PREDICTIONS_PATH" --model.type NoOpModel "${OVERRIDES[@]}" 2>&1 | tee "$RUN_LOG"', 
                         content)

    content = content.replace('EXIT_CODE=$?', 'EXIT_CODE=${PIPESTATUS[0]}')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    os.chmod(output_path, 0o755)

def create_main_runner(bench_name, output_path):
    gen_script = f"run_{bench_name}_gen_local.sh"
    eval_script = f"run_{bench_name}_eval_local.sh"
    
    content = f"""#!/bin/bash
set -eo pipefail

# One-shot local runner for {bench_name.capitalize()} (GEN -> EVAL)
# Usage: bash scripts/run_{bench_name}_local.sh /abs/path/to/config.yaml

CONFIG_PATH="${{1:-}}"
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
MBEF_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 1. Environment Setup
export PYTHONPATH="${{PYTHONPATH:-}}:$MBEF_ROOT"
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"

# Resolve global log dir from config
PY_BIN="$(which python3)"
PARSED_ENV="$("$PY_BIN" -m uni_eval.cli.parse_eval_config \\
    --config "$CONFIG_PATH" \\
    --mbef-root "$MBEF_ROOT" \\
    --format bash)"
eval "$PARSED_ENV"

EXP_NAME="${{EXP_NAME:-{bench_name}_task}}"
export MBEF_LOG_DIR="$MBEF_ROOT/scripts/local_logs/$EXP_NAME"
mkdir -p "$MBEF_LOG_DIR"

echo "==== Starting Two-Stage Local Run: $EXP_NAME ===="
bash "$SCRIPT_DIR/local/{gen_script}" "$CONFIG_PATH"
bash "$SCRIPT_DIR/local/{eval_script}" "$CONFIG_PATH"
echo "==== Finished Two-Stage Local Run: $EXP_NAME ===="
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    os.chmod(output_path, 0o755)

def main():
    scripts_dir = Path("scripts/batch_eval")
    local_dir = Path("scripts/local")
    local_dir.mkdir(parents=True, exist_ok=True)

    single_stage = ['wmdp', 'reason_under_pressure']

    benchmarks = set()
    for f in scripts_dir.glob("run_*_gen.sh"):
        bench_name = f.name[4:-7]
        benchmarks.add(bench_name)
    
    for bench in benchmarks:
        print(f"Processing {bench}...")
        gen_in = scripts_dir / f"run_{bench}_gen.sh"
        eval_in = scripts_dir / f"run_{bench}_eval.sh"
        
        gen_out = local_dir / f"run_{bench}_gen_local.sh"
        eval_out = local_dir / f"run_{bench}_eval_local.sh"
        
        if gen_in.exists():
            convert_script(gen_in, gen_out, 'gen')
        if eval_in.exists():
            convert_script(eval_in, eval_out, 'eval')
            
        create_main_runner(bench, Path("scripts") / f"run_{bench}_local.sh")

    for bench in single_stage:
        print(f"Processing single-stage {bench}...")

if __name__ == "__main__":
    main()
