"""Single-question entry point.

Usage:
    python -m core_judge.run_single --config core_judge/config.yaml \
        --question "..." --response "..."

Or put `input: { question, model_response, subject?, sub_discipline? }` in the
config YAML and simply run `python -m core_judge.run_single --config ...`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .pipeline import JudgePipeline
from .schemas import EvalInput
from .utils import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Core Judge on a single question.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--question", default=None, help="Override question from config")
    parser.add_argument("--response", default=None, help="Override model_response from config")
    parser.add_argument("--subject", default="", help="Optional subject label")
    parser.add_argument("--sub-discipline", default="", help="Optional sub_discipline label")
    parser.add_argument("--output", default=None, help="Override output path")
    return parser.parse_args()


def _resolve_output_path(raw_path: str, timestamp: bool) -> Path:
    path = Path(raw_path)
    if not timestamp:
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix or '.json'}")


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)

    if args.question and args.response:
        eval_input = EvalInput(
            question=args.question.strip(),
            model_response=args.response.strip(),
            subject=args.subject.strip(),
            sub_discipline=args.sub_discipline.strip(),
        )
    elif config.input is not None:
        eval_input = config.input
    else:
        print("error: need --question + --response, or an `input` block in config", file=sys.stderr)
        return 2

    pipeline = JudgePipeline.from_app_config(config)
    output = pipeline.run(eval_input)

    output_path_str = args.output or config.runtime.output_path
    output_path = _resolve_output_path(output_path_str, config.runtime.output_path_timestamp)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[done] E={output.overall_executability} N={output.overall_net_new_risk} -> {output_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
