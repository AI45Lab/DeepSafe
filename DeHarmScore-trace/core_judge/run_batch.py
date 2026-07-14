"""Batch entry point.

Reads an input JSON file or directory of JSON files (each containing at minimum
`question` and `model_response`), runs the judge pipeline on each, and writes
one output JSON per input plus an aggregate JSONL report.

Usage:
    python -m core_judge.run_batch --config core_judge/config.yaml \
        --input data/final --output output/
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

from .pipeline import JudgePipeline
from .schemas import EvalInput
from .utils import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Core Judge in batch mode.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--input", required=True, help="Input JSON file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=0, help="Override runtime.workers")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of examples (0 = all)")
    return parser.parse_args()


def _load_inputs(input_path: Path) -> list[tuple[str, dict]]:
    records: list[tuple[str, dict]] = []
    if input_path.is_dir():
        # Use case-insensitive pattern on Linux (case-sensitive FS) by listing both cases
        matched = sorted(
            set(input_path.glob("*.json"))
            | set(input_path.glob("*.JSON"))
            | set(input_path.glob("*.jsonl"))
            | set(input_path.glob("*.JSONL"))
        )
        for file_path in matched:
            records.extend(_load_json_records(file_path))
    else:
        records.extend(_load_json_records(input_path))
    return records


def _load_json_records(file_path: Path) -> list[tuple[str, dict]]:
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [(file_path.stem, data)]
    if isinstance(data, list):
        return [
            (f"{file_path.stem}_{idx}", row)
            for idx, row in enumerate(data)
            if isinstance(row, dict)
        ]
    return []


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    pipeline = JudgePipeline.from_app_config(config)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_inputs(input_path)
    if args.limit and args.limit > 0:
        records = records[: args.limit]
    if not records:
        print(f"error: no input records found at {input_path}", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"report_{stamp}.jsonl"
    output_lock = Lock()

    workers = max(1, args.workers or config.runtime.workers)
    print(f"[batch] {len(records)} records, workers={workers}", file=sys.stderr)

    def _run_one(record_id: str, payload: dict) -> dict:
        eval_input = EvalInput.from_dict(payload)
        result = pipeline.run(
            eval_input,
            progress_prefix=f"[{record_id}]",
            output_lock=output_lock,
        )
        single_path = output_dir / f"{record_id}.json"
        single_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "id": record_id,
            "E": result.overall_executability,
            "N": result.overall_net_new_risk,
            "prompt_trace_path": result.prompt_trace_path,
            "output_path": str(single_path),
        }

    summaries: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(_run_one, record_id, payload): record_id
            for record_id, payload in records
        }
        for future in as_completed(future_to_id):
            record_id = future_to_id[future]
            try:
                summaries.append(future.result())
            except Exception as exc:
                summaries.append(
                    {
                        "id": record_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=3),
                    }
                )

    with report_path.open("w", encoding="utf-8") as fh:
        for summary in summaries:
            fh.write(json.dumps(summary, ensure_ascii=False) + "\n")

    ok = sum(1 for s in summaries if "error" not in s)
    print(f"[batch] done: {ok}/{len(summaries)} ok, report -> {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
