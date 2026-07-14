"""JSONL entry point for running only the core judge pipeline.

This runner is intentionally narrower than repo-level scripts such as
`run_unsafe_eval.py`: it only reads question/response pairs, calls
`JudgePipeline.run()`, and writes judge outputs back to JSONL.

Usage:
    python -m core_judge.run_jsonl \
        --config core_judge/config.yaml \
        --input path/to/responses.jsonl

    python -m core_judge.run_jsonl \
        --config core_judge/config.yaml \
        --input path/to/responses.jsonl \
        --output ./responses.judged.jsonl \
        --question-field prompt \
        --response-field prediction
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from .pipeline import JudgePipeline
from .schemas import EvalInput
from .utils import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only the Core Judge pipeline on a JSONL file.")
    parser.add_argument("--config", required=True, help="Path to core_judge config.yaml")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", default=None, help="Output JSONL path (default: <input>.judged.jsonl)")
    parser.add_argument("--workers", type=int, default=0, help="Override runtime.workers")
    parser.add_argument("--question-field", default="question", help="Field containing the user question")
    parser.add_argument(
        "--response-field",
        default="_response_text",
        help="Field containing the model response",
    )
    parser.add_argument("--subject-field", default="subject", help="Optional subject field")
    parser.add_argument(
        "--sub-discipline-field",
        default="sub_discipline",
        help="Optional sub_discipline field",
    )
    parser.add_argument(
        "--id-field",
        default="idx",
        help="Field used in progress logs; falls back to line number if missing",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run rows even if judge_E/judge_N already exist",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N non-empty rows (0 = all)",
    )
    return parser.parse_args()


def _default_output_path(input_path: Path) -> Path:
    if input_path.suffix.lower() == ".jsonl":
        return input_path.with_name(f"{input_path.stem}.judged.jsonl")
    return input_path.with_name(f"{input_path.name}.judged.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any] | None]:
    rows: list[dict[str, Any] | None] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            rows.append(None)
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_no}, got {type(payload)}")
        rows.append(payload)
    return rows


def _build_eval_input(
    row: dict[str, Any],
    *,
    question_field: str,
    response_field: str,
    subject_field: str,
    sub_discipline_field: str,
) -> EvalInput:
    question = str(row.get(question_field) or "").strip()
    response = str(row.get(response_field) or "").strip()
    subject = str(row.get(subject_field) or "").strip()
    sub_discipline = str(row.get(sub_discipline_field) or "").strip()
    eval_input = EvalInput(
        question=question,
        model_response=response,
        subject=subject,
        sub_discipline=sub_discipline,
    )
    eval_input.validate()
    return eval_input


def _judge_one(
    pipeline: JudgePipeline,
    row: dict[str, Any],
    *,
    row_id: str,
    question_field: str,
    response_field: str,
    subject_field: str,
    sub_discipline_field: str,
    output_lock: Lock,
) -> dict[str, Any]:
    eval_input = _build_eval_input(
        row,
        question_field=question_field,
        response_field=response_field,
        subject_field=subject_field,
        sub_discipline_field=sub_discipline_field,
    )
    out = pipeline.run(
        eval_input,
        progress_prefix=f"[{row_id}]",
        output_lock=output_lock,
    )

    updated = dict(row)
    updated["model_response"] = out.model_response
    updated["prompt_trace_path"] = out.prompt_trace_path
    updated["judge_E"] = out.overall_executability
    updated["judge_N"] = out.overall_net_new_risk
    updated["rule_E"] = out.rule_executability
    updated["judge_E_reason"] = out.overall_reason
    updated["judge_N_reason"] = out.overall_net_new_reason
    updated["final_score"] = f"{out.overall_executability}_{out.overall_net_new_risk}"
    updated["checklist_cache_hit"] = out.checklist_cache_hit
    updated["search_queries"] = list(out.search_queries)
    updated["search_traces"] = [trace.to_dict() for trace in out.search_traces]
    updated["checklist"] = [item.to_dict() for item in out.checklist]
    updated["checklist_coverage"] = [item.to_dict() for item in out.checklist_coverage]
    updated["coverage_stats"] = out.coverage_stats.to_dict()
    updated["claims"] = [claim.to_dict() for claim in out.claims]
    updated["claim_results"] = [result.to_dict() for result in out.claim_results]
    updated["net_new_stats"] = out.net_new_stats.to_dict()
    updated["timings"] = dict(out.timings)
    updated.pop("judge_error", None)
    return updated


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 2

    output_path = Path(args.output) if args.output else _default_output_path(input_path)
    rows = _read_jsonl(input_path)
    if args.limit > 0:
        seen = 0
        limited_rows: list[dict[str, Any] | None] = []
        for row in rows:
            if row is not None:
                seen += 1
            if seen > args.limit:
                break
            limited_rows.append(row)
        rows = limited_rows

    config = load_config(args.config)
    pipeline = JudgePipeline.from_app_config(config)
    workers = max(1, args.workers or config.runtime.workers)
    output_lock = Lock()

    pending_indices: list[int] = []
    skipped = 0
    for idx, row in enumerate(rows):
        if row is None:
            continue
        if not args.rerun and (row.get("judge_E") is not None or row.get("judge_N") is not None):
            skipped += 1
            continue
        pending_indices.append(idx)

    print(
        f"[judge-jsonl] total_rows={len(rows)} pending={len(pending_indices)} skipped={skipped} workers={workers}",
        file=sys.stderr,
    )
    if not pending_indices:
        output_path.write_text(
            "\n".join("" if row is None else json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        print(f"[judge-jsonl] nothing to do, wrote {output_path}", file=sys.stderr)
        return 0

    start_time = time.time()

    def _row_id(line_index: int, row: dict[str, Any]) -> str:
        value = row.get(args.id_field)
        if value in (None, ""):
            return f"line_{line_index + 1}"
        return str(value)

    if workers <= 1:
        for line_index in pending_indices:
            row = rows[line_index]
            assert row is not None
            try:
                rows[line_index] = _judge_one(
                    pipeline,
                    row,
                    row_id=_row_id(line_index, row),
                    question_field=args.question_field,
                    response_field=args.response_field,
                    subject_field=args.subject_field,
                    sub_discipline_field=args.sub_discipline_field,
                    output_lock=output_lock,
                )
            except Exception as exc:
                row["judge_error"] = f"{type(exc).__name__}: {exc}"
                row["judge_traceback"] = traceback.format_exc(limit=3)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {}
            for line_index in pending_indices:
                row = rows[line_index]
                assert row is not None
                future = executor.submit(
                    _judge_one,
                    pipeline,
                    row,
                    row_id=_row_id(line_index, row),
                    question_field=args.question_field,
                    response_field=args.response_field,
                    subject_field=args.subject_field,
                    sub_discipline_field=args.sub_discipline_field,
                    output_lock=output_lock,
                )
                future_to_index[future] = line_index
            for future in as_completed(future_to_index):
                line_index = future_to_index[future]
                row = rows[line_index]
                assert row is not None
                try:
                    rows[line_index] = future.result()
                except Exception as exc:
                    row["judge_error"] = f"{type(exc).__name__}: {exc}"
                    row["judge_traceback"] = traceback.format_exc(limit=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if row is None:
                fh.write("\n")
            else:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    ok = sum(1 for idx in pending_indices if rows[idx] is not None and "judge_error" not in rows[idx])
    print(
        f"[judge-jsonl] done ok={ok}/{len(pending_indices)} elapsed={elapsed:.2f}s output={output_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
