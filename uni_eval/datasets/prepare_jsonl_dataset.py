                      
"""
Prepare a minimal uni_eval-compatible JSONL dataset.

Goal:
  Convert/validate an input json/jsonl file into a JSONL where each line has at least:
    - id
    - question
    - response

Usage examples:
  python -m uni_eval.datasets.prepare_jsonl_dataset \
    --input /path/raw.jsonl --output /path/ready.jsonl

  python -m uni_eval.datasets.prepare_jsonl_dataset \
    --input /path/raw.json --output /path/ready.jsonl \
    --id-key qid --question-key prompt --response-key answer --auto-id
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional

def _iter_records_from_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON at {path}:{ln}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{ln}, got {type(obj)}")
            yield obj

def _iter_records_from_json(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    rows: Optional[List[Any]] = None
    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        for k in ("data", "items", "examples"):
            v = obj.get(k)
            if isinstance(v, list):
                rows = v
                break

    if rows is None:
        raise ValueError(f"Unsupported JSON structure in {path}: expected a list or a dict with data/items/examples list.")

    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValueError(f"Expected dict rows in {path}, but rows[{i}] is {type(r)}")
        yield r

def iter_records(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.endswith(".jsonl"):
        return _iter_records_from_jsonl(path)
    if path.endswith(".json"):
        return _iter_records_from_json(path)
    raise ValueError(f"Unsupported input extension for {path}. Please provide .jsonl or .json")

def prepare(
    records: Iterable[Dict[str, Any]],
    *,
    id_key: str,
    question_key: str,
    response_key: str,
    auto_id: bool,
    on_error: str,
    strip: bool,
    ensure_unique_id: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()

    def _handle_err(msg: str) -> None:
        if on_error == "skip":
            return
        raise ValueError(msg)

    for idx, r in enumerate(records):
        rid = r.get(id_key)
        if rid is None and auto_id:
            rid = idx

        q = r.get(question_key)
        resp = r.get(response_key)

        if isinstance(q, str) and strip:
            q = q.strip()
        if isinstance(resp, str) and strip:
            resp = resp.strip()

        if rid is None:
            _handle_err(f"Missing id (key='{id_key}') at record #{idx}")
            if on_error == "skip":
                continue

        if q is None or (isinstance(q, str) and not q):
            _handle_err(f"Missing/empty question (key='{question_key}') at record #{idx}, id={rid!r}")
            if on_error == "skip":
                continue

        if resp is None or (isinstance(resp, str) and not resp):
            _handle_err(f"Missing/empty response (key='{response_key}') at record #{idx}, id={rid!r}")
            if on_error == "skip":
                continue

        if ensure_unique_id:
            sid = str(rid)
            if sid in seen:
                _handle_err(f"Duplicate id={rid!r} at record #{idx}")
                if on_error == "skip":
                    continue
            seen.add(sid)

        out.append({"id": rid, "question": q, "response": resp})

    return out

def write_jsonl(rows: List[Dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare uni_eval JSONL dataset with required fields: id/question/response")
    ap.add_argument("--input", required=True, help="Input .jsonl or .json")
    ap.add_argument("--output", required=True, help="Output .jsonl path")
    ap.add_argument("--id-key", default="id", help="Field name for id in input")
    ap.add_argument("--question-key", default="question", help="Field name for question in input")
    ap.add_argument("--response-key", default="response", help="Field name for response in input")
    ap.add_argument("--auto-id", action="store_true", help="If id is missing, auto-generate with row index")
    ap.add_argument(
        "--on-error",
        choices=("error", "skip"),
        default="error",
        help="What to do when a record is invalid",
    )
    ap.add_argument("--no-strip", action="store_true", help="Do not strip leading/trailing whitespace")
    ap.add_argument("--no-unique-id", action="store_true", help="Do not enforce unique ids")
    ap.add_argument("--dry-run", action="store_true", help="Validate only; do not write output")

    args = ap.parse_args()

    rows = prepare(
        iter_records(args.input),
        id_key=args.id_key,
        question_key=args.question_key,
        response_key=args.response_key,
        auto_id=bool(args.auto_id),
        on_error=str(args.on_error),
        strip=not bool(args.no_strip),
        ensure_unique_id=not bool(args.no_unique_id),
    )

    if args.dry_run:
        print(f"OK: {len(rows)} records validated. (dry-run, no output written)")
        return

    write_jsonl(rows, args.output)
    print(f"OK: wrote {len(rows)} records to {args.output}")

if __name__ == "__main__":
    main()

