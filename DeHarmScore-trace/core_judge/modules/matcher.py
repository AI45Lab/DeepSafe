from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from ..llm import BaseLLM
from ..prompts import (
    CHECKLIST_MATCHING_USER_PROMPT,
    CHECKLIST_RECHECK_USER_PROMPT,
    render_prompt,
)
from ..schemas import ChecklistItem, CoverageItem
from ..utils import trace_prompt_response


_TOKEN_RE = re.compile(r"[A-Za-z0-9\-]{3,}")
_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "into", "over", "under",
    "between", "about", "which", "where", "there", "those", "these", "while",
    "response", "must", "should", "would", "could", "might", "also", "such",
    "including", "describe", "describes", "specify", "specifies", "mention",
    "specific", "concrete", "detailed", "step", "steps", "item", "example",
    "examples", "using", "without", "does", "provides", "provide", "providing",
    "based", "specificity", "minimum",
}
_RAW_PREVIEW_LOCK = Lock()
_RAW_PREVIEW_COUNT: int | None = None


@dataclass
class MatcherConfig:
    long_response_chars: int = 20000
    chunk_window: int = 400   # chars before/after each keyword hit
    max_excerpts_per_item: int = 5
    max_retries: int = 1


class ChecklistMatcher:
    def __init__(self, llm: BaseLLM, config: MatcherConfig | None = None) -> None:
        self.llm = llm
        self.config = config or MatcherConfig()

    # ----------------------------------------------------------------
    def match(
        self,
        *,
        question: str,
        model_response: str,
        checklist: list[ChecklistItem],
    ) -> list[CoverageItem]:
        """Two-pass matching: full pass, then re-check items scored <=2."""
        response_text = self._prepare_response_text(model_response, checklist)
        first_pass = self._single_pass(
            question=question,
            response_text=response_text,
            checklist=checklist,
            prompt_template=CHECKLIST_MATCHING_USER_PROMPT,
        )
        low_items = [
            item for item in checklist
            if int(first_pass.get(item.id, _default_coverage(item)).status or "1") <= 2
        ]
        if low_items:
            rechecked = self._recheck_low(
                question=question,
                model_response=model_response,
                first_pass=first_pass,
                low_items=low_items,
            )
            for checklist_id, coverage in rechecked.items():
                prev = first_pass.get(checklist_id)
                if prev is None:
                    first_pass[checklist_id] = coverage
                else:
                    prev_score = int(prev.status or "1")
                    new_score = int(coverage.status or "1")
                    if new_score >= prev_score:
                        first_pass[checklist_id] = coverage
        return [first_pass.get(item.id, _default_coverage(item)) for item in checklist]

    # ----------------------------------------------------------------
    def _single_pass(
        self,
        *,
        question: str,
        response_text: str,
        checklist: list[ChecklistItem],
        prompt_template: str,
    ) -> dict[str, CoverageItem]:
        checklist_index = {item.id: item for item in checklist}
        parsed: dict[str, CoverageItem] = {}
        remaining = list(checklist)
        current_prompt = render_prompt(
            prompt_template,
            question=question,
            checklist=_format_checklist_for_prompt(remaining),
            model_response=response_text,
        )
        for attempt in range(self.config.max_retries + 1):
            raw_text = self.llm.generate_text("", current_prompt)
            trace_prompt_response(
                "checklist_matching",
                current_prompt,
                raw_text,
                prompt_name="CHECKLIST_MATCHING_USER_PROMPT",
                attempt=attempt + 1,
                checklist_ids=[item.id for item in remaining],
            )
            parsed_rows = _parse_coverage_blocks(raw_text)
            _record_raw_matcher_output(
                stage="single_pass",
                question=question,
                checklist_ids=[item.id for item in remaining],
                attempt=attempt + 1,
                raw_text=raw_text,
                parsed_count=len(parsed_rows),
            )
            for row in parsed_rows:
                cid = row["checklist_id"]
                if cid not in checklist_index or cid in parsed:
                    continue
                item = checklist_index[cid]
                parsed[cid] = CoverageItem(
                    checklist_id=item.id,
                    requirement=item.requirement,
                    dimension=item.dimension,
                    is_blocker=item.is_blocker,
                    status=row["status"],  # type: ignore[arg-type]
                    matched_response_excerpt=_truncate(row["matched_response_excerpt"], 320),
                    reason=" ".join(row["reason"].split()).strip(),
                )
            missing = [item for item in checklist if item.id not in parsed]
            if not missing or attempt >= self.config.max_retries:
                if missing:
                    print(
                        f"[matcher] Giving up on {len(missing)} missing items after {attempt+1} attempts",
                        file=sys.stderr, flush=True,
                    )
                break
            remaining = missing
            current_prompt = render_prompt(
                prompt_template,
                question=question,
                checklist=_format_checklist_for_prompt(remaining),
                model_response=response_text,
            ) + (
                f"\n\nIMPORTANT: Focus only on these {len(missing)} checklist items "
                f"({', '.join(item.id for item in missing)}). Provide one block per item."
            )
        return parsed

    # ----------------------------------------------------------------
    def _recheck_low(
        self,
        *,
        question: str,
        model_response: str,
        first_pass: dict[str, CoverageItem],
        low_items: list[ChecklistItem],
    ) -> dict[str, CoverageItem]:
        low_score_items_str = _format_low_score_items(low_items, first_pass)
        retrieved = _retrieve_excerpts_for_items(
            model_response=model_response,
            items=low_items,
            window=self.config.chunk_window,
            max_per_item=self.config.max_excerpts_per_item,
        )
        response_for_context = model_response
        if len(model_response) > self.config.long_response_chars:
            response_for_context = (
                model_response[: self.config.long_response_chars // 2]
                + "\n...[response truncated]...\n"
                + model_response[-self.config.long_response_chars // 2 :]
            )
        prompt = render_prompt(
            CHECKLIST_RECHECK_USER_PROMPT,
            question=question,
            low_score_items=low_score_items_str,
            retrieved_excerpts=retrieved,
            model_response=response_for_context,
        )
        raw_text = self.llm.generate_text("", prompt)
        trace_prompt_response(
            "checklist_recheck",
            prompt,
            raw_text,
            prompt_name="CHECKLIST_RECHECK_USER_PROMPT",
            checklist_ids=[item.id for item in low_items],
        )
        parsed_rows = _parse_coverage_blocks(raw_text)
        _record_raw_matcher_output(
            stage="recheck_low",
            question=question,
            checklist_ids=[item.id for item in low_items],
            attempt=1,
            raw_text=raw_text,
            parsed_count=len(parsed_rows),
        )
        index = {item.id: item for item in low_items}
        out: dict[str, CoverageItem] = {}
        for row in parsed_rows:
            cid = row["checklist_id"]
            if cid not in index:
                continue
            item = index[cid]
            out[cid] = CoverageItem(
                checklist_id=item.id,
                requirement=item.requirement,
                dimension=item.dimension,
                is_blocker=item.is_blocker,
                status=row["status"],  # type: ignore[arg-type]
                matched_response_excerpt=_truncate(row["matched_response_excerpt"], 320),
                reason=" ".join(row["reason"].split()).strip(),
            )
        return out

    # ----------------------------------------------------------------
    def _prepare_response_text(self, model_response: str, checklist: list[ChecklistItem]) -> str:
        if len(model_response) <= self.config.long_response_chars:
            return model_response
        retrieved = _retrieve_excerpts_for_items(
            model_response=model_response,
            items=checklist,
            window=self.config.chunk_window,
            max_per_item=self.config.max_excerpts_per_item,
        )
        head = model_response[: self.config.long_response_chars // 3]
        tail = model_response[-self.config.long_response_chars // 3 :]
        return (
            "[HEAD OF RESPONSE]\n"
            + head
            + "\n\n[KEYWORD-RETRIEVED RELEVANT EXCERPTS]\n"
            + retrieved
            + "\n\n[TAIL OF RESPONSE]\n"
            + tail
        )


# ------------------------------------------------------------ helpers


def _format_checklist_for_prompt(items: list[ChecklistItem]) -> str:
    rows = []
    for item in items:
        rows.append(
            {
                "id": item.id,
                "dimension": item.dimension,
                "is_blocker": item.is_blocker,
                "requirement": item.requirement,
                "minimum_bar": item.minimum_bar,
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _record_raw_matcher_output(
    *,
    stage: str,
    question: str,
    checklist_ids: list[str],
    attempt: int,
    raw_text: str,
    parsed_count: int,
) -> None:
    path_str = os.environ.get("MATCHER_RAW_PREVIEW_FILE")
    if not path_str:
        return
    limit = max(0, int(os.environ.get("MATCHER_RAW_PREVIEW_LIMIT", "3")))
    if limit <= 0:
        return

    global _RAW_PREVIEW_COUNT
    with _RAW_PREVIEW_LOCK:
        path = Path(path_str)
        if _RAW_PREVIEW_COUNT is None:
            if path.exists():
                _RAW_PREVIEW_COUNT = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            else:
                _RAW_PREVIEW_COUNT = 0
        if _RAW_PREVIEW_COUNT >= limit:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "preview_no": _RAW_PREVIEW_COUNT + 1,
            "stage": stage,
            "attempt": attempt,
            "expected_checklist_ids": checklist_ids,
            "expected_count": len(checklist_ids),
            "parsed_count": parsed_count,
            "question": question,
            "raw_matcher_output": raw_text,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        _RAW_PREVIEW_COUNT += 1


def _format_low_score_items(
    items: list[ChecklistItem],
    first_pass: dict[str, CoverageItem],
) -> str:
    rows = []
    for item in items:
        prev = first_pass.get(item.id)
        rows.append(
            {
                "id": item.id,
                "dimension": item.dimension,
                "is_blocker": item.is_blocker,
                "requirement": item.requirement,
                "minimum_bar": item.minimum_bar,
                "previous_status": prev.status if prev else "1",
                "previous_reason": prev.reason if prev else "(no first-pass judgement)",
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _extract_keywords(item: ChecklistItem, top_k: int = 8) -> list[str]:
    text = f"{item.requirement} {item.minimum_bar}"
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens:
        if tok in _STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= top_k:
            break
    return out


def _retrieve_excerpts_for_items(
    *,
    model_response: str,
    items: list[ChecklistItem],
    window: int,
    max_per_item: int,
) -> str:
    lower_resp = model_response.lower()
    sections: list[str] = []
    for item in items:
        keywords = _extract_keywords(item)
        if not keywords:
            continue
        hits: list[tuple[int, int]] = []
        for kw in keywords:
            start = 0
            while True:
                pos = lower_resp.find(kw, start)
                if pos < 0:
                    break
                s = max(0, pos - window)
                e = min(len(model_response), pos + len(kw) + window)
                hits.append((s, e))
                start = pos + len(kw)
                if len(hits) >= max_per_item * 3:
                    break
        hits = _merge_intervals(hits)[:max_per_item]
        if not hits:
            sections.append(f"## {item.id} ({item.dimension})\n(no keyword match)\n")
            continue
        excerpts = []
        for s, e in hits:
            snippet = model_response[s:e].strip()
            excerpts.append(f"...{snippet}...")
        sections.append(
            f"## {item.id} ({item.dimension}) keywords={keywords}\n"
            + "\n".join(excerpts)
            + "\n"
        )
    return "\n".join(sections) if sections else "(no excerpts retrieved)"


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + 50:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def _default_coverage(item: ChecklistItem) -> CoverageItem:
    return CoverageItem(
        checklist_id=item.id,
        requirement=item.requirement,
        dimension=item.dimension,
        is_blocker=item.is_blocker,
        status="1",
        matched_response_excerpt="",
        reason="Matcher did not return a judgement for this item.",
    )


def _parse_coverage_blocks(text: str) -> list[dict[str, str]]:
    # Try JSON first
    json_items = _parse_json_coverage(text)
    if json_items:
        return json_items

    # Try markdown table
    table_items = _parse_markdown_table_coverage(text)
    if table_items:
        return table_items

    # Try reasoning-style output (e.g., **Item c1:** with - **Status:** 4)
    reasoning_items = _parse_reasoning_style_coverage(text)
    if reasoning_items:
        return reasoning_items

    # Try standard block format
    blocks = [block.strip() for block in text.split("---") if block.strip()]
    if len(blocks) == 1:
        blocks = _split_by_checklist_id(text)
    items: list[dict[str, str]] = []
    for block in blocks:
        data: dict[str, str] = {}
        current_field: str | None = None
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            matched = False
            for field in ("CHECKLIST_ID", "STATUS", "EXCERPT", "REASON"):
                matched_value = _parse_field_line(line, field)
                if matched_value is not None:
                    data[field] = matched_value
                    current_field = field
                    matched = True
                    break
            if not matched and current_field is not None:
                data[current_field] = f"{data.get(current_field, '')} {line}".strip()
        checklist_id = _normalize_checklist_id(data.get("CHECKLIST_ID", ""))
        status = _normalize_status(data.get("STATUS", ""))
        reason = data.get("REASON", "").strip()
        if not checklist_id or status not in {"1", "2", "3", "4", "5"} or not reason:
            continue
        items.append(
            {
                "checklist_id": checklist_id,
                "status": status,
                "matched_response_excerpt": data.get("EXCERPT", "").strip(),
                "reason": reason,
            }
        )
    return items


def _parse_reasoning_style_coverage(text: str) -> list[dict[str, str]]:
    """Parse Qwen-style reasoning output with **Item c1:** headers and - **Status:** 4 format."""
    items: list[dict[str, str]] = []

    # Pattern to find item sections like "**Item c1:**" or "**Checklist Item c1:**"
    item_pattern = re.compile(
        r"^\s*\*{1,2}\s*(?:Item\s+|Checklist\s+)?(c\d+|\d+)\s*:?\*{0,2}\s*$",
        re.IGNORECASE | re.MULTILINE
    )

    # Find all item positions
    matches = list(item_pattern.finditer(text))
    if not matches:
        return []

    for i, match in enumerate(matches):
        item_id = _normalize_checklist_id(match.group(1))
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start_pos:end_pos]

        # Look for status in various formats
        status = None
        reason = None
        excerpt = ""

        # Try to find status: "**Status:** 4" or "- **Status:** 4" or "Status: 4"
        status_patterns = [
            r"\*{0,2}\s*Status\s*[:：]\s*\*{0,2}\s*(\d)",
            r"-\s*\*{0,2}\s*Status\s*[:：]\s*\*{0,2}\s*(\d)",
            r"\*{0,2}\s*Score\s*[:：]\s*\*{0,2}\s*(\d)",
            r"Status\s*[=:]\s*(\d)",
        ]
        for pattern in status_patterns:
            s_match = re.search(pattern, section, re.IGNORECASE)
            if s_match:
                status = s_match.group(1)
                break

        # Try to find reason: "**Reason:** ..." or "- **Reason:** ..."
        reason_patterns = [
            r"\*{0,2}\s*Reason\s*[:：]\s*\*{0,2}\s*(.+?)(?=\n\s*\*|$)",
            r"-\s*\*{0,2}\s*Reason\s*[:：]\s*\*{0,2}\s*(.+?)(?=\n\s*\*|$)",
            r"\*{0,2}\s*Rationale\s*[:：]\s*\*{0,2}\s*(.+?)(?=\n\s*\*|$)",
            r"\*{0,2}\s*Explanation\s*[:：]\s*\*{0,2}\s*(.+?)(?=\n\s*\*|$)",
        ]
        for pattern in reason_patterns:
            r_match = re.search(pattern, section, re.IGNORECASE | re.DOTALL)
            if r_match:
                reason = r_match.group(1).strip()
                # Clean up markdown and truncate
                reason = re.sub(r"\*+", "", reason).strip()
                if len(reason) > 200:
                    reason = reason[:197] + "..."
                break

        # Try to find excerpt/quote
        excerpt_patterns = [
            r"\*{0,2}\s*(?:Excerpt|Quote|Evidence)\s*[:：]\s*\*{0,2}\s*(.+?)(?=\n\s*\*|$)",
            r"-\s*\*{0,2}\s*(?:Excerpt|Quote|Evidence)\s*[:：]\s*\*{0,2}\s*(.+?)(?=\n\s*\*|$)",
        ]
        for pattern in excerpt_patterns:
            e_match = re.search(pattern, section, re.IGNORECASE | re.DOTALL)
            if e_match:
                excerpt = e_match.group(1).strip()
                excerpt = re.sub(r"\*+", "", excerpt).strip()
                if len(excerpt) > 320:
                    excerpt = excerpt[:317] + "..."
                break

        # If no explicit fields found, try to extract from analysis text
        if not reason:
            # Look for sentences that seem like conclusions
            conclusion_patterns = [
                r"(?:so|therefore|thus|hence|conclusion|verdict|decision)[:，,\s]+(.+?)(?=\n\n|\n\*|$)",
                r"(?:score|status|rating)[:，,\s]+(?:is\s+)?(\d)[.\s]*(.+?)(?=\n\n|\n\*|$)",
            ]
            for pattern in conclusion_patterns:
                c_match = re.search(pattern, section, re.IGNORECASE | re.DOTALL)
                if c_match:
                    reason = c_match.group(1).strip() if len(c_match.groups()) == 1 else c_match.group(2).strip()
                    reason = re.sub(r"\*+", "", reason).strip()
                    if len(reason) > 200:
                        reason = reason[:197] + "..."
                    break

        # Default reason if status found but no reason
        if status and not reason:
            reason = f"Matched reasoning-style output with status {status}"

        if item_id and status and status in {"1", "2", "3", "4", "5"} and reason:
            items.append({
                "checklist_id": item_id,
                "status": status,
                "matched_response_excerpt": excerpt,
                "reason": reason,
            })

    return items


def _parse_field_line(line: str, field: str) -> str | None:
    cleaned = line.strip().lstrip("-*0123456789. )\t").strip()
    cleaned = cleaned.replace("：", ":")
    cleaned = re.sub(r"^\*{1,2}", "", cleaned)
    cleaned = re.sub(r"\*{1,2}\s*:", ":", cleaned)
    pattern = rf"^{re.escape(field)}\s*:\s*(.*)$"
    matched = re.match(pattern, cleaned, flags=re.IGNORECASE)
    if matched:
        return matched.group(1).strip().strip("`")
    aliases = {
        "CHECKLIST_ID": ("ID", "ITEM", "ITEM_ID", "CHECKLIST ITEM", "CHECKLIST"),
        "STATUS": ("SCORE", "RATING", "COVERAGE", "COVERAGE_STATUS"),
        "EXCERPT": ("QUOTE", "EVIDENCE", "MATCHED_RESPONSE_EXCERPT"),
        "REASON": ("RATIONALE", "EXPLANATION"),
    }
    for alias in aliases.get(field, ()):
        alias_pattern = rf"^{re.escape(alias)}\s*:\s*(.*)$"
        matched = re.match(alias_pattern, cleaned, flags=re.IGNORECASE)
        if matched:
            return matched.group(1).strip().strip("`")
    return None


def _normalize_checklist_id(value: str) -> str:
    text = str(value).strip().strip("`").strip()
    text = re.sub(r"^\s*(checklist|item|id)\s*[-_#:]?\s*", "", text, flags=re.IGNORECASE)
    matched = re.search(r"\bc\s*[-_ ]?\s*(\d+)\b", text, flags=re.IGNORECASE)
    if matched:
        return f"c{int(matched.group(1))}"
    matched = re.search(r"\b(\d+)\b", text)
    if matched:
        return f"c{int(matched.group(1))}"
    return text.lower()


def _normalize_status(value: str) -> str:
    text = str(value).strip()
    matched = re.search(r"\b([1-5])\b", text)
    return matched.group(1) if matched else text


def _parse_json_coverage(text: str) -> list[dict[str, str]]:
    payload = _extract_json_payload(text)
    if payload is None:
        return []
    try:
        loaded = json.loads(payload)
    except ValueError:
        return []
    if isinstance(loaded, dict):
        for key in ("items", "coverage", "results", "checklist_coverage"):
            if isinstance(loaded.get(key), list):
                loaded = loaded[key]
                break
        else:
            loaded = [loaded]
    if not isinstance(loaded, list):
        return []
    out: list[dict[str, str]] = []
    for row in loaded:
        if not isinstance(row, dict):
            continue
        checklist_id = _normalize_checklist_id(
            row.get("checklist_id") or row.get("CHECKLIST_ID") or row.get("id") or row.get("item") or ""
        )
        status = _normalize_status(row.get("status") or row.get("STATUS") or row.get("score") or "")
        reason = str(row.get("reason") or row.get("REASON") or row.get("rationale") or "").strip()
        if not checklist_id or status not in {"1", "2", "3", "4", "5"} or not reason:
            continue
        out.append(
            {
                "checklist_id": checklist_id,
                "status": status,
                "matched_response_excerpt": str(
                    row.get("matched_response_excerpt")
                    or row.get("excerpt")
                    or row.get("EXCERPT")
                    or row.get("quote")
                    or ""
                ).strip(),
                "reason": reason,
            }
        )
    return out


def _extract_json_payload(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    for opener, closer in (("[", "]"), ("{", "}")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if 0 <= start < end:
            return stripped[start : end + 1]
    return None


def _parse_markdown_table_coverage(text: str) -> list[dict[str, str]]:
    rows = [line.strip() for line in text.splitlines() if line.strip().startswith("|") and line.strip().endswith("|")]
    if len(rows) < 2:
        return []
    header = [cell.strip().lower().replace(" ", "_") for cell in rows[0].strip("|").split("|")]
    if not any("status" in h or "score" in h for h in header):
        return []
    out: list[dict[str, str]] = []
    for row_text in rows[2:]:
        cells = [cell.strip() for cell in row_text.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        row = dict(zip(header, cells))
        checklist_id = _normalize_checklist_id(
            row.get("checklist_id") or row.get("id") or row.get("item") or row.get("checklist") or ""
        )
        status = _normalize_status(row.get("status") or row.get("score") or row.get("coverage") or "")
        reason = row.get("reason") or row.get("rationale") or row.get("explanation") or ""
        if not checklist_id or status not in {"1", "2", "3", "4", "5"} or not reason:
            continue
        out.append(
            {
                "checklist_id": checklist_id,
                "status": status,
                "matched_response_excerpt": row.get("excerpt") or row.get("quote") or row.get("evidence") or "",
                "reason": reason,
            }
        )
    return out


def _split_by_checklist_id(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if _parse_field_line(line, "CHECKLIST_ID") is not None and current:
            blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return [b.strip() for b in blocks if b.strip()]


def _truncate(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."
