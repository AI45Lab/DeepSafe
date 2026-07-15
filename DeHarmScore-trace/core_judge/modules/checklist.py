from __future__ import annotations

import json

from ..llm import BaseLLM
from ..prompts import CHECKLIST_GENERATION_USER_PROMPT, render_prompt
from ..schemas import Checklist, ChecklistItem, EvidenceItem, SearchTrace
from ..utils import trace_prompt_response


class ChecklistGenerator:
    """Generate a flat question-specific tiered checklist (E1-E4)."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def generate(
        self,
        *,
        question: str,
        search_queries: list[str],
        search_traces: list[SearchTrace],
        max_checklist_items: int = 10,
    ) -> Checklist:
        evidence_ids, flattened_search = _collect_evidence(search_traces=search_traces)
        user_prompt = render_prompt(
            CHECKLIST_GENERATION_USER_PROMPT,
            question=question,
            search_queries=json.dumps(search_queries, ensure_ascii=False, indent=2),
            search_evidence=json.dumps(flattened_search, ensure_ascii=False, indent=2),
            max_checklist_items=max_checklist_items,
        )
        raw_text = self.llm.generate_text("", user_prompt)
        trace_prompt_response(
            "checklist_generation",
            user_prompt,
            raw_text,
            prompt_name="CHECKLIST_GENERATION_USER_PROMPT",
            search_query_count=len(search_queries),
            search_evidence_count=len(flattened_search),
            max_checklist_items=max_checklist_items,
        )
        items = _parse_checklist_blocks(raw_text, evidence_ids=evidence_ids)
        if not items:
            raise RuntimeError(f"Checklist generator returned no valid items. Raw: {raw_text!r}")
        items = _renumber(items)
        return Checklist(items=items)


def _collect_evidence(*, search_traces: list[SearchTrace]) -> tuple[set[str], list[dict]]:
    evidence_ids: set[str] = set()
    flattened: list[dict] = []
    for trace in search_traces:
        for item in trace.results:
            evidence_ids.add(item.id)
            flattened.append(item.to_dict())
    return evidence_ids, flattened


def _parse_checklist_blocks(text: str, *, evidence_ids: set[str]) -> list[ChecklistItem]:
    blocks = _split_blocks(text)
    items: list[ChecklistItem] = []
    for block in blocks:
        data: dict[str, str] = {}
        current_field: str | None = None
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            matched = False
            for field in ("ITEM", "DIMENSION", "IS_BLOCKER", "REQUIREMENT", "MINIMUM_BAR", "WHY_THIS_DIMENSION", "EVIDENCE_IDS"):
                prefix = f"{field}:"
                if line.startswith(prefix):
                    data[field] = line.split(":", 1)[1].strip()
                    current_field = field
                    matched = True
                    break
            if not matched and current_field is not None:
                data[current_field] = f"{data.get(current_field, '')} {line}".strip()
        requirement = " ".join(data.get("REQUIREMENT", "").split()).strip()
        if not requirement:
            continue
        dimension = data.get("DIMENSION", "procedures").strip().lower()
        if dimension not in {"materials", "procedures", "validation"}:
            dimension = "procedures"
        is_blocker_str = data.get("IS_BLOCKER", "false").strip().lower()
        is_blocker = is_blocker_str == "true"
        parsed_evidence_ids: list[str] = []
        for raw_eid in data.get("EVIDENCE_IDS", "").split(","):
            eid = raw_eid.strip()
            if eid and eid in evidence_ids and eid not in parsed_evidence_ids:
                parsed_evidence_ids.append(eid)
        items.append(
            ChecklistItem(
                id=data.get("ITEM", "").strip() or f"c{len(items) + 1}",
                requirement=requirement,
                dimension=dimension,  # type: ignore[arg-type]
                is_blocker=is_blocker,
                minimum_bar=" ".join(data.get("MINIMUM_BAR", "").split()).strip(),
                rationale=" ".join(data.get("WHY_THIS_DIMENSION", "").split()).strip(),
                evidence_ids=parsed_evidence_ids,
            )
        )
    return items


def _split_blocks(text: str) -> list[str]:
    explicit = [block.strip() for block in text.split("---") if block.strip()]
    if len(explicit) > 1:
        return explicit
    blocks: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("ITEM:") and current:
            blocks.append("\n".join(current).strip())
            current = [line]
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [b for b in blocks if b]


def _renumber(items: list[ChecklistItem]) -> list[ChecklistItem]:
    out: list[ChecklistItem] = []
    for idx, item in enumerate(items, start=1):
        out.append(
            ChecklistItem(
                id=f"c{idx}",
                requirement=item.requirement,
                dimension=item.dimension,
                is_blocker=item.is_blocker,
                minimum_bar=item.minimum_bar,
                rationale=item.rationale,
                evidence_ids=list(item.evidence_ids),
                scope="global",
            )
        )
    return out
