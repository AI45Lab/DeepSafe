"""Question-keyed checklist cache.

The tiered checklist only depends on the question, so we cache it on disk
keyed by sha256(question_text) to avoid regenerating for every response.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..schemas import Checklist, ChecklistItem


class ChecklistCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def key_for(question: str) -> str:
        normalized = " ".join(str(question).split()).strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, question: str) -> tuple[Checklist, list[str]] | None:
        path = self.cache_dir / f"{self.key_for(question)}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
        items_raw = payload.get("items")
        if not isinstance(items_raw, list):
            return None
        items: list[ChecklistItem] = []
        for raw in items_raw:
            if not isinstance(raw, dict):
                continue
            items.append(
                ChecklistItem(
                    id=str(raw.get("id", "")),
                    requirement=str(raw.get("requirement", "")),
                    dimension=str(raw.get("dimension", "procedures")),  # type: ignore[arg-type]
                    is_blocker=bool(raw.get("is_blocker", False)),
                    minimum_bar=str(raw.get("minimum_bar", "")),
                    rationale=str(raw.get("rationale", "")),
                    evidence_ids=list(raw.get("evidence_ids") or []),
                    scope="global",
                )
            )
        if not items:
            return None
        search_queries = [str(q) for q in payload.get("search_queries") or []]
        return Checklist(items=items), search_queries

    def put(self, question: str, checklist: Checklist, search_queries: list[str]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{self.key_for(question)}.json"
        payload = {
            "question": question,
            "search_queries": list(search_queries),
            "items": [item.to_dict() for item in checklist.items],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
