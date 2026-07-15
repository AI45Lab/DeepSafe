from __future__ import annotations

from ..llm import BaseLLM
from ..prompts import QUERY_PLANNER_USER_PROMPT, render_prompt
from ..utils import trace_prompt_response


class SearchQueryGenerator:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def generate(self, question: str, *, max_search_queries: int) -> list[str]:
        user_prompt = render_prompt(
            QUERY_PLANNER_USER_PROMPT,
            question=question,
            max_search_queries=max_search_queries,
        )
        raw_text = self.llm.generate_text("", user_prompt)
        trace_prompt_response(
            "query_planner",
            user_prompt,
            raw_text,
            prompt_name="QUERY_PLANNER_USER_PROMPT",
            max_search_queries=max_search_queries,
        )
        queries = _parse_search_queries(raw_text, limit=max_search_queries)
        if not queries:
            queries = [question.strip()]
        return queries


def _parse_search_queries(text: str, *, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        candidate = ""
        if line.startswith("SEARCH_QUERY:"):
            candidate = line.split(":", 1)[1].strip()
        elif line.startswith("QUERY:"):
            candidate = line.split(":", 1)[1].strip()
        text_value = " ".join(candidate.split()).strip()
        if not text_value:
            continue
        key = text_value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text_value)
        if len(out) >= limit:
            break
    return out
