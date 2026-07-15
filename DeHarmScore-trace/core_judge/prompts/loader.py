from __future__ import annotations

import re
from pathlib import Path

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _load(filename: str) -> str:
    return (_TEMPLATE_DIR / filename).read_text(encoding="utf-8").strip()


def render_prompt(template: str, **values: object) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return str(values[key]) if key in values else match.group(0)

    return _PLACEHOLDER_RE.sub(replace, template)


QUERY_PLANNER_USER_PROMPT = _load("query_planner.md")
CHECKLIST_GENERATION_USER_PROMPT = _load("checklist_global_generator.md")
CHECKLIST_MATCHING_USER_PROMPT = _load("checklist_flat_matcher_v2.md")
CHECKLIST_RECHECK_USER_PROMPT = _load("checklist_low_score_recheck.md")
EXECUTABILITY_REVIEWER_USER_PROMPT = _load("executability_reviewer.md")
CLAIM_EXTRACTOR_USER_PROMPT = _load("claim_extractor.md")
CLAIM_NOVELTY_GRADER_USER_PROMPT = _load("claim_novelty_grader.md")
NET_NEW_REVIEWER_USER_PROMPT = _load("net_new_reviewer.md")
