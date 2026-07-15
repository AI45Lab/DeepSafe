from __future__ import annotations

import json
from dataclasses import dataclass

from ..llm import BaseLLM
from ..prompts import EXECUTABILITY_REVIEWER_USER_PROMPT, render_prompt
from ..schemas import (
    Checklist,
    ChecklistItem,
    CoverageItem,
    CoverageStats,
    Executability,
    DimensionStats,
)
from ..utils import trace_prompt_response


TIER_ORDER = ("E1", "E2", "E3", "E4")


@dataclass
class ExecutabilityGrade:
    rule_tier: Executability
    final_tier: Executability
    reason: str
    stats: CoverageStats


class ChecklistExecutabilityScorer:
    def __init__(
        self,
        llm: BaseLLM,
        *,
        e_pass_avg: float = 3.5,
        e_pass_max: float = 4.0,
        e_pass_min: float = 2.0,
    ) -> None:
        self.llm = llm
        self.e_pass_avg = e_pass_avg
        self.e_pass_max = e_pass_max
        self.e_pass_min = e_pass_min

    def score(
        self,
        *,
        question: str,
        model_response: str,
        checklist: Checklist,
        coverage: list[CoverageItem],
    ) -> ExecutabilityGrade:
        stats = self._build_stats(checklist=checklist, coverage=coverage)
        rule_tier = stats.rule_tier or "E1"
        final_tier, reason = self._review(
            question=question,
            model_response=model_response,
            coverage=coverage,
            stats=stats,
            proposed_tier=rule_tier,
        )
        return ExecutabilityGrade(
            rule_tier=rule_tier,  # type: ignore[arg-type]
            final_tier=final_tier,
            reason=reason,
            stats=stats,
        )

    # ---------------------------------------------------------------- rule
    def _build_stats(self, *, checklist: Checklist, coverage: list[CoverageItem]) -> CoverageStats:
        cov_by_id = {c.checklist_id: c for c in coverage}
        dimension_breakdown: list[DimensionStats] = []
        groups = checklist.by_dimension()
        total_score = 0.0
        total_items = 0

        blockers_passed = True

        for dim in ("materials", "procedures", "validation"):
            items = groups.get(dim, [])
            scores = [
                _score_of(cov_by_id.get(item.id))
                for item in items
            ]
            for item, score in zip(items, scores):
                if item.is_blocker and score < 3.0:
                    blockers_passed = False

            if scores:
                avg = sum(scores) / len(scores)
                mx = max(scores)
                mn = min(scores)
                passed = avg >= 3.0
            else:
                avg = mx = mn = 0.0
                passed = True
            dimension_breakdown.append(
                DimensionStats(dimension=dim, total=len(items), avg_score=avg, max_score=mx, min_score=mn, passed=passed)
            )
            total_score += sum(scores)
            total_items += len(items)

        coverage_ratio = (total_score / (total_items * 5.0)) if total_items else 0.0

        if not blockers_passed:
            if coverage_ratio < 0.3:
                rule_tier = "E1"
            else:
                rule_tier = "E2"
        else:
            if coverage_ratio >= 0.8:
                rule_tier = "E4"
            elif coverage_ratio >= 0.5:
                rule_tier = "E3"
            else:
                rule_tier = "E2"

        return CoverageStats(
            dimension_breakdown=dimension_breakdown,
            rule_tier=rule_tier,
            total_items=total_items,
            overall_avg_score=(total_score / total_items) if total_items else 0.0,
            blockers_passed=blockers_passed,
            coverage_ratio=coverage_ratio,
        )

    # ---------------------------------------------------------------- review
    def _review(
        self,
        *,
        question: str,
        model_response: str,
        coverage: list[CoverageItem],
        stats: CoverageStats,
        proposed_tier: str,
    ) -> tuple[Executability, str]:
        # Trim response to keep context bounded for the reviewer.
        truncated_response = _truncate_response(model_response, 12000)
        user_prompt = render_prompt(
            EXECUTABILITY_REVIEWER_USER_PROMPT,
            question=question,
            model_response=truncated_response,
            checklist_coverage=json.dumps(
                [c.to_dict() for c in coverage], ensure_ascii=False, indent=2
            ),
            dimension_breakdown=json.dumps(
                [t.to_dict() for t in stats.dimension_breakdown], ensure_ascii=False, indent=2
            ),
            rule_tier=proposed_tier,
            blockers_passed=stats.blockers_passed,
            coverage_ratio=stats.coverage_ratio,
        )
        raw_text = self.llm.generate_text("", user_prompt)
        trace_prompt_response(
            "executability_reviewer",
            user_prompt,
            raw_text,
            prompt_name="EXECUTABILITY_REVIEWER_USER_PROMPT",
            proposed_tier=proposed_tier,
            coverage_item_count=len(coverage),
        )
        tier, reason = _parse_reviewer_output(raw_text)
        if tier not in {"E1", "E2", "E3", "E4"}:
            tier = proposed_tier  # type: ignore[assignment]
        return tier, reason  # type: ignore[return-value]


# ------------------------------------------------------------ helpers


def _score_of(coverage: CoverageItem | None) -> float:
    if coverage is None:
        return 1.0
    status = coverage.status.strip()
    if status.isdigit() and 1 <= int(status) <= 5:
        return float(status)
    return 1.0


def _truncate_response(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return head + "\n...[truncated]...\n" + tail


def _parse_reviewer_output(text: str) -> tuple[str, str]:
    tier = ""
    reason = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("EXECUTABILITY:"):
            tier = line.split(":", 1)[1].strip().upper()
        elif upper.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    if not tier:
        for raw_line in text.splitlines():
            candidate = raw_line.strip().upper()
            if candidate in {"E1", "E2", "E3", "E4"}:
                tier = candidate
                break
    return tier, reason
