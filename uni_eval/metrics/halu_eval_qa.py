from __future__ import annotations

from typing import Any, Dict, List

from uni_eval.metrics.base import BaseMetric
from uni_eval.registry import METRICS

@METRICS.register_module()
class HaluEvalQAMetric(BaseMetric):
    """
    Match the official-style accounting:
    - invalid/failed parse counts as incorrect
    - report overall accuracy
    """

    def compute(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        total = len(results)
        if total == 0:
            return {
                "accuracy": 0.0,
                "total": 0.0,
                "correct": 0.0,
                "incorrect": 0.0,
                "invalid": 0.0,
            }

        correct = 0
        invalid = 0
        for r in results:
                                               
            if r.get("judgement") == "failed!" or (r.get("judgement_valid") is False):
                invalid += 1
                continue
            if bool(r.get("correct")):
                correct += 1

        incorrect = total - correct
        acc = correct / total if total else 0.0
        return {
            "accuracy": float(acc),
            "total": float(total),
            "correct": float(correct),
            "incorrect": float(incorrect),
            "invalid": float(invalid),
        }

