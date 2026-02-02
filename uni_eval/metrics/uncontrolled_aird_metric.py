from collections import defaultdict
from typing import Any, Dict, List

from uni_eval.metrics.base import BaseMetric
from uni_eval.registry import METRICS

@METRICS.register_module()
class UncontrolledAIRDMetric(BaseMetric):
    """
    Minimal diagnostics metric for multi-turn conversation runs.
    This is intentionally lightweight; real safety metrics can be added later.
    """

    def compute(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        total = len(results)
        if total == 0:
            return {
                "num_samples": 0.0,
                "final_nonempty_rate": 0.0,
                "avg_turns": 0.0,
            }

        nonempty = 0
        total_turns = 0

        by_cat = defaultdict(lambda: {"n": 0, "nonempty": 0, "turns": 0})

        for r in results:
            pred = (r.get("prediction") or "").strip()
            turns = r.get("turn_predictions") or []
            meta = r.get("meta") or {}
            cat = meta.get("category") or "Unknown"

            if pred:
                nonempty += 1
                by_cat[cat]["nonempty"] += 1

            total_turns += len(turns)
            by_cat[cat]["turns"] += len(turns)
            by_cat[cat]["n"] += 1

        out: Dict[str, float] = {
            "num_samples": float(total),
            "final_nonempty_rate": float(nonempty) / float(total),
            "avg_turns": float(total_turns) / float(total),
        }

        for cat, s in by_cat.items():
            n = max(int(s["n"]), 1)
            out[f"by_category.{cat}.final_nonempty_rate"] = float(s["nonempty"]) / float(n)
            out[f"by_category.{cat}.avg_turns"] = float(s["turns"]) / float(n)

        return out

