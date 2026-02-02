from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from uni_eval.metrics.base import BaseMetric
from uni_eval.registry import METRICS

def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0

def _prf(true: List[int], pred: List[int]) -> Tuple[float, float, float]:
    """
    Binary classification PRF for label=1 positive.
    """
    tp = sum(1 for t, p in zip(true, pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true, pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true, pred) if t == 1 and p == 0)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1

@METRICS.register_module()
class MedHalluDetectionMetric(BaseMetric):
    """
    Match author-style metrics:
    - Parse pred_int (0/1/2) already done in evaluator
    - Exclude not_sure (2) from PRF and accuracy calculations
    - Track percent_of_time_not_sure_chosen overall and per-group

    Framework requirement:
    - Provide subclass results by hallucination category.
    """

    def compute(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        total = len(results)
        if total == 0:
            return {"overall_percent_of_time_not_sure_chosen": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        not_sure_total = sum(1 for r in results if int(r.get("pred_int", 0)) == 2)
        valid = [r for r in results if int(r.get("pred_int", 0)) != 2]
        true = [int(r.get("ground_truth", 0)) for r in valid]
        pred = [int(r.get("pred_int", 0)) for r in valid]

        precision, recall, f1 = _prf(true, pred) if valid else (0.0, 0.0, 0.0)
        acc = _safe_div(sum(1 for t, p in zip(true, pred) if t == p), len(true)) if valid else 0.0
        metrics: Dict[str, float] = {
            "total": float(total),
            "valid_total": float(len(valid)),
            "overall_percent_of_time_not_sure_chosen": float(_safe_div(not_sure_total, total)),
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        for diff in ["easy", "medium", "hard"]:
            group = [r for r in results if ((r.get("meta") or {}).get("difficulty") or "").strip().lower() == diff]
            if not group:
                continue
            ns = sum(1 for r in group if int(r.get("pred_int", 0)) == 2)
            gvalid = [r for r in group if int(r.get("pred_int", 0)) != 2]
            gtrue = [int(r.get("ground_truth", 0)) for r in gvalid]
            gpred = [int(r.get("pred_int", 0)) for r in gvalid]
            gp, gr, gf1 = _prf(gtrue, gpred) if gvalid else (0.0, 0.0, 0.0)
            gacc = _safe_div(sum(1 for t, p in zip(gtrue, gpred) if t == p), len(gtrue)) if gvalid else 0.0
            metrics[f"{diff}_percent_of_time_not_sure_chosen"] = float(_safe_div(ns, len(group)))
            metrics[f"{diff}_accuracy"] = float(gacc)
            metrics[f"{diff}_precision"] = float(gp)
            metrics[f"{diff}_recall"] = float(gr)
            metrics[f"{diff}_f1"] = float(gf1)

        by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in results:
            cat = ((r.get("meta") or {}).get("category") or "").strip()
            if cat:
                by_cat[cat].append(r)

        for cat, group in by_cat.items():
            ns = sum(1 for r in group if int(r.get("pred_int", 0)) == 2)
            gvalid = [r for r in group if int(r.get("pred_int", 0)) != 2]
            gtrue = [int(r.get("ground_truth", 0)) for r in gvalid]
            gpred = [int(r.get("pred_int", 0)) for r in gvalid]
            gp, gr, gf1 = _prf(gtrue, gpred) if gvalid else (0.0, 0.0, 0.0)
            gacc = _safe_div(sum(1 for t, p in zip(gtrue, gpred) if t == p), len(gtrue)) if gvalid else 0.0

            prefix = f"category::{cat}"
            metrics[f"{prefix}::total"] = float(len(group))
            metrics[f"{prefix}::valid_total"] = float(len(gvalid))
            metrics[f"{prefix}::not_sure_rate"] = float(_safe_div(ns, len(group)))
            metrics[f"{prefix}::accuracy"] = float(gacc)
            metrics[f"{prefix}::precision"] = float(gp)
            metrics[f"{prefix}::recall"] = float(gr)
            metrics[f"{prefix}::f1"] = float(gf1)

        return metrics

