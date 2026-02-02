from typing import Any, Dict, List, Tuple
from uni_eval.registry import METRICS
                                   
try:
    from uni_eval.metrics.base import BaseMetric
except Exception:
    BaseMetric = object

@METRICS.register_module()
class ArgusDomainMetric(BaseMetric):
    """
    Compute per-domain and overall means for Argus dec_metric/bas_metric.

    Input can be:
      - outputs: Dict with key "details" -> List[items]
      - or directly a List[items]

    Each item must contain:
      - "domain": str
      - "metric": [dec_metric: float, bas_metric: float]
    """
    def compute(self, outputs: Any) -> Dict[str, float]:
        if isinstance(outputs, dict) and "details" in outputs:
            items = outputs["details"]
        else:
            items = outputs

        total_dec, total_bas, total_cnt = 0.0, 0.0, 0
        by_dom: Dict[str, Dict[str, float]] = {}

        for item in items:
            domain = item.get("domain", "Unknown")
            metric = item.get("metric")
            if not isinstance(metric, (list, tuple)) or len(metric) < 2:
                         
                continue
            dec, bas = float(metric[0]), float(metric[1])

            total_dec += dec
            total_bas += bas
            total_cnt += 1

            dom_stats = by_dom.setdefault(domain, {"dec_sum": 0.0, "bas_sum": 0.0, "cnt": 0})
            dom_stats["dec_sum"] += dec
            dom_stats["bas_sum"] += bas
            dom_stats["cnt"] += 1

        metrics: Dict[str, float] = {}
                       
        metrics["argus_overall_dec_mean"] = (total_dec / total_cnt) if total_cnt > 0 else 0.0
        metrics["argus_overall_bas_mean"] = (total_bas / total_cnt) if total_cnt > 0 else 0.0
        metrics["argus_overall_count"] = float(total_cnt)

        for dom, s in by_dom.items():
            if s["cnt"] > 0:
                metrics[f"argus_{dom}_dec_mean"] = s["dec_sum"] / s["cnt"]
                metrics[f"argus_{dom}_bas_mean"] = s["bas_sum"] / s["cnt"]
                metrics[f"argus_{dom}_count"] = float(s["cnt"])

        return metrics