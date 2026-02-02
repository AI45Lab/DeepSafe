import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, DefaultDict, List
from collections import defaultdict

from uni_eval.registry import SUMMARIZERS
from uni_eval.summarizers.base import BaseSummarizer

logger = logging.getLogger(__name__)

def _json_sanitize(obj: Any, *, _seen: set = None) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return "<circular_ref>"
    _seen = _seen.copy()
    _seen.add(oid)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = k if isinstance(k, str) else str(k)
            out[key] = _json_sanitize(v, _seen=_seen)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize(v, _seen=_seen) for v in obj]
    return str(obj)

@SUMMARIZERS.register_module()
class MOSSBenchSummarizer(BaseSummarizer):
    """
    Summarizer for MOSSBench evaluation results.

    从 details 中读取每条数据的 score 和类型信息，进行类型统计。
    """

    def _extract_type(self, item: Dict[str, Any]) -> str:
        """
        从 item 中提取类型信息。
        
        尝试从以下位置提取：
        item["meta"]["meta_data_over"] (主要来源)
        
        Returns:
            类型字符串，如果无法提取则返回 "unknown"
        """
        meta = item.get("meta", {})
        if isinstance(meta, dict):
                                   
            meta_data_over = meta.get("meta_data_over", None)
            if meta_data_over is not None and str(meta_data_over).strip() != "":
                type_str = str(meta_data_over).strip()
                           
                if type_str in ["Exaggerated Risk", "Negated Harm", "Counterintuitive Interpretation"]:
                    return type_str
        return "unknown"

    def summarize(self, results: Dict[str, Any], output_dir: str):
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, "result.json")
        logger.info(f"Saving full results to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            safe_results = _json_sanitize(results)
            json.dump(safe_results, f, indent=2, ensure_ascii=False)

        metrics = results.get("metrics", {})
        overall_refusal_rate = metrics.get("RefusalRate", None)
        overall_total = metrics.get("Total", 0)

        details: List[Dict[str, Any]] = results.get("details", []) or []
        
        type_name_map = {
            "Exaggerated Risk": "夸大风险",
            "Negated Harm": "否定危害",
            "Counterintuitive Interpretation": "反直觉解释",
        }

        type_stats: DefaultDict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "refusal": 0}
        )
        
        for item in details:
                                              
            score = item.get("score", None)
            if score is None:
                continue

            type_key = self._extract_type(item)
            type_stats[type_key]["total"] += 1
            
            if score == 1:
                type_stats[type_key]["refusal"] += 1

        type_refusal_rates = {}
        type_counts = {}
        for type_key, stats in type_stats.items():
            denom = stats["total"]
            if denom > 0:
                type_refusal_rates[type_key] = stats["refusal"] / denom
            type_counts[type_key] = stats["total"]

        print("\n" + "=" * 70)
        print("MOSSBench Evaluation Results Summary")
        print("=" * 70)

        print("\n## Refusal Rate (拒绝率)")
        print("-" * 70)
        if overall_refusal_rate is not None:
            print(f"  总拒绝率: {overall_refusal_rate * 100:.2f}%")
        if overall_total is not None:
            print(f"  总数量: {overall_total}")

        if type_refusal_rates:
            print("\n## 各类型拒绝率")
            print("-" * 70)
            for eng_name, cn_name in type_name_map.items():
                if eng_name in type_refusal_rates:
                    refusal_rate = type_refusal_rates[eng_name]
                    print(f"  {cn_name}: {refusal_rate * 100:.2f}%")

        print("\n## 中文类型统计")
        print("-" * 70)
        for eng_name, cn_name in type_name_map.items():
            count = type_stats.get(eng_name, {}).get("refusal", 0)
            print(f"  {cn_name}: {count}")

        print("\n" + "=" * 70 + "\n")

        md_file = os.path.join(output_dir, "report.md")
        logger.info(f"Generating Markdown report at {md_file}...")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# MOSSBench Evaluation Report\n\n")
            
            f.write("## Refusal Rate (拒绝率)\n\n")
            f.write("| 指标 | 值 |\n")
            f.write("| :--- | :--- |\n")
            if overall_refusal_rate is not None:
                f.write(f"| 总拒绝率 | {overall_refusal_rate * 100:.2f}% |\n")
            if overall_total is not None:
                f.write(f"| 总数量 | {overall_total} |\n")
            f.write("\n")

            if type_refusal_rates:
                f.write("## 各类型拒绝率\n\n")
                f.write("| 类型 | 拒绝率 |\n")
                f.write("| :--- | :--- |\n")
                for eng_name, cn_name in type_name_map.items():
                    if eng_name in type_refusal_rates:
                        refusal_rate = type_refusal_rates[eng_name]
                        f.write(f"| {cn_name} | {refusal_rate * 100:.2f}% |\n")
                f.write("\n")

            f.write("## 中文类型统计\n\n")
            f.write("| 类型 | 数量 |\n")
            f.write("| :--- | :--- |\n")
            for eng_name, cn_name in type_name_map.items():
                count = type_stats.get(eng_name, {}).get("refusal", 0)
                f.write(f"| {cn_name} | {count} |\n")
            f.write("\n")

            f.write("## Configuration\n\n")
            f.write("```json\n")
            json.dump(results.get("config", {}), f, indent=2, default=str)
            f.write("\n```\n")

        logger.info(f"Report saved to {md_file}")