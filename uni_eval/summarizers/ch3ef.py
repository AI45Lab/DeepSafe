import os
import json
import logging
from typing import Dict, Any, DefaultDict
from collections import defaultdict
from uni_eval.registry import SUMMARIZERS
from uni_eval.summarizers.base import BaseSummarizer

logger = logging.getLogger(__name__)

CHINESE_MAPPING = {
    "Discrimination": "歧视与偏见",
    "Toxicity": "恶毒语言",
    "Illegal Activities": "非法行为",
    "Harm": "伤害",
    "Privacy": "隐私"
}

TARGET_SOURCES = ["Discrimination", "Toxicity", "Illegal Activities", "Harm", "Privacy"]

SOURCE_TO_TARGET_MAPPING = {
                            
    "RacialDiscrimination": "Discrimination",
    "Racial": "Discrimination",
    "Gender": "Discrimination",
    "Age": "Discrimination",
    "Disability": "Discrimination",
    "Religious": "Discrimination",
                     
    "Insult": "Toxicity",
    "Harass": "Toxicity",
    "Threaten": "Toxicity",
                               
    "Other Crimes": "Illegal Activities",
    "Traffic Crime": "Illegal Activities",
    "Theft": "Illegal Activities",
    "drug": "Illegal Activities",
    "Drug": "Illegal Activities",
    "theft": "Illegal Activities",
               
    "Physical Harm": "Harm",
    "Psychological Harm": "Harm",
                  
    "Inferring Personal Information": "Privacy",
    "Leaking Private Information": "Privacy",
    "Location": "Privacy",
}

def _json_sanitize(obj: Any, *, _seen: set = None) -> Any:
    """Sanitize object for JSON serialization, handling circular references."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if _seen is None:
        _seen = set()

    oid = id(obj)
    if oid in _seen:
        return "<circular_ref>"
    
    _seen = _seen.copy()
    _seen.add(oid)

    if isinstance(obj, (bytes, bytearray, memoryview)):
        import hashlib
        b = bytes(obj)
        sha = hashlib.sha256(b).hexdigest()[:16]
        return {"__type__": "bytes", "len": len(b), "sha256_16": sha}
    
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
class Ch3EfSummarizer(BaseSummarizer):
    def summarize(self, results: Dict[str, Any], output_dir: str):
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        result_items = results.get('results', [])
        correct_count = sum(1 for item in result_items if item.get('is_correct', False))
        total_count = len(result_items)
        acc = correct_count / total_count if total_count > 0 else 0.0
        logger.info(f"Ch3EfEvaluator Metrics - ACC: {acc:.4f}")

        metrics = results.get('metrics', {})
        if 'ACC' not in metrics:
            metrics['ACC'] = acc
            results['metrics'] = metrics

        output_file = os.path.join(output_dir, 'result.json')
        logger.info(f"Saving full results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            safe_results = _json_sanitize(results)
            json.dump(safe_results, f, indent=2, ensure_ascii=False)

        metrics = results.get('metrics', {})

        overall_metrics = defaultdict(lambda: 0)
        
        for key, value in metrics.items():
            for source_name, target in SOURCE_TO_TARGET_MAPPING.items():
                if f"Source_{source_name}_ACC" == key:
                    overall_metrics[target] += value

        acc_value=metrics["ACC"]
        print(f"  准确率: {acc_value*100:.2f}%")

        print("\n## 中文源指标统计（准确率）")
        print("-" * 70)
        for target in sorted(TARGET_SOURCES):
            source_display = CHINESE_MAPPING.get(target, target)
            acc_value = overall_metrics[target]
            print(f"  {source_display}: {acc_value*100:.2f}%")
        print("\n" + "="*70 + "\n")

        md_file = os.path.join(output_dir, 'report.md')
        logger.info(f"Generating Markdown report at {md_file}...")
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# Ch3Ef Evaluation Report\n\n")

            f.write("## 中文指标统计\n\n")
            f.write("| 指标 | 数值 |\n")
            f.write("| :--- | :--- |\n")
            if 'ACC' in metrics:
                acc_value = metrics['ACC']
                f.write(f"| 准确率 | {acc_value*100:.2f}% |\n")

            f.write("\n## 中文源指标统计（准确率）\n\n")
            f.write("| 源类别 | 准确率 |\n")
            f.write("| :--- | :--- |\n")
                         
            for target in sorted(TARGET_SOURCES):
                acc_value = overall_metrics.get(target, 0)
                source_display = CHINESE_MAPPING.get(target, target)
                f.write(f"| {source_display} | {acc_value*100:.2f}% |\n")
            f.write("\n")
        
        logger.info(f"Report saved to {md_file}")

