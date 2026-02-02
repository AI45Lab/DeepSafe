import os
import json
import logging
import hashlib
from typing import Dict, Any, Set
from uni_eval.registry import SUMMARIZERS
from uni_eval.summarizers.base import BaseSummarizer

logger = logging.getLogger(__name__)

def _json_sanitize(obj: Any, *, _seen: Set[int] | None = None) -> Any:
                                                                      
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
class StandardSummarizer(BaseSummarizer):

    def _format_metric_value(self, v: Any) -> str:

        if isinstance(v, bool):
            return str(v)
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            return f"{v:.4f}"
                                                                   
        try:
            fv = float(v)                          
            if fv.is_integer():
                return str(int(fv))
            return f"{fv:.4f}"
        except Exception:
            return str(v)

    def summarize(self, results: Dict[str, Any], output_dir: str):
        output_dir = os.path.abspath(output_dir)
        output_file = os.path.join(output_dir, 'result.json')
        print(f"Saving full results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            safe_results = _json_sanitize(results)
            json.dump(safe_results, f, indent=2, ensure_ascii=False)
            
        metrics = results.get('metrics', {})
        print("\n" + "="*50)
        print("Evaluation Results Summary")
        print("="*50)
        for k, v in metrics.items():
            print(f"{k}: {self._format_metric_value(v)}")
        print("="*50 + "\n")
        
        md_file = os.path.join(output_dir, 'report.md')
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# Evaluation Report\n\n")
            f.write(f"## Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            for k, v in metrics.items():
                f.write(f"| {k} | {self._format_metric_value(v)} |\n")
            
            f.write(f"\n## Configuration\n\n")
            f.write("```json\n")
            json.dump(results.get('config', {}), f, indent=2, default=str)
            f.write("\n```\n")
            
        print(f"Report saved to {md_file}")

