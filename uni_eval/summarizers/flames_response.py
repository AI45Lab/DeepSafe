import os
import json
from typing import Any, Dict, List

from uni_eval.registry import SUMMARIZERS
from uni_eval.summarizers.standard import StandardSummarizer

@SUMMARIZERS.register_module()
class FlamesResponseSummarizer(StandardSummarizer):

    def __init__(self, filename: str = "flames_target_responses.jsonl", **kwargs):
        self.filename = filename

    def summarize(self, results: Dict[str, Any], output_dir: str):
        super().summarize(results, output_dir)

        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        details: List[Dict[str, Any]] = results.get("details", []) or []
        out_path = os.path.join(output_dir, self.filename)

        if details:
            with open(out_path, "w", encoding="utf-8") as f:
                for item in details:
                    response = item.get("response", item.get("prediction", "")) or ""
                    
                    rec = {
                        "dimension": item.get("dimension", ""),
                        "prompt": item.get("prompt", ""),
                        "response": response,
                    }
                    if "subcomponent" in item:
                        rec["subcomponent"] = item.get("subcomponent", "")
                    
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"Saved Flames scorer input JSONL to {out_path}")
