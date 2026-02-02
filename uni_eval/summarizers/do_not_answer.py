import os
import json
from collections import defaultdict
from typing import Any, Dict, List

from uni_eval.registry import SUMMARIZERS
from uni_eval.summarizers.base import BaseSummarizer

def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"

@SUMMARIZERS.register_module()
class DoNotAnswerSummarizer(BaseSummarizer):
    """
    Writes:
      - result.json (full results)
      - report.md (overall + breakdown tables)
    """

    def summarize(self, results: Dict[str, Any], output_dir: str) -> None:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        out_json = os.path.join(output_dir, "result.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        details: List[Dict[str, Any]] = results.get("details", []) or []
        total = len(details)

        harmful_count = sum(int(d.get("harmful", 0)) for d in details)
        harmless_rate = (total - harmful_count) / total if total else 0.0

        action_counts = defaultdict(int)
        risk_stats = defaultdict(lambda: {"total": 0, "harmful": 0})
        type_stats = defaultdict(lambda: {"total": 0, "harmful": 0})

        for d in details:
            action_counts[int(d.get("action", -1))] += 1
            ra = d.get("risk_area", "Unknown")
            ht = d.get("harm_type", "Unknown")
            risk_stats[ra]["total"] += 1
            risk_stats[ra]["harmful"] += int(d.get("harmful", 0))
            type_stats[ht]["total"] += 1
            type_stats[ht]["harmful"] += int(d.get("harmful", 0))

        harmful_examples = sorted(
            [d for d in details if int(d.get("harmful", 0)) == 1],
            key=lambda x: float(x.get("harmful_confidence", 0.0)),
            reverse=True,
        )[:10]

        md_file = os.path.join(output_dir, "report.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# Do-Not-Answer Evaluation Report\n\n")
            f.write("## Overall\n\n")
            f.write(f"- **total_count**: {total}\n")
            f.write(f"- **harmful_count**: {harmful_count}\n")
            f.write(f"- **harmless_rate**: {_pct(harmless_rate)}\n\n")

            f.write("## Action Distribution (0-5)\n\n")
            f.write("| action | count | rate |\n")
            f.write("| :--- | ---: | ---: |\n")
            for a in range(-1, 6):
                c = action_counts.get(a, 0)
                if c == 0 and a == -1:
                    continue
                f.write(f"| {a} | {c} | {_pct(c / total) if total else '0.00%'} |\n")
            f.write("\n")

            f.write("## Breakdown by risk_area\n\n")
            f.write("| risk_area | count | harmless_rate |\n")
            f.write("| :--- | ---: | ---: |\n")
            for ra, st in sorted(risk_stats.items(), key=lambda x: x[0]):
                t = st["total"]
                h = st["harmful"]
                f.write(f"| {ra} | {t} | {_pct((t - h) / t) if t else '0.00%'} |\n")
            f.write("\n")

            f.write("## Breakdown by types_of_harm\n\n")
            f.write("| harm_type | count | harmless_rate |\n")
            f.write("| :--- | ---: | ---: |\n")
            for ht, st in sorted(type_stats.items(), key=lambda x: x[0]):
                t = st["total"]
                h = st["harmful"]
                f.write(f"| {ht} | {t} | {_pct((t - h) / t) if t else '0.00%'} |\n")
            f.write("\n")

            if harmful_examples:
                f.write("## Top Harmful Examples (by harmful_confidence)\n\n")
                for ex in harmful_examples:
                    prompt = (ex.get("prompt") or "").strip()
                    resp = (ex.get("response") or "").strip()
                    f.write(f"### id={ex.get('id')} | harmful_conf={ex.get('harmful_confidence'):.4f} | action={ex.get('action')}\n\n")
                    f.write("**Prompt**\n\n")
                    f.write("```text\n")
                    f.write(prompt[:2000] + ("\n...[truncated]" if len(prompt) > 2000 else ""))
                    f.write("\n```\n\n")
                    f.write("**Response**\n\n")
                    f.write("```text\n")
                    f.write(resp[:2000] + ("\n...[truncated]" if len(resp) > 2000 else ""))
                    f.write("\n```\n\n")

            f.write("## Configuration\n\n")
            f.write("```json\n")
            json.dump(results.get("config", {}), f, indent=2, ensure_ascii=False, default=str)
            f.write("\n```\n")

        print(f"Report saved to {md_file}")

