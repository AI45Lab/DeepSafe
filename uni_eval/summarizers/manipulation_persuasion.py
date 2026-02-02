import os
import json
import logging
from typing import Any, Dict, List

from uni_eval.registry import SUMMARIZERS
from uni_eval.summarizers.base import BaseSummarizer

logger = logging.getLogger(__name__)

@SUMMARIZERS.register_module()
class ManipulationPersuasionSummarizer(BaseSummarizer):
    """
    Summarizer for manipulation/persuasion experiments.

    - Saves the full ``results`` dict to ``result.json`` (same as StandardSummarizer).
    - Prints numeric metrics.
    - Writes a Markdown report including:
        - overall metrics
        - per-model persuasion status table, if available.
    """

    def summarize(self, results: Dict[str, Any], output_dir: str) -> None:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, "result.json")
        print(f"Saving full results to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        metrics = results.get("metrics", {}) or {}
        details: List[Dict[str, Any]] = results.get("details", []) or []

        print("\n" + "=" * 50)
        print("Manipulation / Persuasion Evaluation Summary")
        print("=" * 50)
        for k, v in metrics.items():
            if k == "per_topic":
                continue
            print(f"{k}: {v}")

        if "per_topic" in metrics:
            print("\nPer-Topic Results:")
            for topic, topic_data in metrics["per_topic"].items():
                print(f"  - {topic}: Success Rate={topic_data['success_rate']:.4f}, Opinion Shift={topic_data['avg_opinion_shift']:.4f}")
        print("=" * 50 + "\n")

        md_file = os.path.join(output_dir, "report.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# Manipulation / Persuasion Evaluation Report\n\n")

            f.write("## Overall Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            for k, v in metrics.items():
                if k == "per_topic":
                    continue
                try:
                    f.write(f"| {k} | {float(v):.4f} |\n")
                except (TypeError, ValueError):
                                                                      
                    f.write(f"| {k} | {v} |\n")

            if "per_topic" in metrics:
                f.write("\n## Per-Topic Scores\n\n")
                f.write("| Topic | Success Rate | Opinion Shift |\n")
                f.write("| :--- | :--- | :--- |\n")
                for topic, topic_data in metrics["per_topic"].items():
                    f.write(f"| {topic} | {topic_data.get('success_rate', 0):.4f} | {topic_data.get('avg_opinion_shift', 0):.4f} |\n")

            statuses = [(d.get("model"), d.get("persuasion_status")) for d in details]
            statuses = [(m, s) for (m, s) in statuses if m is not None and s is not None]

            if statuses:
                f.write("\n## Per-Model Persuasion Status\n\n")
                f.write("| Model | Status |\n")
                f.write("| :--- | :--- |\n")
                for model_name, status in statuses:
                    f.write(f"| {model_name} | {status} |\n")

            f.write("\n## Configuration\n\n")
            f.write("```json\n")
            json.dump(results.get("config", {}), f, indent=2, ensure_ascii=False, default=str)
            f.write("\n```\n")

        print(f"Report saved to {md_file}")

