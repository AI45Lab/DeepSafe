"""Paired classification metrics; evidence quotations are not span localization predictions."""
from valueqa_classification_review.evaluate import report
from uni_eval.metrics.base import BaseMetric
from uni_eval.registry import METRICS


@METRICS.register_module()
class ValueQAClassificationMetric(BaseMetric):
    def compute(self, results):
        rows = [{"sample_id": str(item["id"]), "label": item["ground_truth"]} for item in results]
        outputs = [item["attribution"] for item in results]
        summary, _, _ = report(rows, outputs)
        flat = {key: summary[key] for key in (
            "total_samples", "labeled_samples", "evaluated_samples", "review_evaluated_samples",
            "review_pending_samples", "review_skipped_samples", "review_required_samples", "completed_initial", "completed_review",
            "structure_valid_review", "accepted_changes")}
        for group in ("scores", "review_attempted_scores"):
            for key, scores in summary[group].items():
                name = key.removesuffix("_label")
                prefix = name if group == "scores" else "paired_review_" + name
                for stat in ("n", "valid", "acc", "macro_f1"):
                    flat[prefix + "_" + stat] = scores[stat]
        for group in ("experimental_changes", "final_changes"):
            flat.update({group + "_" + stat: summary[group][stat] for stat in (
                "corrected_error", "spoiled_correct", "wrong_to_wrong", "net_corrected", "delta_acc")})
        return flat
