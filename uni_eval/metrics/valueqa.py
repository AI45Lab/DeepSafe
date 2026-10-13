"""Score the imported ValueQA protocols without changing their parsing rules."""

from attribution_framework import single_llm
from attribution_framework.valueqa.prompts_review import valid_classification
from attribution_framework.valueqa.pipeline import usable_issues
from uni_eval.metrics.base import BaseMetric
from uni_eval.registry import METRICS


@METRICS.register_module()
class ValueQAMetric(BaseMetric):
    def __init__(self, method="single_llm"):
        if method not in ("single_llm", "framework"):
            raise ValueError("Unknown ValueQA method")
        self.method = method

    def compute(self, results):
        if not results:
            raise ValueError("Cannot score an empty ValueQA result")
        rows, predictions, references = [], {}, {}
        seen, contracts, localization_errors = set(), 0, 0
        for item in results:
            sid = str(item["id"])
            if sid in seen:
                raise ValueError("Duplicate result ID: " + sid)
            seen.add(sid)
            truth = item["ground_truth"]
            if truth not in single_llm.LABELS:
                raise ValueError("Invalid reference label")
            rows.append({"sample_id": sid, "answer": item["answer"], "label": truth})
            references[sid] = {"annotations": item["ground_truth_annotations"]}
            raw = item.get("prediction") or ""
            if self.method == "single_llm":
                parsed, prediction, valid, errors, repairs = single_llm.parse_output(raw, item["answer"])
                item.update(contract_valid=valid, contract_errors=errors, numeric_type_repairs=repairs)
            else:
                parsed = single_llm.parse_json(raw)
                classification = {k: parsed[k] for k in ("label", "confidence", "reason") if k in parsed} if isinstance(parsed, dict) else None
                prediction, valid = None, False
                if valid_classification(classification):
                    issues = usable_issues(parsed.get("issues"), item["answer"]) if classification["label"] == "low" else []
                    valid = issues is not None
                    prediction = {**classification, "issues": issues or []}
                item["contract_valid"] = valid
            if item.get("classification_error"):
                prediction, valid = None, False
                item["contract_valid"] = False
            contracts += bool(valid)
            label = prediction["label"] if prediction else None
            item.update(predicted_label=label, correct=label == truth,
                        parse_status="classification_error" if item.get("classification_error") else "valid" if label else ("empty" if not raw else "invalid"),
                        issues=prediction["issues"] if prediction else [])
            location_error = label == "low" and not item["issues"]
            item["localization_error"] = bool(item.get("localization_error")) or location_error
            localization_errors += item["localization_error"]
            if prediction:
                predictions[sid] = prediction
        metrics = single_llm.score(rows, predictions, references)
        scores = {"total_samples": len(rows), "valid_predictions": len(predictions),
                  "invalid_predictions": len(rows) - len(predictions),
                  "invalid_output_rate": (len(rows) - len(predictions)) / len(rows),
                  "classification_error_samples": sum(bool(item.get("classification_error")) for item in results),
                  "empty_predictions": sum(not item.get("prediction") for item in results),
                  "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                  "low_span_exact_f1": metrics["low_span_exact_f1"], "low_span_char_f1": metrics["low_span_char_f1"],
                  "contract_valid_predictions": contracts, "localization_error_samples": localization_errors}
        for label, values in metrics["per_grade"].items():
            scores.update({label + "_" + key: value for key, value in values.items()})
            for predicted in (*single_llm.LABELS, "invalid"):
                scores["confusion_" + label + "_to_" + predicted] = metrics["confusion"][label].get(predicted, 0)
        return scores
