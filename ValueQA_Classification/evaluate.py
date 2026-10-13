"""Compare frozen initial, independent review, structure-gated trial and final predictions."""
import argparse
from collections import Counter
from pathlib import Path
if __package__:
    from .common import LABELS, digest, load_samples, read_jsonl, write_json, write_jsonl
else:
    from common import LABELS, digest, load_samples, read_jsonl, write_json, write_jsonl


def truths(rows, reference_path=None):
    ids = {str(r["sample_id"]) for r in rows}
    if reference_path:
        references = read_jsonl(reference_path)
        result = {}
        for ref in references:
            sid = str(ref.get("sample_id", ""))
            if sid in result:
                raise ValueError("Duplicate reference ID: " + sid)
            label = ref.get("label", ref.get("reference_level"))
            if label not in LABELS:
                raise ValueError("Invalid reference label: " + sid)
            result[sid] = label
        missing = ids - set(result)
        if missing:
            raise ValueError(f"References missing {len(missing)} sample IDs")
        return {sid: result[sid] for sid in ids}
    return {str(r["sample_id"]): r["label"] for r in rows if r.get("label") in LABELS}


def classification(gold, predictions, key):
    confusion = {label: Counter() for label in LABELS}
    valid, correct = 0, 0
    for sid, truth in gold.items():
        label = predictions.get(sid, {}).get(key)
        label = label if label in LABELS else "invalid"
        confusion[truth][label] += 1
        valid += label != "invalid"
        correct += label == truth
    details = {}
    for label in LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        guessed = sum(confusion[t][label] for t in LABELS)
        p, r = tp / guessed if guessed else 0, tp / support if support else 0
        details[label] = {"precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0, "support": support}
    n = len(gold)
    return {"n": n, "valid": valid, "correct": correct, "acc": correct/n if n else None,
            "macro_f1": sum(v["f1"] for v in details.values())/3 if n else None,
            "per_class": details, "confusion": {k: dict(v) for k,v in confusion.items()}}


def compare(gold, predictions, key):
    counts, transitions = Counter(), {}
    errors = []
    for sid, truth in gold.items():
        row = predictions.get(sid, {})
        initial, final = row.get("initial_label"), row.get(key)
        if initial == final:
            continue
        if initial == truth and final != truth:
            category = "spoiled_correct"
        elif initial != truth and final == truth:
            category = "corrected_error"
        else:
            category = "wrong_to_wrong"
        counts[category] += 1
        direction = f"{initial}->{final}"
        transitions.setdefault(direction, Counter())[category] += 1
        errors.append({"sample_id": sid, "reference_label": truth, "initial_label": initial,
                       "comparison_label": final, "category": category,
                       "decision_reason": row.get("decision_reason")})
    net = counts["corrected_error"] - counts["spoiled_correct"]
    return {"corrected_error": counts["corrected_error"], "spoiled_correct": counts["spoiled_correct"],
            "wrong_to_wrong": counts["wrong_to_wrong"], "net_corrected": net,
            "delta_acc": net / len(gold) if gold else None,
            "transitions": {k: dict(v) for k,v in transitions.items()}}, errors


def report(rows, outputs, reference_path=None):
    predictions = {str(r["sample_id"]): r for r in outputs}
    if len(predictions) != len(outputs):
        raise ValueError("Duplicate prediction ID")
    all_gold = truths(rows, reference_path)
    # A pending, never-started sample is coverage, not an observed model error.
    # All attempted failures remain in the evaluated denominator.
    gold = {sid:truth for sid,truth in all_gold.items() if sid in predictions
            and predictions[sid].get("initial_status") != "pending"}
    review_gold={sid:truth for sid,truth in gold.items()
                 if predictions[sid].get("review_status") in ("running","retryable","succeeded","terminal_failed")}
    summary = {"total_samples": len(rows), "labeled_samples": len(all_gold), "evaluated_samples":len(gold),
               "review_evaluated_samples":len(review_gold),
               "review_pending_samples":sum(r.get("initial_status")=="succeeded" and r.get("review_status")=="pending" for r in outputs),
               "review_skipped_samples":sum(r.get("review_status")=="skipped" for r in outputs),
               "review_required_samples":sum(r.get("review_required") is True for r in outputs),
               "completed_initial": sum(r.get("initial_label") in LABELS for r in outputs),
               "completed_review": sum(r.get("review_status") == "succeeded" for r in outputs),
               "structure_valid_review": sum(r.get("review_structure_valid") is True for r in outputs),
               "accepted_changes": sum(r.get("accepted") is True for r in outputs),
               "scores": {key: classification(review_gold if key=="candidate_label" else gold, predictions, key) for key in
                          ("initial_label", "candidate_label", "experimental_label", "final_label")}}
    summary["review_attempted_scores"]={key:classification(review_gold,predictions,key) for key in
                                       ("initial_label","candidate_label","experimental_label","final_label")}
    summary["experimental_changes"], trial_errors = compare(gold, predictions, "experimental_label")
    summary["final_changes"], final_errors = compare(gold, predictions, "final_label")
    summary["note"] = "Initial/experimental/final scores cover attempted initials. Candidate scores cover attempted reviews, including failures, excluding pending reviews. review_attempted_scores uses one paired review subset. experimental_label is uncalibrated. Labels never enter model prompts."
    return summary, trial_errors, final_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, outputs = load_samples(args.input), read_jsonl(args.predictions)
    known = {str(r["sample_id"]) for r in rows}
    if any(str(o["sample_id"]) not in known for o in outputs):
        raise ValueError("Predictions contain IDs outside this input")
    summary, trial, final = report(rows, outputs, args.references)
    summary["predictions_sha256"] = digest(outputs)
    write_json(args.output_dir / "metrics.json", summary)
    write_jsonl(args.output_dir / "experimental_changes.jsonl", trial)
    write_jsonl(args.output_dir / "final_changes.jsonl", final)
    print(summary["scores"])


if __name__ == "__main__":
    main()
