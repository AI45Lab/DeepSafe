"""Three-grade classification and low-span metrics."""

from collections import Counter

from .prompts import LABELS, unique_span_position


def score(samples, predictions, annotations, field):
    confusion = {label: Counter() for label in LABELS}
    exact, predicted_n, reference_n = 0, 0, 0
    char_tp = char_p = char_r = 0
    for sample in samples:
        sid = str(sample["sample_id"])
        output = predictions.get(sid, {})
        if field == "initial":
            output = output.get("initial", {})
        else:
            output = output.get("final", {})
        if not isinstance(output, dict):
            output = {}
        predicted = output.get("label") if output.get("label") in LABELS else "invalid"
        truth = sample["label"]
        confusion[truth][predicted] += 1
        refs = annotations[sid]["annotations"] if truth == "low" else []
        ref_text = Counter(x["text"] for x in refs)
        reference_n += sum(ref_text.values())
        ref_positions = {i for x in refs for i in range(x["start"], x["end"])}
        char_r += len(ref_positions)
        if field == "initial" or predicted != "low":
            continue  # Agent1 deliberately does not locate spans.
        found = []
        for issue in output.get("issues", []):
            start = unique_span_position(sample["answer"], issue["span"])
            if start is not None:
                found.append((start, start + len(issue["span"]), issue["span"]))
        pred_text = Counter(x[2] for x in found)
        exact += sum((ref_text & pred_text).values())
        predicted_n += sum(pred_text.values())
        pred_positions = {i for start, end, _ in found for i in range(start, end)}
        char_p += len(pred_positions)
        char_tp += len(pred_positions & ref_positions)
    per_grade = {}
    for label in LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[x][label] for x in LABELS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_grade[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    ep = exact / predicted_n if predicted_n else 0.0
    er = exact / reference_n if reference_n else 0.0
    cp = char_tp / char_p if char_p else 0.0
    cr = char_tp / char_r if char_r else 0.0
    return {"total": len(samples), "accuracy": sum(confusion[x][x] for x in LABELS) / len(samples),
            "macro_f1": sum(per_grade[x]["f1"] for x in LABELS) / len(LABELS),
            "per_grade": per_grade, "confusion": {x: dict(confusion[x]) for x in LABELS},
            "low_span_exact_f1": 2 * ep * er / (ep + er) if ep + er else 0.0,
            "low_span_char_f1": 2 * cp * cr / (cp + cr) if cp + cr else 0.0}


