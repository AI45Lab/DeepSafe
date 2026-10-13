"""Select reviews by the predicted class probability; never use reference labels."""
import math

DEFAULT_THRESHOLDS = {"high": 0.9, "medium": 0.9, "low": 0.7}


def thresholds(config=None):
    values = (config or {}).get("review_confidence_thresholds", DEFAULT_THRESHOLDS)
    if (not isinstance(values, dict) or set(values) != set(DEFAULT_THRESHOLDS)
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                   for v in values.values())):
        raise ValueError("review_confidence_thresholds requires high/medium/low finite probabilities in [0,1]")
    return dict(values)


def route_initial(initial, config=None):
    levels = thresholds(config)
    label = initial.get("label") if isinstance(initial, dict) else None
    result = {"review_required": False, "initial_confidence": None,
              "review_threshold": levels.get(label), "review_route_reason": "initial_unavailable"}
    if label not in levels:
        return result
    probs = initial.get("probabilities")
    valid = (isinstance(probs, dict) and set(probs) == set(levels)
             and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in probs.values())
             and abs(sum(probs.values()) - 1) <= 0.001
             and probs[label] >= max(probs.values()))
    if not valid:
        result.update(review_required=True, review_route_reason="missing_or_invalid_probability_review")
        return result
    confidence = probs[label]
    required = confidence < levels[label]
    result.update(review_required=required, initial_confidence=confidence,
                  review_route_reason="below_class_threshold" if required else "confidence_sufficient_keep_initial")
    return result


def select_review(checkpoint, sid, config):
    first = checkpoint.get(sid, "initial")
    if not first or first["status"] != "succeeded":
        return False
    route = route_initial(first.get("output"), config)
    if route["review_required"]:
        return True
    if checkpoint.get(sid, "review") is None:
        checkpoint.record(sid, "review", status="skipped", attempt=0, usage={}, routing=route)
    return False
