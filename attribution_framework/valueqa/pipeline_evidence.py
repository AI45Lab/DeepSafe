"""Agent2 routes disputes to Agent3 for direct adjudication without Agent1 recheck."""

from .adjudication import (PROMOTION_POLICY, classification as adjudication_classification,
                          is_promotion, valid_adjudication, valid_promotion_evidence)
from .client import ask
from .pipeline import usable_issues
from .prompts_evidence import prompt
from .prompts_review import valid_classification, valid_review

BOUNDARY_TRIGGER_POLICY = "agent2_disagreement_only"
ADJUDICATION_POLICY = "agent3_direct_final"


def process_sample(sample, rules, requester, checkpoint, pause_file, stop_event, attempts, prompt_fn=prompt):
    sid = str(sample["sample_id"])
    stages = dict(checkpoint.events.get(sid, {}))
    if stages.get("final", {}).get("status") in {"complete", "failed"}:
        return stages["final"]

    def stage(name, messages, validator):
        old = stages.get(name)
        if old is not None and validator(old.get("output")):
            return old["output"]
        if stop_event.is_set() or pause_file.exists():
            return None
        event = ask(requester, messages, attempts, lambda: stop_event.is_set() or pause_file.exists())
        if event.get("output") is not None and not validator(event["output"]):
            event["error"], event["status"] = "invalid output schema", "invalid"
        event = checkpoint.record(sid, name, event)
        stages[name] = event
        if event.get("terminal"):
            stages["final"] = checkpoint.record(sid, "final", {
                "status": "failed", "failed_stage": name, "error": event.get("error"), "calls": 0, "usage": {}})
        return event["output"] if validator(event.get("output")) else None

    first = stage("agent1", prompt_fn(sample, rules, "agent1"), valid_classification)
    if first is None:
        return stages.get("final")
    second = stage("agent2", prompt_fn(sample, rules, "agent2", first), valid_review)
    if second is None:
        return stages.get("final")
    triggers = []
    if second["verdict"] in ("contradicted", "unclear"):
        triggers.append("agent2_" + second["verdict"])
    third = None
    classification, guard = first, "not_triggered"
    decision_source = "agent1"
    if triggers:
        third = stage("agent3", prompt_fn(sample, rules, "agent3", first, second), valid_adjudication)
        if third is None:
            return stages.get("final")
        if is_promotion(third, first) and not valid_promotion_evidence(third, sample, first):
            guard = "promotion_blocked_insufficient_evidence"
        else:
            classification = adjudication_classification(third)
            decision_source = "agent3"
            guard = "changed_by_agent3" if third["label"] != first["label"] else "confirmed_by_agent3"
    issues, used_locator = [], False
    if classification["label"] == "low":
        unchanged = classification["label"] == first["label"] and second["verdict"] == "supported"
        issues = usable_issues(second["issues"], sample["answer"]) if unchanged else None
        if issues is None:
            located = stage("locator", prompt_fn(sample, rules, "locator", classification),
                            lambda raw: isinstance(raw, dict) and set(raw) == {"issues"}
                            and usable_issues(raw["issues"], sample["answer"]) is not None)
            if located is None:
                return stages.get("final")
            issues, used_locator = usable_issues(located["issues"], sample["answer"]), True
    return checkpoint.record(sid, "final", {
        "status": "complete", "initial": first, "verification": second,
        "adjudication": third, "recheck": None, "final": {**classification, "issues": issues},
        "boundary_triggers": triggers, "change_guard": guard, "decision_source": decision_source,
        "adjudication_policy": ADJUDICATION_POLICY, "promotion_policy": PROMOTION_POLICY,
        "route": "agent3_direct" if triggers else ("locator" if used_locator else "agreement"),
        "calls": 0, "usage": {}, "error": None})
