"""Agent routing, selective localization, and resumable execution."""

import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .client import ask
from .prompts import prompt, valid_label, validated_issues


def usable_issues(raw, answer):
    """Keep up to three distinct, individually valid exact-answer spans."""
    if not isinstance(raw, list):
        return None
    issues = []
    seen = set()
    for item in raw:
        valid = validated_issues([item], answer)
        if valid is not None and valid[0]["span"] not in seen:
            issues.append(valid[0])
            seen.add(valid[0]["span"])
            if len(issues) == 3:
                break
    return issues or None


def process_sample(sample, rules, requester, checkpoint, pause_file, stop_event, attempts,
                   prompt_fn=prompt):
    sid = str(sample["sample_id"])
    stages = dict(checkpoint.events.get(sid, {}))
    if stages.get("final", {}).get("status") in {"complete", "failed"}:
        return stages["final"]

    def stage(name, content, validator):
        old = stages.get(name)
        if old is not None and validator(old.get("output")):
            return old["output"]
        if stop_event.is_set() or pause_file.exists():
            return None
        event = checkpoint.record(sid, name, ask(
            requester, content, attempts, lambda: stop_event.is_set() or pause_file.exists()))
        stages[name] = event
        if event.get("terminal"):
            stages["final"] = checkpoint.record(sid, "final", {
                "status": "failed", "failed_stage": name, "error": event.get("error"),
                "calls": 0, "usage": {}})
        return event["output"] if validator(event["output"]) else None

    first = stage("agent1", prompt_fn(sample, rules, "agent1"), valid_label)
    if first is None:
        return stages.get("final")
    second = stage("agent2", prompt_fn(sample, rules, "agent2", first),
                   lambda x: valid_label(x) and x.get("verdict") in {"supported", "contradicted", "unclear"})
    if second is None:
        return stages.get("final")
    agree = second["verdict"] == "supported" and second["label"] == first["label"]
    if ((first["label"] != "medium" and first["missing_requirement"].strip()) or
            (second["label"] != "medium" and second["missing_requirement"].strip())):
        agree = False
    if agree:
        final_label, final_reason, adjudicated = first["label"], second["reason"], False
    else:
        third = stage("agent3", prompt_fn(sample, rules, "agent3", first, second), valid_label)
        if third is None:
            return stages.get("final")
        final_label, final_reason, adjudicated = third["label"], third["reason"], True
    issues = []
    if final_label == "low":
        direct = usable_issues(second.get("issues"), sample["answer"]) if agree else None
        if direct is not None:
            issues = direct
        else:
            located = stage("locator", prompt_fn(sample, rules, "locator", {"label": final_label, "reason": final_reason}),
                            lambda x: isinstance(x, dict) and usable_issues(x.get("issues"), sample["answer"]) is not None)
            if located is None:
                return stages.get("final")
            issues = usable_issues(located["issues"], sample["answer"])
    return checkpoint.record(sid, "final", {
        "status": "complete", "initial": first, "verification": second,
        "adjudication": stages.get("agent3", {}).get("output") if adjudicated else None,
        "final": {"label": final_label, "reason": final_reason, "issues": issues},
        "route": "agent3" if adjudicated else ("locator" if final_label == "low" and direct is None else "agreement"),
        "calls": 0, "usage": {}, "error": None})


def run_samples(samples, rules, requester, checkpoint, pause_file, workers, attempts,
                prompt_fn=prompt, process_fn=None):
    processor = process_fn or process_sample
    stop = threading.Event()
    completed = {str(s["sample_id"]): checkpoint.events[str(s["sample_id"])]["final"]
                 for s in samples if checkpoint.events.get(str(s["sample_id"]), {}).get("final", {}).get("status") == "complete"}
    failed = {str(s["sample_id"]): checkpoint.events[str(s["sample_id"])]["final"]
              for s in samples if checkpoint.events.get(str(s["sample_id"]), {}).get("final", {}).get("status") == "failed"}
    pending = [s for s in samples if str(s["sample_id"]) not in completed and str(s["sample_id"]) not in failed]
    print("Checkpoint: %d complete, %d failed, %d pending" %
          (len(completed), len(failed), len(pending)), flush=True)
    next_index = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        running = {}

        def submit():
            nonlocal next_index
            while (len(running) < workers and next_index < len(pending) and
                   not pause_file.exists() and not stop.is_set()):
                sample = pending[next_index]
                next_index += 1
                running[pool.submit(processor, sample, rules, requester, checkpoint,
                                    pause_file, stop, attempts, prompt_fn)] = sample

        submit()
        while running:
            try:
                done, _ = wait(running, return_when=FIRST_COMPLETED)
            except KeyboardInterrupt:
                stop.set()
                print("Stopping after in-flight requests; completed stages are saved.", flush=True)
                done, _ = wait(running)
            for future in done:
                sample = running.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    print("Sample %s failed: %s: %s" % (sample["sample_id"], type(exc).__name__, exc), file=sys.stderr)
                    result = None
                if result is not None and result.get("status") == "complete":
                    completed[str(sample["sample_id"])] = result
                    print("Completed %d/%d (id=%s)" % (len(completed), len(samples), sample["sample_id"]), flush=True)
                elif result is not None and result.get("status") == "failed":
                    failed[str(sample["sample_id"])] = result
                    print("Failed permanently (id=%s, stage=%s)" %
                          (sample["sample_id"], result.get("failed_stage")), flush=True)
            submit()
    return completed, failed, len(completed) + len(failed) != len(samples)


