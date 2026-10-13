"""Check evidence contracts, then apply a versioned acceptance policy. No model calls."""
if __package__:
    from .common import LABELS, ROOT, digest, file_hash
    from .boundary_contract import validate_boundary_review
else:
    from common import LABELS, ROOT, digest, file_hash
    from boundary_contract import validate_boundary_review

GATE_VERSION = "single_call_boundary_gate_v3"
TRANSITIONS = {f"{a}->{b}" for a in LABELS for b in LABELS if a != b}


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate_review(raw, row):
    return validate_boundary_review(raw, row)


def gate_identity():
    return digest({"version":GATE_VERSION,"source_sha256":file_hash(ROOT/"gate.py"),
                   "contract_sha256":file_hash(ROOT/"boundary_contract.py")})


def validate_policy(policy, engine_identity, expected_gate_identity=None):
    expected_gate_identity=expected_gate_identity or gate_identity()
    if (not isinstance(policy, dict) or policy.get("schema_version") != 2
            or policy.get("engine_identity") != engine_identity
            or policy.get("gate_identity") != expected_gate_identity
            or policy.get("validated") is not True
            or not isinstance(policy.get("enabled_transitions"), list)
            or any(not isinstance(t,str) or t not in TRANSITIONS for t in policy["enabled_transitions"])
            or not nonempty(policy.get("validation_protocol"))
            or not nonempty(policy.get("confirmation_dataset_sha256"))):
        raise ValueError("Calibrated policy requires matching engine/gate identities and recorded independent validation")


def decide(initial, review, row, mode="observe", policy=None):
    if mode not in ("observe", "experimental", "calibrated"):
        raise ValueError("Unknown gate mode")
    label = initial.get("label") if isinstance(initial, dict) else None
    candidate = review.get("label") if isinstance(review, dict) else None
    errors, locations = validate_review(review, row)
    eligible = label in LABELS and candidate in LABELS and not errors
    result = {"initial_label": label, "candidate_label": candidate if candidate in LABELS else None,
              "review_structure_valid": not errors, "structure_errors": errors,
              "evidence_locations": locations, "experimental_label": candidate if eligible else label,
              "final_label": label, "accepted": False, "gate_version": GATE_VERSION, "mode": mode}
    if label not in LABELS:
        result["decision_reason"] = "initial_failed_no_fallback_label"
    elif not eligible:
        result["decision_reason"] = "review_failed_or_evidence_contract_invalid"
    elif candidate == label:
        result["decision_reason"] = "agreement_keep_initial"
    elif mode == "observe":
        result["decision_reason"] = "observation_only_keep_initial"
    elif mode == "experimental" or (policy and f"{label}->{candidate}" in policy.get("enabled_transitions", [])):
        result.update(final_label=candidate, accepted=True,
                      decision_reason="experimental_unvalidated_change" if mode == "experimental" else "policy_allowed_change")
    else:
        result["decision_reason"] = "transition_not_enabled_keep_initial"
    return result
