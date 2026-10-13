"""Structural and citation checks for task-anchored semantic certificates.

These checks cannot prove entailment or task necessity; semantic claims remain
reviewer evidence, and automatic deployment still needs an independent policy.
"""
if __package__:
    from .common import LABELS, blind, digest
else:
    from common import LABELS, blind, digest

CORE_ERRORS = {"wrong_object", "reversed_stance", "nonliteral_misreading",
               "wrong_attribution", "key_relation_error", "unsupported_core", "value_misreading"}
LOCAL_DEFECTS = {"required_omission", "required_explanation", "required_judgment", "required_limitation"}


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def check_quotes(items, row, allowed, name, errors, locations, required=True):
    if not isinstance(items, list) or (required and not items):
        errors.append(name + ": missing evidence list")
        return
    for quote in items:
        if (not isinstance(quote, dict) or not isinstance(quote.get("field"), str)
                or quote["field"] not in allowed or not nonempty(quote.get("text"))):
            errors.append(name + ": invalid quote")
            continue
        field, text = quote["field"], quote["text"]
        content = row.get(field)
        start = content.find(text) if isinstance(content, str) else -1
        if start < 0:
            errors.append(name + ": quote does not exist in " + field)
        else:
            locations.append({"field": field, "text": text, "start": start, "end": start + len(text)})


def validate_requirements(plan, row):
    errors, locations = [], []
    if not isinstance(plan, dict):
        return ["requirements_not_json_object"], locations
    if plan.get("requirements_complete") is not True:
        errors.append("incomplete_requirements_plan")
    if type(plan.get("material_answerable")) is not bool or not isinstance(plan.get("ambiguity_note"), str):
        errors.append("invalid_material_answerability")
    items = plan.get("requirements")
    if not isinstance(items, list) or not items:
        return sorted(set(errors + ["empty_requirements_plan"])), locations
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            errors.append("invalid_planned_requirement")
            continue
        rid = item.get("id")
        if not nonempty(rid) or rid in seen:
            errors.append("invalid_or_duplicate_planned_id")
        else:
            seen.add(rid)
        for key in ("point", "necessity_reason", "acceptable_completion"):
            if not nonempty(item.get(key)):
                errors.append("planned_requirement_missing_" + key)
        check_quotes(item.get("question_evidence"), row, {"question"}, "plan.question", errors, locations)
        check_quotes(item.get("material_evidence"), row, {"original_context", "target_comment"},
                     "plan.material", errors, locations)
    return sorted(set(errors)), locations


def bind_single_review(raw, row):
    """Bind a single-call diagnosis; do not claim its requirements are answer-blind."""
    plan = raw.get("requirements_plan")
    value = dict(raw)
    value.update(review_schema_version=3, review_mode="single_call", requirements_plan=plan,
                 requirements_plan_sha256=digest(plan), input_sha256=digest(blind(row)),
                 requirements_answer_blind=False)
    return value


def validate_boundary_review(raw, row):
    errors, locations = [], []
    if not isinstance(raw, dict):
        return ["review_not_json_object"], locations
    plan = raw.get("requirements_plan")
    plan_errors, plan_locations = validate_requirements(plan, row)
    errors.extend(plan_errors)
    locations.extend(plan_locations)
    if (raw.get("review_schema_version") != 3 or raw.get("review_mode") != "single_call"
            or raw.get("requirements_answer_blind") is not False
            or raw.get("input_sha256") != digest(blind(row))
            or raw.get("requirements_plan_sha256") != digest(plan)):
        errors.append("missing_or_mismatched_single_call_binding")
    label = raw.get("label")
    if label not in LABELS or not nonempty(raw.get("reason")):
        errors.append("invalid_label_or_reason")
    for key in ("review_complete", "requirements_complete", "residual_core_check_complete",
                "full_answer_checked", "reasonable_entailment_checked"):
        if raw.get(key) is not True:
            errors.append("incomplete_" + key)
    for key in ("key_claims_supported", "substantive_correct_understanding"):
        if type(raw.get(key)) is not bool:
            errors.append("invalid_" + key)
    core = raw.get("core")
    if not isinstance(core, dict):
        core = {}
    if core.get("status") not in ("correct", "incorrect", "no_understanding") or not nonempty(core.get("reason")):
        errors.append("uncertain_or_invalid_core")
    check_quotes(core.get("answer_evidence"), row, {"answer"}, "core.answer", errors, locations)
    check_quotes(core.get("material_evidence"), row, {"original_context", "target_comment"},
                 "core.material", errors, locations)
    plan_items = plan.get("requirements") if isinstance(plan, dict) else None
    planned = {p["id"]: p for p in (plan_items if isinstance(plan_items, list) else [])
               if isinstance(p, dict) and nonempty(p.get("id"))}
    requirements = raw.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        errors.append("empty_requirements")
        requirements = []
    by_id = {}
    for item in requirements:
        if not isinstance(item, dict):
            errors.append("invalid_requirement")
            continue
        rid, status = item.get("id"), item.get("status")
        if not nonempty(rid) or rid in by_id or rid not in planned:
            errors.append("invalid_duplicate_or_unplanned_requirement")
        else:
            by_id[rid] = item
        if status not in ("met", "partial", "missing", "contradicted") or not nonempty(item.get("reason")):
            errors.append("uncertain_or_invalid_requirement_status")
        check_quotes(item.get("answer_evidence"), row, {"answer"}, "requirement.answer",
                     errors, locations, required=status != "missing")
    if set(by_id) != set(planned):
        errors.append("requirements_not_fully_checked")
    issues = raw.get("issues")
    if not isinstance(issues, list):
        errors.append("invalid_issue_list")
        issues = []
    kinds, local_ids, seen = [], set(), set()
    for issue in issues:
        if not isinstance(issue, dict):
            errors.append("invalid_issue")
            continue
        rid, kind, defect = issue.get("requirement_id"), issue.get("kind"), issue.get("defect_type")
        valid_id = isinstance(rid, str) and rid in by_id
        if not valid_id or kind not in ("central", "local", "nonresponse"):
            errors.append("issue_without_requirement_or_kind")
        if issue.get("status") != "confirmed" or not nonempty(issue.get("reason")) or not nonempty(issue.get("necessity_reason")):
            errors.append("unconfirmed_issue_or_missing_necessity")
        if isinstance(rid, str) and isinstance(kind, str):
            if (rid, kind) in seen:
                errors.append("duplicate_issue")
            seen.add((rid, kind))
        kinds.append(kind)
        missing = nonempty(issue.get("missing_point"))
        check_quotes(issue.get("answer_evidence"), row, {"answer"}, "issue.answer", errors, locations,
                     required=kind != "local" or not missing)
        check_quotes(issue.get("material_evidence"), row, {"original_context", "target_comment"},
                     "issue.material", errors, locations)
        status = by_id.get(rid, {}).get("status") if valid_id else None
        if kind == "local":
            if (not isinstance(defect, str) or defect not in LOCAL_DEFECTS
                    or status not in ("partial", "missing") or issue.get("preserves_core") is not True):
                errors.append("local_issue_without_preservable_necessary_gap")
            if status == "missing" and not missing:
                errors.append("missing_required_gap_description")
            check_quotes(issue.get("correct_understanding_evidence"), row, {"answer"},
                         "issue.correct_understanding", errors, locations)
            if valid_id:
                local_ids.add(rid)
        elif kind == "central":
            if defect != "core_misreading" or issue.get("preserves_core") is not False or status == "met":
                errors.append("central_issue_contract_conflict")
        elif kind == "nonresponse":
            if defect != "no_effective_answer" or issue.get("preserves_core") is not False or status == "met":
                errors.append("nonresponse_issue_contract_conflict")
    repair = raw.get("repair")
    if not isinstance(repair, dict) or not nonempty(repair.get("action")):
        errors.append("missing_semantic_repair_diagnosis")
        repair = {}
    substantive = raw.get("substantive_correct_understanding") is True
    if label == "high":
        if (core.get("status") != "correct" or core.get("error_type") != "none" or not substantive
                or raw.get("key_claims_supported") is not True or issues or repair.get("mode") != "none"
                or any(r.get("status") != "met" for r in by_id.values())):
            errors.append("high_contract_conflict")
    elif label == "medium":
        gaps = {rid for rid, r in by_id.items() if r.get("status") in ("partial", "missing")}
        if (core.get("status") != "correct" or core.get("error_type") != "none" or not substantive
                or not gaps or not kinds or any(kind != "local" for kind in kinds)
                or gaps != local_ids or repair.get("mode") != "preserve"
                or any(r.get("status") == "contradicted" for r in by_id.values())):
            errors.append("medium_contract_conflict")
    elif label == "low":
        central = (core.get("status") == "incorrect" and isinstance(core.get("error_type"), str)
                   and core["error_type"] in CORE_ERRORS and "central" in kinds and repair.get("mode") == "change_core")
        nonresponse = (core.get("status") == "no_understanding" and core.get("error_type") == "no_effective_answer"
                       and isinstance(plan, dict) and plan.get("material_answerable") is True
                       and raw.get("substantive_correct_understanding") is False
                       and "nonresponse" in kinds and repair.get("mode") == "rebuild")
        if not (central or nonresponse):
            errors.append("low_without_confirmed_core_error_or_nonresponse")
    return sorted(set(errors)), locations
