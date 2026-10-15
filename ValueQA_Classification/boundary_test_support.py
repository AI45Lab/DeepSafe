"""Synthetic certificates shared by transport/integration tests; never inference data."""
import copy
import json
if __package__:
    from .boundary_contract import bind_single_review
else:
    from boundary_contract import bind_single_review


def certify_review(value, row):
    value = copy.deepcopy(value)
    plan = {"requirements_complete": True, "material_answerable": True, "ambiguity_note": "",
            "requirements": [{key: item[key] for key in ("id", "point", "question_evidence", "material_evidence")}
                             for item in value["requirements"]]}
    for item in plan["requirements"]:
        item.update(necessity_reason="该项由题目明确要求", acceptable_completion=item["point"])
    value.update(full_answer_checked=True, reasonable_entailment_checked=True,
                 key_claims_supported=True, substantive_correct_understanding=True,
                 repair={"mode": "none" if value["label"] == "high" else "preserve", "action": "补充必要日期"})
    value["core"]["error_type"] = "none"
    for issue in value["issues"]:
        issue.update(defect_type="required_omission", necessity_reason="日期由题目明确要求",
                     preserves_core=True, correct_understanding_evidence=value["core"]["answer_evidence"])
    value["requirements_plan"] = plan
    return bind_single_review(value, row)


def stage_response(messages, value):
    return value


def visible_sample(messages):
    content = messages[1]["content"]
    if not content.startswith("{"):
        content = content.split("\n", 1)[1]
    value = json.loads(content)
    return value.get("sample", value)
