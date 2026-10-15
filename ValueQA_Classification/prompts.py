"""Independent evidence regrading; no initial label/reason or numeric quality scores."""
import json
if __package__:
    from .common import ROOT, blind
    from .initial_protocol import system_prompt
else:
    from common import ROOT, blind
    from initial_protocol import system_prompt


def initial_messages(row, schema):
    return [{"role": "system", "content": system_prompt(schema)},
            {"role": "user", "content": json.dumps(blind(row), ensure_ascii=False)}]


def review_messages(row, schema, requirements=None):
    """One complete-answer review; requirements are diagnosed within this call."""
    classification = (ROOT / "classification_prompt_strict.txt").read_text(encoding="utf-8-sig").strip()
    template = (ROOT / "review_prompt.txt").read_text(encoding="utf-8")
    if requirements is not None:
        raise ValueError("Single-stage review does not accept a separately generated requirements plan")
    data = blind(row)
    return [{"role": "system", "content": classification + "\n\n证据复核与输出要求：\n" + template},
            {"role": "user", "content": "以下是完整待分析数据：\n" + json.dumps(data, ensure_ascii=False)}]
