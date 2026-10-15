"""Exact prompt/window preparation copied from the supplied training snapshot.
No label, ID or reference is serialized into model inputs.
"""
import json
if __package__:
    from .common import FIELDS, LABELS
else:
    from common import FIELDS, LABELS


def label_token_ids(tokenizer):
    values = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    if any(len(ids) != 1 for ids in values) or len({ids[0] for ids in values}) != 3:
        raise ValueError("This classifier requires three distinct single-token labels")
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must provide EOS and PAD tokens")
    return [ids[0] for ids in values]


def system_prompt(schema):
    criteria = schema["criteria"]
    return ("你是评论理解答案质量分类器。" + schema["instructions"] + "\n"
            + "\n".join(label + "：" + criteria[label] for label in LABELS)
            + "\n只输出一个类别：high、medium 或 low。不要解释。")


def prompt_ids(tokenizer, row, context, schema):
    # Never serialize sample labels, reference annotations or source metadata into the prompt.
    visible = {key: context if key == "original_context" else row[key] for key in FIELDS}
    messages = [{"role": "system", "content": system_prompt(schema)},
                {"role": "user", "content": json.dumps(visible, ensure_ascii=False)}]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False)
    # Different transformers versions return either an input_ids list or a BatchEncoding.
    if hasattr(encoded, "input_ids"):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return encoded


def prepare_samples(rows, tokenizer, config, schema):
    """Keep all context characters; repeat question/comment/answer in each window."""
    label_token_ids(tokenizer)
    budget = config["max_length"] - 2  # Training appends label token and <|im_end|>.
    prepared = []
    for row in rows:
        context = row["original_context"]
        complete = prompt_ids(tokenizer, row, context, schema)
        windows, spans = [], []
        if len(complete) <= budget:
            windows, spans = [complete], [[0, len(context)]]
        else:
            if len(prompt_ids(tokenizer, row, "", schema)) >= budget:
                raise ValueError("Question/comment/answer exceed the budget; increase max_length: " + row["sample_id"])
            start = 0
            while start < len(context):
                low, high, end = start + 1, len(context), start
                while low <= high:
                    middle = (low + high) // 2
                    if len(prompt_ids(tokenizer, row, context[start:middle], schema)) <= budget:
                        end, low = middle, middle + 1
                    else:
                        high = middle - 1
                if end <= start:
                    raise ValueError("Cannot fit any context: " + row["sample_id"])
                if end < len(context):
                    boundary = context.rfind("\n", max(start + 1, end - 200), end)
                    if boundary > start:
                        end = boundary + 1
                ids = prompt_ids(tokenizer, row, context[start:end], schema)
                if len(ids) > budget:
                    raise ValueError("Window exceeds the token budget")
                windows.append(ids)
                spans.append([start, end])
                if end == len(context):
                    break
                overlap = config["context_overlap_tokens"]
                offsets = tokenizer(context[start:end], add_special_tokens=False,
                                    return_offsets_mapping=True)["offset_mapping"]
                next_start = (start + offsets[max(0, len(offsets) - overlap)][0]
                              if overlap else end)
                if next_start <= start:
                    raise ValueError("Context overlap prevents progress; reduce overlap or increase max_length")
                start = next_start
            covered = 0
            for start, end in spans:
                if start > covered:
                    raise ValueError("Context coverage gap")
                covered = max(covered, end)
            if covered != len(context):
                raise ValueError("Context suffix was lost")
        prepared.append({"sample_id": row["sample_id"], "windows": windows, "window_spans": spans,
                         "full_tokens": len(complete) + 2,
                         "target_index": LABELS.index(row["label"]) if row.get("label") in LABELS else -1,
                         "evaluation_dimension": row.get("evaluation_dimension"),
                         "canonical_source_group": row.get("canonical_source_group")})
    return prepared
