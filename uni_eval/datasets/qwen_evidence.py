"""Full-context Qwen evidence records and reference-only span annotations."""

import json
from pathlib import Path

from qwen_evidence_review.common import LABELS, FIELDS, blind
from uni_eval.datasets.base import BaseDataset
from uni_eval.registry import DATASETS

ROOT = Path(__file__).resolve().parents[2]
TEXT_FIELDS = ("original_context", "target_comment", "question", "answer")


def read_records(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if path.suffix.lower() == ".json":
        records = json.loads(path.read_text(encoding="utf-8-sig"))
    else:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("Qwen evidence data must contain JSON objects")
    return records


@DATASETS.register_module()
class QwenEvidenceDataset(BaseDataset):
    def load(self):
        rows = read_records(self.path)
        reference_path = self.kwargs.get("references_path")
        references = {}
        if reference_path:
            for ref in read_records(reference_path):
                sid = str(ref["sample_id"])
                if sid in references:
                    raise ValueError("Duplicate reference ID: " + sid)
                references[sid] = ref
        seen = set()
        for row in rows:
            rid = row.get("sample_id")
            if not isinstance(rid, (str, int)) or isinstance(rid, bool) or not str(rid).strip():
                raise ValueError("Qwen evidence requires a nonempty sample_id")
            sid = str(rid)
            if sid in seen:
                raise ValueError("Duplicate sample ID: " + sid)
            seen.add(sid)
            for name in TEXT_FIELDS:
                if name == "target_comment" and name in row and row[name] is None:
                    continue
                if not isinstance(row.get(name), str) or not row[name].strip():
                    raise ValueError("Missing Qwen evidence material: " + name + " for " + sid)
            label = row.get("label")
            if label is None and self.kwargs.get("labels_from_references", True):
                label = references.get(sid, {}).get("label")
            if label not in LABELS:
                raise ValueError("Invalid reference label for " + sid)
            ref = references.get(sid) if reference_path else row
            if ref is None or ref.get("label") != label:
                raise ValueError("Missing or mismatched reference for " + sid)
            annotations = ref.get("annotations")
            if not isinstance(annotations, list):
                raise ValueError("Reference annotations must be an array for " + sid)
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    raise ValueError("Invalid reference annotation for " + sid)
                start, end, text = (annotation.get(k) for k in ("start", "end", "text"))
                field = annotation.get("field", "answer")
                content = row.get(field) if isinstance(field, str) and field in TEXT_FIELDS else None
                if (not isinstance(content, str) or type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(content)
                        or not isinstance(text, str) or content[start:end] != text):
                    raise ValueError("Reference offsets/text mismatch for " + sid)
            # This is the complete model-input object. Reference fields stay outside it.
            inputs = {"sample_id": sid, **{name: row[name] for name in TEXT_FIELDS}}
            # Preserve source fields verbatim; DeepSafe-only fields are added to the in-memory record.
            self.data.append({**row, "id": rid, "answer": row["answer"], "qwen_input": inputs,
                              "prompt": json.dumps(blind(inputs), ensure_ascii=False), "ground_truth": label,
                              "ground_truth_annotations": annotations,
                              "ground_truth_spans": [annotation["text"] for annotation in annotations],
                              "meta": {"prompt_version": "qwen_evidence_v1"}})
        if not self.data:
            raise ValueError("Qwen evidence dataset is empty")
        if reference_path and set(references) != seen:
            raise ValueError("Reference IDs do not match the complete input dataset")
