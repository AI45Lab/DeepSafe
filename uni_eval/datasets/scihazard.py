import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from uni_eval.datasets.base import BaseDataset
from uni_eval.registry import DATASETS

logger = logging.getLogger(__name__)


@DATASETS.register_module()
class SciHazardDataset(BaseDataset):
    """SciHazard dataset adapter for DeHarmScore JSONL files.

    Modes:
      - full: load both `2400unsafe_final.jsonl` and `600safe_final.jsonl`
      - subset: load `scihazard_subset.jsonl`

    Output schema:
      - id
      - prompt
      - question
      - ground_truth
      - label
      - split
      - source_file
      - meta
    """

    def __init__(
        self,
        path: str,
        mode: str = "full",
        unsafe_filename: str = "2400unsafe_final.jsonl",
        safe_filename: str = "600safe_final.jsonl",
        subset_filename: str = "scihazard_subset.jsonl",
        ensure_unique_id: bool = True,
        strict: bool = True,
        **kwargs,
    ):
        self.mode = str(mode).strip().lower()
        self.unsafe_filename = unsafe_filename
        self.safe_filename = safe_filename
        self.subset_filename = subset_filename
        self.ensure_unique_id = bool(ensure_unique_id)
        super().__init__(path=path, strict=strict, **kwargs)

    def _resolve_file(self, root: str, filename: str) -> str:
        if os.path.isabs(filename):
            return filename
        return os.path.join(root, filename)

    def _read_jsonl_file(self, path: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    raise ValueError(f"Invalid JSON at {path}:{ln}: {e}") from e
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected JSON object at {path}:{ln}, got {type(obj)}")
                records.append(obj)
        return records

    def _build_entry(
        self,
        raw: Dict[str, Any],
        *,
        split: str,
        source_file: str,
        local_index: int,
    ) -> Dict[str, Any]:
        question = str(raw.get("question", "")).strip()
        if not question:
            msg = f"SciHazardDataset: empty question in {source_file} at item {local_index}"
            if self.strict:
                raise ValueError(msg)
            logger.warning(msg)
            return {}

        idx = raw.get("idx")
        rid = f"{split}_{idx}" if idx is not None else f"{split}_{local_index}"

        is_harmful = raw.get("is_harmful")
        if isinstance(is_harmful, bool):
            ground_truth = is_harmful
        else:
            ground_truth = split == "unsafe"
        label = "unsafe" if ground_truth else "safe"

        entry: Dict[str, Any] = {
            "id": rid,
            "idx": idx,
            "prompt": question,
            "question": question,
            "ground_truth": ground_truth,
            "label": label,
            "split": split,
            "source_file": os.path.basename(source_file),
            "prediction": None,
            "meta": {
                "raw": raw,
                "split": split,
                "source_file": os.path.basename(source_file),
                "risk_level": raw.get("risk_level"),
                "subject": raw.get("subject"),
                "sub_discipline": raw.get("sub_discipline"),
                "category": raw.get("category"),
                "substance": raw.get("substance"),
            },
        }

        # Surface common fields directly for convenience in templates/evaluators.
        for key in (
            "subject",
            "sub_discipline",
            "category",
            "substance",
            "risk_level",
            "classification_reasoning",
            "reasoning",
            "defense_angle",
            "lifecycle_stage",
            "technical_bottleneck",
            "classified_by",
            "is_harmful",
        ):
            if key in raw:
                entry[key] = raw.get(key)

        return entry

    def _load_split(
        self,
        path: str,
        *,
        split: str,
    ) -> List[Dict[str, Any]]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"SciHazardDataset file not found: {path}")

        rows = self._read_jsonl_file(path)
        out: List[Dict[str, Any]] = []
        for i, raw in enumerate(rows):
            entry = self._build_entry(raw, split=split, source_file=path, local_index=i)
            if entry:
                out.append(entry)
        return out

    def _full_files(self, root: str) -> List[Tuple[str, str]]:
        return [
            (self._resolve_file(root, self.unsafe_filename), "unsafe"),
            (self._resolve_file(root, self.safe_filename), "safe"),
        ]

    def load(self) -> List[Dict[str, Any]]:
        root = getattr(self, "path", None)
        if not root or not os.path.isdir(root):
            logger.error("SciHazardDataset: path is not a directory: %s", root)
            self.data = []
            return self.data

        if self.mode not in {"full", "subset"}:
            raise ValueError(f"SciHazardDataset: unsupported mode={self.mode!r}, expected 'full' or 'subset'")

        all_records: List[Dict[str, Any]] = []
        if self.mode == "full":
            for fpath, split in self._full_files(root):
                all_records.extend(self._load_split(fpath, split=split))
        else:
            subset_path = self._resolve_file(root, self.subset_filename)
            all_records.extend(self._load_split(subset_path, split="subset"))

        if self.ensure_unique_id:
            seen = set()
            for item in all_records:
                rid = str(item["id"])
                if rid in seen:
                    raise ValueError(f"SciHazardDataset: duplicate id detected: {rid}")
                seen.add(rid)

        self.data = all_records
        logger.info(
            "SciHazardDataset: loaded %d items in mode=%s from %s",
            len(self.data),
            self.mode,
            root,
        )
        return self.data
