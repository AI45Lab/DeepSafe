from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from uni_eval.datasets.base import BaseDataset
from uni_eval.registry import DATASETS

logger = logging.getLogger(__name__)

@DATASETS.register_module()
class HaluEvalQADataset(BaseDataset):
    """
    QA-only hallucination detection dataset loader.

    Expected parquet schema (as in /datasets/HaluEval/README.md):
      - id: str
      - passage: str (document/context)
      - question: str
      - answer: str
      - label: str  ("PASS"=no hallucination, "FAIL"=hallucination)
      - score: int  (1 for PASS, 0 for FAIL)
      - source_ds: str

    Output item schema:
      - id: str
      - prompt: str (kept for generic runners; evaluator.generate_predictions is recommended)
      - ground_truth: "Yes" (hallucination) / "No" (no hallucination)
      - meta: includes passage/question/answer/label/score/source_ds
    """

    REQUIRED_FIELDS: Tuple[str, ...] = ("id", "passage", "question", "answer", "label")

    def __init__(
        self,
        path: str,
        *,
        strict: bool = True,
        **kwargs,
    ):
        super().__init__(path=path, strict=strict, **kwargs)

    @staticmethod
    def _label_to_ground_truth(label: str) -> str:

        lab = (label or "").strip().upper()
        if lab == "FAIL":
            return "Yes"
        if lab == "PASS":
            return "No"
                                                     
        raise ValueError(f"Unknown label={label!r} (expected PASS/FAIL)")

    def load(self) -> None:
        try:
            import pandas as pd                
        except Exception as e:                    
            raise ImportError("HaluEvalQADataset requires pandas (and pyarrow/fastparquet for parquet).") from e

        df = pd.read_parquet(self.path)

        out: List[Dict[str, Any]] = []
        seen = set()
        for i, row in df.iterrows():
            obj = row.to_dict()
            where = f"{self.path}:row={i}"
            self._validate_required_fields(obj, strict=self.strict, where=where)

            rid = str(obj.get("id")).strip()
            if rid in seen:
                raise ValueError(f"Duplicate id={rid!r} at {where}")
            seen.add(rid)

            passage = (obj.get("passage") or "").strip()
            question = (obj.get("question") or "").strip()
            answer = (obj.get("answer") or "").strip()
            label = (obj.get("label") or "").strip()
            score = obj.get("score", None)
            source_ds = (obj.get("source_ds") or "").strip()

            gt = self._label_to_ground_truth(label)

            prompt = (
                "You are a hallucination detector. Answer only Yes or No.\n\n"
                f"#Document#: {passage}\n"
                f"#Question#: {question}\n"
                f"#Answer#: {answer}\n"
                "#Your Judgement#:"
            )

            out.append(
                {
                    "id": rid,
                    "prompt": prompt,
                    "prediction": None,
                    "ground_truth": gt,
                    "meta": {
                        "passage": passage,
                        "question": question,
                        "answer": answer,
                        "label": label,
                        "score": score,
                        "source_ds": source_ds,
                    },
                }
            )

        self.data = out

