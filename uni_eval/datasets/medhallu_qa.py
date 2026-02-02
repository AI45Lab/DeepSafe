from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from uni_eval.datasets.base import BaseDataset
from uni_eval.registry import DATASETS

@DATASETS.register_module()
class MedHalluQADataset(BaseDataset):
    """
    MedHallu QA hallucination detection dataset.

    Input parquet columns (per dataset card):
      - Question
      - Knowledge
      - Ground Truth
      - Difficulty Level
      - Hallucinated Answer
      - Category of Hallucination

    Output:
      - Double expansion per row:
        - GT item: label_int=0 (factual)
        - Hallucinated item: label_int=1 (hallucinated)
      - Unique id: f"{row_id}__gt" / f"{row_id}__hall"
      - prompt is a placeholder; generation should use evaluator.generate_predictions().
    """

    REQUIRED_FIELDS: Tuple[str, ...] = (
        "Question",
        "Knowledge",
        "Ground Truth",
        "Difficulty Level",
        "Hallucinated Answer",
        "Category of Hallucination",
    )

    def __init__(
        self,
        path: str,
        *,
        subset_name: str = "pqa_labeled",
        sample_mode: str = "expand",                       
        seed: int = 0,
        strict: bool = True,
        **kwargs,
    ):
        self.subset_name = str(subset_name or "pqa_labeled")
        self.sample_mode = str(sample_mode or "expand").strip().lower()
        if self.sample_mode not in {"expand", "random_one"}:
            raise ValueError(f"Unsupported sample_mode={sample_mode!r} (expected 'expand' or 'random_one')")
        self.seed = int(seed or 0)
        super().__init__(path=path, strict=strict, **kwargs)

    def load(self) -> None:
        try:
            import pandas as pd                
        except Exception as e:                    
            raise ImportError("MedHalluQADataset requires pandas (and pyarrow/fastparquet for parquet).") from e

        df = pd.read_parquet(self.path)

        out: List[Dict[str, Any]] = []
        seen = set()
        rng = random.Random(self.seed)

        for i, row in df.iterrows():
            obj = row.to_dict()
            where = f"{self.path}:row={i}"
            self._validate_required_fields(obj, strict=self.strict, where=where)

            question = (obj.get("Question") or "").strip()
            knowledge = obj.get("Knowledge")
            gt_answer = (obj.get("Ground Truth") or "").strip()
            hall_answer = (obj.get("Hallucinated Answer") or "").strip()
            difficulty = (obj.get("Difficulty Level") or "").strip().lower()
            category = (obj.get("Category of Hallucination") or "").strip()

            base_id = f"{self.subset_name}-{i}"
            items: List[Tuple[str, str, int, str]] = []
            if self.sample_mode == "expand":
                items = [
                    ("gt", gt_answer, 0, "NoHallucination"),
                    ("hall", hall_answer, 1, category),
                ]
            else:

                if rng.randint(0, 1) == 0:
                    items = [("gt", gt_answer, 0, "NoHallucination")]
                else:
                    items = [("hall", hall_answer, 1, category)]

            for suffix, ans, label_int, cat in items:
                rid = f"{base_id}__{suffix}" if self.sample_mode == "expand" else base_id
                if rid in seen:
                    raise ValueError(f"Duplicate id={rid!r} at {where}")
                seen.add(rid)

                prompt = f"Question: {question}\nAnswer: {ans}\nYour Judgement:"

                out.append(
                    {
                        "id": rid,
                        "prompt": prompt,
                        "prediction": None,
                        "ground_truth": int(label_int),
                        "meta": {
                            "subset": self.subset_name,
                            "question": question,
                            "knowledge": knowledge,
                            "answer": ans,
                            "answer_source": suffix,             
                            "difficulty": difficulty,
                            "category": cat,
                            "sample_mode": self.sample_mode,
                            "seed": self.seed,
                        },
                    }
                )

        self.data = out

