import json
from typing import Any, Dict, List

from uni_eval.registry import DATASETS
from uni_eval.datasets.base import BaseDataset

@DATASETS.register_module()
class PrecomputedDataset(BaseDataset):
    """
    Dataset that loads a JSON list of items that already contain:
      - prompt: str
      - prediction: str (optional, but required if evaluator requires it)

    This is used for two-phase runs where we generate predictions in Phase 1
    and then judge them in Phase 2 without loading the target model.
    """

    def load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raise ValueError("PrecomputedDataset expects a JSON list at dataset.path")

        out: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            prompt = item.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            rec: Dict[str, Any] = {
                "id": item.get("id", idx),
                "prompt": prompt,
            }
            if "prediction" in item:
                rec["prediction"] = item.get("prediction")
                                                 
            if "meta" in item:
                rec["meta"] = item.get("meta")
            out.append(rec)

        self.data = out

