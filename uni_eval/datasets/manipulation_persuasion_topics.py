import json
import os
from typing import Any, Dict, List

from uni_eval.registry import DATASETS
from uni_eval.datasets.base import BaseDataset

@DATASETS.register_module()
class ManipulationPersuasionTopicDataset(BaseDataset):
    """
    Topic dataset for manipulation/persuasion experiments.

    Expected file format
    --------------------
    JSON or JSONL where each record has at least:
      - ``id``: int or str (topic id)
      - ``topic``: str, the Chinese statement of the issue, e.g.
          "自动驾驶汽车将提高安全性并取代人类驾驶员"

    Standardized item format:
      {
        "id": <id>,
        "prompt": <topic>,        # for compatibility with StandardEvaluator etc.
        "topic": <topic>,
        "ground_truth": None,
        "meta": {"raw": <original_record>}
      }
    """

    def load(self) -> None:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"ManipulationPersuasionTopicDataset file not found: {self.path}")

        records: List[Dict[str, Any]] = []
        if self.path.endswith(".jsonl"):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
        else:
            with open(self.path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                records = [x for x in obj if isinstance(x, dict)]
            elif isinstance(obj, dict):
                                                                  
                for k, v in obj.items():
                    if isinstance(v, str):
                        records.append({"id": k, "topic": v})

        data: List[Dict[str, Any]] = []
        for idx, rec in enumerate(records):
            topic = rec.get("topic")
            if not isinstance(topic, str) or not topic.strip():
                continue
            item_id = rec.get("id", idx)
            data.append(
                {
                    "id": item_id,
                    "prompt": topic,
                    "topic": topic,
                    "ground_truth": None,
                    "meta": {"raw": rec},
                }
            )

        self.data = data

