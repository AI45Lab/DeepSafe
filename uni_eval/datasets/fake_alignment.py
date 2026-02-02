import json
import os
from typing import Any, Dict, List
import logging

from uni_eval.registry import DATASETS
from uni_eval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

@DATASETS.register_module()
class FakeAlignmentDataset(BaseDataset):
    """
    Dataset loader for Fake-Alignment (FINE).

    Expected input: JSONL (one JSON object per line), e.g. Fake-Alignment-main/safety.jsonl
    with (at least) fields:
      - id
      - category
      - question
      - Positive Option
      - Negative Option

    We keep the raw data unchanged and only map fields into uni_eval's standard sample format:
      - prompt: question
      - meta.positive_option / meta.negative_option: pairwise options
      - meta.raw: original line for traceability
    """

    def load(self) -> None:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Fake-Alignment dataset not found at: {self.path}")

        raw_data: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_data.append(json.loads(line))

        self.data = []
        for idx, item in enumerate(raw_data):
                                                      
            q = item.get("question", "")
            pos = item.get("Positive Option", "")
            neg = item.get("Negative Option", "")
            cat = item.get("category", "")
            item_id = item.get("id", idx)

            if not q:
                q = item.get("prompt") or item.get("input") or ""
            if not pos:
                pos = item.get("positive_option") or item.get("positive") or ""
            if not neg:
                neg = item.get("negative_option") or item.get("negative") or ""

            if not q:
                logger.warning("Skipping item with empty question. idx=%s keys=%s", idx, list(item.keys()))
                continue

            self.data.append(
                {
                    "id": item_id,
                    "prompt": q,
                    "ground_truth": None,
                    "category": cat,
                    "meta": {
                        "positive_option": pos,
                        "negative_option": neg,
                        "raw": item,
                    },
                }
            )

        print(f"Loaded {len(self.data)} samples from Fake-Alignment dataset.")

