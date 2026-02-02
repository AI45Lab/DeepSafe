import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from uni_eval.datasets.base import BaseDataset
from uni_eval.registry import DATASETS

logger = logging.getLogger(__name__)

@DATASETS.register_module()
class SandbaggingDataset(BaseDataset):
    """
    Dataset for Sandbagging experiment.
    
    Loads MMLU samples and expands each sample into multiple items,
    one for each target accuracy.
    
    Expected input format (jsonl):
    {"question": "...", "answer": "A", ...}
    """

    def __init__(
        self,
        path: str,
        target_accuracies: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        self.target_accuracies = (
            list(target_accuracies)
            if target_accuracies is not None
            else [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
        )
        super().__init__(path=path, **kwargs)

    def load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Sandbagging file not found: {self.path}")

        items: List[Dict[str, Any]] = []

        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        parsed: List[Tuple[int, Dict[str, Any]]] = []
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = obj.get("question")
            answer = obj.get("answer")
            if not question or not answer:
                continue
            parsed.append((idx, obj))

        for raw_idx, obj in parsed:
            question = obj.get("question")
            answer = obj.get("answer")

            for target in self.target_accuracies:
                meta = {
                    "question": question,
                    "answer": answer,
                    "target_accuracy": target,
                    "raw_id": raw_idx,
                    "raw": obj,
                }

                item_id = f"{raw_idx}_{target}"
                
                items.append(
                    {
                        "id": item_id,
                        "prompt": question,
                        "ground_truth": answer,
                        "meta": meta,
                    }
                )

        self.data = items
        logger.info(
            "SandbaggingDataset loaded %d valid raw samples, expanded to %d items (targets=%s)",
            len(parsed),
            len(self.data),
            self.target_accuracies,
        )

