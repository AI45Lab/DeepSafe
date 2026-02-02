import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Sequence

from uni_eval.datasets.base import BaseDataset
from uni_eval.registry import DATASETS

logger = logging.getLogger(__name__)

@DATASETS.register_module()
class UncontrolledAIRDDataset(BaseDataset):
    """
    Uncontrolled_AIRD dataset loader (minimal, generation-oriented).

    Current focus: `actor` task (multi-turn) from Attack_600-style JSON:
    each item contains:
      - id, query_id, category, actor_name, relationship
      - plain_query (the hidden/target harmful query)
      - multi_turn_queries (a list of multi-turn prompts)

    Standardized item format:
      {
        "id": str,
        "prompt": str,               # plain_query (for compatibility / indexing)
        "ground_truth": None,
        "meta": {...}                # includes multi_turn_queries and other fields
      }
    """

    def __init__(
        self,
        path: str,
        task: str = "actor",
        stages: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        self.task = task

        self.stages = list(stages) if stages is not None else ["monitor"]
        super().__init__(path=path, **kwargs)

    def load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Uncontrolled_AIRD file not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raise ValueError(f"Expected a list in {self.path}, got: {type(raw)}")

        items: List[Dict[str, Any]] = []
        for obj in raw:
            if not isinstance(obj, dict):
                continue

            sample_id = obj.get("id", None)
            plain_query = obj.get("plain_query", "") or ""
            multi_turn_queries = obj.get("multi_turn_queries", []) or []

            if self.task == "actor":
                if not isinstance(multi_turn_queries, list) or len(multi_turn_queries) == 0:
                    logger.warning("Skip sample with empty multi_turn_queries: id=%s", sample_id)
                    continue

            for stage in self.stages:
                meta = {
                    "task": self.task,
                    "stage": stage,
                    "query_id": obj.get("query_id"),
                    "category": obj.get("category"),
                    "actor_name": obj.get("actor_name"),
                    "relationship": obj.get("relationship"),
                    "plain_query": plain_query,
                    "multi_turn_queries": multi_turn_queries,
                    "raw": obj,
                }

                base_id = str(sample_id) if sample_id is not None else str(len(items))
                items.append(
                    {
                        "id": f"{base_id}_{stage}",
                        "prompt": plain_query,
                        "ground_truth": None,
                        "meta": meta,
                    }
                )

        self.data = items
        logger.info("UncontrolledAIRDDataset loaded %d items (task=%s)", len(self.data), self.task)

