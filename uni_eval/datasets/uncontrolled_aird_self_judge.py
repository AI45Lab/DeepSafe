import json
import logging
import os
from collections import defaultdict
import random
from typing import Any, Dict, List, Optional, Sequence

from uni_eval.datasets.base import BaseDataset
from uni_eval.registry import DATASETS

logger = logging.getLogger(__name__)

@DATASETS.register_module()
class UncontrolledAIRDSelfJudgeDataset(BaseDataset):
    """
    Dataset for Exp2 (AI Self-Annotation / Safety Judgment).

    Expected input: saferlhf_simple.json-style:
      [{"qid":..., "question":..., "answer":..., "safe_label": true/false}, ...]

    We expand each record into one or more stages (eval/deploy), so metrics can compute
    SR per stage and stage gap within a single run.
    """

    def __init__(
        self,
        path: str,
        stages: Optional[Sequence[str]] = None,
        **kwargs,
    ):
                                                                                                     
        self.stages = list(stages) if stages is not None else ["eval", "deploy"]
        super().__init__(path=path, **kwargs)

    def load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Uncontrolled_AIRD self-judge file not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raise ValueError(f"Expected a list in {self.path}, got: {type(raw)}")

        out: List[Dict[str, Any]] = []

        qid_counts = defaultdict(int)
        for obj in raw:
            if not isinstance(obj, dict):
                continue

            qid = obj.get("qid")
            question = (obj.get("question") or "").strip()
            answer = (obj.get("answer") or "").strip()
            safe_label = 1 if bool(obj.get("safe_label")) else 0

            if not question or not answer:
                continue

            base_key = str(qid) if qid is not None else str(len(out))
            qid_counts[base_key] += 1

            occ_suffix = "" if qid_counts[base_key] == 1 else f"__dup{qid_counts[base_key]}"

            for stage in self.stages:
                meta = {
                    "stage": stage,
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "safe_label": safe_label,
                    "raw": obj,
                }
                out.append(
                    {
                        "id": f"{base_key}{occ_suffix}_{stage}",
                        "prompt": question,                                                               
                        "ground_truth": safe_label,
                        "meta": meta,
                    }
                )

        self.data = out
        logger.info("UncontrolledAIRDSelfJudgeDataset loaded %d items", len(self.data))

