import os
import glob
import logging
import hashlib
from typing import Optional, Sequence, List, Dict, Any

import pandas as pd

from uni_eval.datasets.base import BaseDataset
from uni_eval.prompts import PROMPT_REGISTRY
from uni_eval.registry import DATASETS

logger = logging.getLogger(__name__)

@DATASETS.register_module()
class HarmBenchDataset(BaseDataset):
    """HarmBench Dataset (load from original CSV files).

    CSV columns:
      Behavior, FunctionalCategory, SemanticCategory, Tags, ContextString, BehaviorID
    """

    def __init__(self, path: str, subdirs: Optional[Sequence[str]] = None, **kwargs):
                   
        if subdirs is not None:
            logger.warning("HarmBenchDataset: 'subdirs' is ignored; loading from CSV files.")
        super().__init__(path, **kwargs)

    def _norm_str(self, v) -> str:
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        return str(v)

    def _parse_tags(self, tags_str: str):
        if not tags_str:
            return []
        return [t.strip() for t in tags_str.split(",") if t.strip()]

    def load(self) -> List[Dict[str, Any]]:
        """Load data from CSV file(s)."""
        root = getattr(self, "path", None)
        if not root:
            logger.error("No dataset path provided (path is empty).")
            self.data = []
            self.datat = self.data
            return self.data

        files: List[str] = []
        if os.path.isdir(root):
            files = sorted(glob.glob(os.path.join(root, "*.csv")))
            if not files:
                logger.error(f"No CSV files found under directory: {root}")
        elif os.path.isfile(root) and root.lower().endswith(".csv"):
            files = [root]
        else:
            logger.error(f"Path is neither a CSV file nor a directory with CSVs: {root}")

        all_records: List[Dict[str, Any]] = []

        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception as e:
                logger.error(f"Failed to read CSV {f}: {e}")
                continue

            base = os.path.basename(f)
            for i, row in df.iterrows():
                           
                Behavior = self._norm_str(row.get("Behavior"))
                FunctionalCategory = self._norm_str(row.get("FunctionalCategory"))
                SemanticCategory = self._norm_str(row.get("SemanticCategory"))
                Tags = self._norm_str(row.get("Tags"))
                ContextString = self._norm_str(row.get("ContextString"))
                BehaviorID = self._norm_str(row.get("BehaviorID"))

                r: Dict[str, Any] = {
                    "behavior": Behavior,
                    "domain": FunctionalCategory,
                    "category": SemanticCategory,
                    "tags": self._parse_tags(Tags),
                    "context": ContextString,
                    "id": BehaviorID,
                }
                r['prompt'] = Behavior               
                all_records.append(r)

        self.data = all_records
        logger.info(f"Loaded {len(self.data)} rows from CSV(s) at {root}")

if __name__ == "__main__":
    dataset = HarmBenchDataset(path="/mnt/shared-storage-user/guoshaoxiong/Multi-Level-Bionic-Evaluation-Framework/configs/datasets/harmbench_behaviors_text_all.csv")
    dataset.load()
    print(f"Loaded {len(dataset.data)} records.")
    if dataset.data:
        import random
                               
        domain_dict = {}
        for record in dataset.data:
            domain = record.get("domain", "unknown")
            if domain not in domain_dict:
                domain_dict[domain] = []
            domain_dict[domain].append(record)
        for domain, records in domain_dict.items():
            sample = random.choice(records)
            print(f"Domain: {domain}, Sample ID: {sample['id']}, Prompt: {sample}...")