import os
import glob
import logging
import hashlib
from typing import List, Dict, Any

import pandas as pd

from uni_eval.datasets.base import BaseDataset
from uni_eval.registry import DATASETS

logger = logging.getLogger(__name__)

@DATASETS.register_module()
class XSTestDataset(BaseDataset):
    """XSTest Dataset (load from CSV files).

    CSV columns:
      id, prompt, type, label, focus, note
    """

    def __init__(self, path: str, **kwargs):
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
                           
                cid = self._norm_str(row.get("id"))
                prompt = self._norm_str(row.get("prompt"))
                ctype = self._norm_str(row.get("type"))
                label_text = self._norm_str(row.get("label"))
                focus = self._norm_str(row.get("focus"))
                note = self._norm_str(row.get("note"))

                r: Dict[str, Any] = {
                           
                    "id": cid if cid else None,
                    "prompt": prompt,
                    "type": ctype,
                    "label": label_text,                       
                    "focus": focus,
                    "note": note,
                }

                if not r["id"]:
                    src_key = f"{base}|{i}"
                    r["id"] = hashlib.md5(src_key.encode("utf-8")).hexdigest()[:16]

                all_records.append(r)

        self.data = all_records
        self.datat = self.data         
        logger.info(f"Loaded {len(self.data)} rows from CSV(s) at {root}")
        return self.data

if __name__ == "__main__":
    dataset = XSTestDataset(path="/mnt/shared-storage-user/guoshaoxiong/Multi-Level-Bionic-Evaluation-Framework/configs/datasets/xstest_prompts.csv")
    data = dataset.load()
    print(f"Loaded {len(data)} records.")
    if data:
        print(data[0])