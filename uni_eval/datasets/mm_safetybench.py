import logging
import os
import base64
from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd
try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    try:
        import pandas as pd
        PYARROW_AVAILABLE = False
    except ImportError:
        raise ImportError(
            "MMSafetyBenchDataset requires either 'pyarrow' or 'pandas' to read parquet files. "
            "Please install one of them: pip install pyarrow or pip install pandas"
        )

from uni_eval.datasets.base import BaseDataset
from uni_eval.registry import DATASETS

logger = logging.getLogger(__name__)

MM_SAFETYBENCH_CATEGORIES: Sequence[str] = [
    "EconomicHarm",
    "Financial_Advice",
    "Fraud",
    "Gov_Decision",
    "HateSpeech",
    "Health_Consultation",
    "Illegal_Activitiy",
    "Legal_Opinion",
    "Malware_Generation",
    "Physical_Harm",
    "Political_Lobbying",
    "Privacy_Violence",
    "Sex",
]

MM_SAFETYBENCH_SPLITS: Sequence[str] = [
    "SD",
    "SD_TYPO",
    "TYPO",
    "Text_only",
]

@DATASETS.register_module()
class MMSafetyBenchDataset(BaseDataset):
    """
    Dataset loader for MM-SafetyBench (parquet version).
    Standardized item format (each row becomes):
        {
            "id": str,
            "prompt": str,          # the text question
            "ground_truth": None,   # open-ended safety probing
            "meta": {
                "category": str,    # one of MM_SAFETYBENCH_CATEGORIES
                "split": str,       # SD / SD_TYPO / TYPO / Text_only
                "label": Any,       # optional scenario label column
                "image": Any,       # raw image column from parquet (for image-based splits)
                "raw": Dict[str, Any],  # full original row for debugging
            },
        }
    """

    def __init__(
        self,
        path: str,
        include_categories: Optional[Sequence[str]] = None,
        include_splits: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        """
        Args:
            path: Root directory of MM-SafetyBench (the folder that contains `data/`).
            include_categories: Optional subset of categories to load.
            include_splits: Optional subset of splits to load.
        """
        self.include_categories = (
            list(include_categories) if include_categories is not None else list(MM_SAFETYBENCH_CATEGORIES)
        )
        self.include_splits = (
            list(include_splits) if include_splits is not None else list(MM_SAFETYBENCH_SPLITS)
        )
        super().__init__(path=path, **kwargs)

    def load(self):
                                                                                      
        data_root = os.path.join(self.path, "data")
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"MM-SafetyBench data dir not found at {data_root}")

        logger.info(
            "Loading MM-SafetyBench from %s (categories=%s, splits=%s)",
            data_root,
            self.include_categories,
            self.include_splits,
        )

        all_items: List[Dict[str, Any]] = []
        for category in self.include_categories:
            for split in self.include_splits:
                parquet_path = os.path.join(data_root, category, f"{split}.parquet")
                if not os.path.exists(parquet_path):
                    logger.warning("Parquet file missing: %s", parquet_path)
                    continue

                try:
                    if PYARROW_AVAILABLE:
                                                                             
                        table = pq.read_table(parquet_path)
                                                  
                        rows = []
                        for i in range(len(table)):
                            row_dict = {}
                            for col_name in table.column_names:
                                col = table[col_name]
                                val = col[i]
                                                                              
                                if hasattr(val, 'as_py'):
                                    row_dict[col_name] = val.as_py()
                                else:
                                    row_dict[col_name] = val
                            rows.append(row_dict)
                    else:
                                            
                        df = pd.read_parquet(parquet_path)
                        rows = [row.to_dict() for _, row in df.iterrows()]
                except Exception as e:
                    logger.error("Failed to read parquet file %s: %s", parquet_path, e)
                    continue

                logger.info(
                    "Loaded %d rows from %s (%s / %s)",
                    len(rows),
                    parquet_path,
                    category,
                    split,
                )

                for row_dict in rows:
                    item = self._build_item_from_row(row_dict=row_dict, category=category, split=split)
                    if item is not None:
                        all_items.append(item)

        self.data = all_items
        logger.info("Total MM-SafetyBench items loaded: %d", len(self.data))

    def _build_item_from_row(self, row_dict: Dict[str, Any], category: str, split: str) -> Optional[Dict[str, Any]]:
        """Convert a single parquet row (as dict) into the standardized dict."""
        question = row_dict.get("question", None)
        if not isinstance(question, str) or not question.strip():
                                                      
            logger.debug("Skipping row without valid question (category=%s, split=%s)", category, split)
            return None

        q_text = question.strip()

        orig_id = row_dict.get("id", None)
        if orig_id is None:
                                      
            sample_id = f"{category}:{split}:{hash(q_text)}"
        else:
            sample_id = f"{category}:{split}:{orig_id}"

        label = row_dict.get("label", None)
        image_obj = row_dict.get("image", None)

        meta: Dict[str, Any] = {
            "category": category,
            "split": split,
            "label": label,
            "orig_id": orig_id,
            "image": image_obj,
            "raw": row_dict.copy(),
        }

        prompt: Union[str, List[Dict[str, Any]]] = q_text

        img_b64 = None
        mime = "image/png"
        if image_obj is not None:
                        
            if isinstance(image_obj, (bytes, bytearray, memoryview)):
                img_b64 = base64.b64encode(bytes(image_obj)).decode("utf-8")
                                 
            elif isinstance(image_obj, dict):
                b = image_obj.get("bytes")
                if isinstance(b, (bytes, bytearray, memoryview)):
                    img_b64 = base64.b64encode(bytes(b)).decode("utf-8")
                                                                                    
                path = image_obj.get("path")
                if img_b64 is None and isinstance(path, str) and os.path.exists(path):
                    with open(path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode("utf-8")
                    lower = path.lower()
                    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
                        mime = "image/jpeg"
                    elif lower.endswith(".png"):
                        mime = "image/png"
                    elif lower.endswith(".webp"):
                        mime = "image/webp"
                    elif lower.endswith(".gif"):
                        mime = "image/gif"

        if img_b64:
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                        {"type": "text", "text": q_text},
                    ],
                }
            ]

        return {
            "id": str(sample_id),
            "prompt": prompt,
            "ground_truth": None,
            "meta": meta,
        }

