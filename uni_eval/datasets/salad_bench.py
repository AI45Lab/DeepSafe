import json
import logging
from typing import Dict, Any, Optional

from uni_eval.registry import DATASETS
from uni_eval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

@DATASETS.register_module()
class SaladDataset(BaseDataset):
    """
    Unified dataset loader for Salad-Bench (supports both MCQ and QA/Safety tasks).
    
    Automatically detects the data format based on JSON fields:
    - MCQ Mode: Input has "mcq", "gt", "choices" fields.
    - QA Mode: Input has "question" field (e.g., base_set or attack_enhanced_set).
    """

    def load(self):
        """Load Salad-Bench data (MCQ or QA) and standardize."""
        print(f"DEBUG: Loading from {self.path}")
        with open(self.path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not raw_data:
            self.data = []
            return

        first_item = raw_data[0]
        print(f"DEBUG: First item keys: {list(first_item.keys())}")
        
        if "mcq" in first_item:
            self._load_mcq(raw_data)
        elif "question" in first_item:
            self._load_qa(raw_data)
        else:
            print(f"WARNING: Unknown Salad-Bench data format. Keys found: {list(first_item.keys())}")
            self.data = []

    def _load_mcq(self, raw_data: list):
        """Handle MCQ subset."""
                                              
        self.data = []
        for idx, item in enumerate(raw_data):
            standardized = {
                "id": idx,
                "prompt": item.get("mcq", ""),
                "ground_truth": item.get("gt", ""),
                "meta": {
                    "task_type": "mcq",
                    "baseq": item.get("baseq", ""),
                    "choices": item.get("choices", ""),
                    "categories": {
                        "level1": item.get("1-category", ""),
                        "level2": item.get("2-category", ""),
                        "level3": item.get("3-category", ""),
                    },
                    "raw": item,
                },
            }
            self.data.append(standardized)

    def _load_qa(self, raw_data: list):
        """Handle QA / Safety subset."""
                                             
        self.data = []
        for idx, item in enumerate(raw_data):
            standardized = {
                "id": item.get("qid", idx),
                "prompt": item.get("question", ""),
                "ground_truth": None,                         
                "meta": {
                    "task_type": "qa",
                    "categories": {
                        "level1": item.get("1-category", ""),
                        "level2": item.get("2-category", ""),
                        "level3": item.get("3-category", ""),
                    },
                    "source": item.get("source", ""),
                    "raw": item,
                },
            }
            self.data.append(standardized)
