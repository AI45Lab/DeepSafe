import os
from typing import Any, Dict, List
import datasets

from uni_eval.registry import DATASETS
from uni_eval.datasets.base import BaseDataset

@DATASETS.register_module()
class BeaverTailsDataset(BaseDataset):
    """
    BeaverTails (PKU-Alignment/BeaverTails) dataset adapter.

    This adapter formats each sample as a *classification* prompt and expects the
    model to answer with exactly one token: SAFE or UNSAFE.

    Notes:
    - BeaverTails contains safety-related annotations. Different config variants
      exist on HF; this loader tries to be robust by inferring common field names.
    - If your local setup requires offline caching, set HF_DATASETS_CACHE or pass cache_dir via env.
    """
    def _load_hf(self):
        subset_files = {
            "30k": "round0/30k/test.jsonl.gz",
        }
        raw_data = {}
        for subset, json_file in subset_files.items():
                                          
            dataframe = datasets.load_dataset("json", data_files=os.path.join(self.path, json_file))["train"]
            raw_data[subset] = dataframe
        return raw_data

    @staticmethod
    def _extract_true_category_keys(category):
        """
        category: dict like { "a": False, "b": True, ... }
        return: list of keys where value is True (robust to 1/"true")
        """
        category_keys = ""
        for k, v in category.items():
            if v is True:
                category_keys = k
            elif isinstance(v, (int, float)) and int(v) == 1:
                category_keys = k
            elif isinstance(v, str) and v.strip().lower() in {"true", "1", "yes"}:
                category_keys = k
        return category_keys

    def load(self):
        """
        Load BeaverTails data (json).
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"BeaverTails dataset not found at: {self.path}")

        raw_data = self._load_hf()

        self.data = []
        for subset in raw_data.keys():
            for item in raw_data[subset]:
                print(item)
                prompt = item["prompt"]
                category = item["category"]
                category_keys = self._extract_true_category_keys(category)
                self.data.append(
                    {
                        "prompt": prompt,
                        "meta": {
                            "subset": subset,
                            "category": category_keys,
                        },
                    }
                )
        
        print(f"Loaded {len(self.data)} samples from BeaverTails dataset.")

