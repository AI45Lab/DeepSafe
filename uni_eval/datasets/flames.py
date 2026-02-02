import json
import os
from uni_eval.registry import DATASETS
from uni_eval.datasets.base import BaseDataset

@DATASETS.register_module()
class FlamesDataset(BaseDataset):
    """
    Dataset loader for the Flames dataset (Flames_1k_Chinese).
    Expected format: JSONL where each line contains (at least):
      - dimension: one of [Fairness, Safety, Morality, Legality, Data protection]
      - subcomponent: finer category (optional in some releases)
      - prompt: the adversarial prompt (Chinese)
    """

    def load(self):
        """
        Load Flames data (JSONL).
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Flames dataset not found at: {self.path}")

        raw_data = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_data.append(json.loads(line))

        for item in raw_data:
            prompt = item.get("prompt")
            if not prompt:
                                                             
                prompt = item.get("question") or item.get("input") or item.get("text")
            if not prompt:
                continue

            dimension = item.get("dimension", "")
            subcomponent = item.get("subcomponent", "")

            self.data.append(
                {
                    "prompt": prompt,
                                                                       
                    "ground_truth": "",
                    "dimension": dimension,
                    "subcomponent": subcomponent,
                                                              
                    "meta": {"raw": item},
                }
            )
        
        print(f"Loaded {len(self.data)} samples from Flames dataset.")
