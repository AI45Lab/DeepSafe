import os
import random
from uni_eval.registry import DATASETS
from uni_eval.datasets.base import BaseDataset

@DATASETS.register_module()
class WMDPDataset(BaseDataset):
    """
    Dataset loader for the WMDP dataset (cais/wmdp).
    Subset: [wmdp-bio, wmdp-chem, wmdp-cyber]
    Expected format: parquet where each line contains (at least):
      - question: the harmful prompt (English)
      - choices: four options (str in List), for example 
        [
            "SpyCEP from Streptococcus pyogenes",
            "Eremophilane O-acetyltransferase from Penicillium roqueforti",
            "Interleukin-8 (IL-8) from Homo sapiens",
            "Acyl-CoA synthetase from Alternaria alternata"
        ]
      - answer: one of [0, 1, 2, 3]
    """

    def _load_hf(self):

        try:
            import datasets                
        except Exception as e:                    
            raise ImportError(
                "Failed to import HuggingFace `datasets` required for WMDP.\n"
                "Fix: `pip install -U datasets xxhash` in the runtime environment."
            ) from e

        subset_files = {
            "wmdp-bio": "wmdp-bio/test-00000-of-00001.parquet", 
            "wmdp-chem": "wmdp-chem/test-00000-of-00001.parquet", 
            "wmdp-cyber": "wmdp-cyber/test-00000-of-00001.parquet"
        }
        raw_data = {}
        for subset, parquet_file in subset_files.items():
            dataframe = datasets.load_dataset("parquet", data_files=os.path.join(self.path, parquet_file))["train"]
            raw_data[subset] = dataframe
        return raw_data

    def load(self):
        """
        Load WMDP data (parquet).
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"WMDP dataset not found at: {self.path}")

        raw_data = self._load_hf()

        self.data = []
        for subset in raw_data.keys():
            for item in raw_data[subset]:
                question = item["question"]
                choices = item["choices"]
                answer_idx = item["answer"]
                idx2letter = ["A", "B", "C", "D"]

                prompt = question + "\n\n"
                for i, option in enumerate(choices):
                    prompt += f"{idx2letter[i]}. {option}\n"
                prompt += "\nAnswer with one letter (A, B, C, or D)."

                self.data.append(
                    {
                        "prompt": prompt,
                        "ground_truth": idx2letter[answer_idx],
                        "meta": {
                                "subset": subset,
                            },
                    }
                )

        print(f"Loaded {len(self.data)} samples from WMDP dataset.")
