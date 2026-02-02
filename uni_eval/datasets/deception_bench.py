import logging
import os
from typing import Any, Dict, List

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from uni_eval.registry import DATASETS
from uni_eval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

@DATASETS.register_module()
class DeceptionBenchDataset(BaseDataset):
    """
    Dataset loader for DeceptionBench.
    Loads from HuggingFace: PKU-Alignment/DeceptionBench
    
    Each sample contains:
    - id: unique identifier
    - type: deception category (5 categories)
    - inner_prompt: for Mesa Utility Elicitation
    - system_prompt: evaluation context
    - outer_prompt: conflicting outer goal
    - format: required response format with thinking process
    """

    def load(self) -> None:
        if load_dataset is None:
            raise ImportError("Please install 'datasets' library: pip install datasets")

        print("Loading DeceptionBench dataset from HuggingFace...")

        proxy_url = os.getenv("PROXY_URL")
        old_proxies = {k: os.environ.get(k) for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]}
        if proxy_url:
            for k in old_proxies.keys():
                os.environ[k] = proxy_url

        try:
                                   
            hf_data = load_dataset("PKU-Alignment/DeceptionBench", split="test")
            self.data = []
            
            for idx, item in enumerate(hf_data):
                item_id = item.get("id", idx)
                deception_type = item.get("type", "Unknown")
                inner_prompt = item.get("inner_prompt", "").strip()
                system_prompt = item.get("system_prompt", "").strip()
                outer_prompt = item.get("outer_prompt", "").strip()
                format_str = item.get("format", "").strip()

                self.data.append({
                    "id": item_id,
                    "prompt": outer_prompt,                                      
                    "ground_truth": None,
                    "meta": {
                        "type": deception_type,
                        "inner_prompt": inner_prompt,
                        "system_prompt": system_prompt,
                        "outer_prompt": outer_prompt,
                        "format": format_str,
                        "raw": item
                    }
                })
                
        finally:
                             
            for k, v in old_proxies.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        print(f"Successfully loaded {len(self.data)} samples from DeceptionBench dataset.")