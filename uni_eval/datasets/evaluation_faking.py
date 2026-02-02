import json
import os
import logging
from typing import Any, Dict, List, Optional

try:
    from datasets import load_dataset, load_from_disk
except ImportError:
    load_dataset = None
    load_from_disk = None

from uni_eval.registry import DATASETS
from uni_eval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

LOCAL_HUGGINFACE_REPO_MAPPING = {
    "strongreject": "walledai/StrongREJECT",
    "sorrybench": "sorry-bench/sorry-bench-202406"
}

def _maybe_apply_proxy():
    """Temporarily apply PROXY_URL to HF-related env vars."""
    proxy_url = os.getenv("PROXY_URL")
    keys = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]
    old = {k: os.environ.get(k) for k in keys}
    if proxy_url:
        for k in keys:
            os.environ[k] = proxy_url
    return old

def _restore_proxy(old: Dict[str, Optional[str]]):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

@DATASETS.register_module()
class EvaluationFakingDataset(BaseDataset):
    """
    Output item schema (flattened for metric convenience):
     
    """

    def __init__(self, 
                 path: str, 
                 dataset_name: str = 'strongreject',                    
                 split: str = "train", 
                 use_jailbreak: bool = True,
                 jailbreak_template: str = "aim",                            
                 **kwargs):
        self.dataset_name = dataset_name
        self.split = split
        self.use_jailbreak = use_jailbreak
        self.jailbreak_template_name = jailbreak_template
        super().__init__(path=path, **kwargs)

    def _load_local(self) -> Optional[List[Dict[str, Any]]]:
        if not self.path or self.path == "HF":
            return None

        if not os.path.exists(self.path):
            return None

        if os.path.isfile(self.path):
            lower = self.path.lower()
            if lower.endswith(".jsonl"):
                try:
                    return BaseDataset._read_jsonl(self.path)
                except Exception as e:
                    logger.warning(f"Failed to load jsonl from {self.path}: {e}")
            if lower.endswith(".json"):
                try:
                    with open(self.path, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    if isinstance(obj, list):
                        return [x for x in obj if isinstance(x, dict)]
                    if isinstance(obj, dict):
                        for k in ("data", "rows", "samples"):
                            v = obj.get(k)
                            if isinstance(v, list):
                                return [x for x in v if isinstance(x, dict)]
                except Exception as e:
                    logger.warning(f"Failed to load json from {self.path}: {e}")

        if load_dataset is None:
            raise ImportError("Please install 'datasets' library: pip install datasets")

        if os.path.isdir(self.path) and os.path.exists(os.path.join(self.path, "dataset_info.json")) and load_from_disk:
            try:
                ds = load_from_disk(self.path)
                                                  
                if hasattr(ds, "keys") and self.split in ds:
                    return list(ds[self.split])
                return list(ds)
            except Exception as e:
                logger.warning(f"Failed to load_from_disk from {self.path}: {e}")

        parquet_files: List[str] = []
        if os.path.isfile(self.path) and self.path.endswith(".parquet"):
            parquet_files = [self.path]
        elif os.path.isdir(self.path):
            for fn in os.listdir(self.path):
                if fn.endswith(".parquet"):
                    parquet_files.append(os.path.join(self.path, fn))

        if parquet_files:
            try:
                ds = load_dataset("parquet", data_files=parquet_files, split="train")
                return list(ds)
            except Exception as e:
                logger.warning(f"Failed to load parquet from {self.path}: {e}")

        return None

    def load(self) -> None:
        if load_dataset is None:
            raise ImportError("Please install 'datasets' library: pip install datasets")

        rows: Optional[List[Dict[str, Any]]] = self._load_local()
        if rows is None:
            old = _maybe_apply_proxy()
            try:
                hf_dataset = LOCAL_HUGGINFACE_REPO_MAPPING[self.dataset_name]
                ds = load_dataset(hf_dataset, split=self.split)
                rows = list(ds)
            finally:
                _restore_proxy(old)

        if rows is None:
            raise RuntimeError("Failed to load Do-Not-Answer dataset (both local and HF).")

        jailbreak_prompt = None
        if self.use_jailbreak and 'strongreject' in self.dataset_name:
            if self.jailbreak_template_name == "aim":
                                                          
                from uni_eval.prompts import AIM_JAILBREAK_PROMPT
                jailbreak_prompt = AIM_JAILBREAK_PROMPT
            else:
                                                  
                jailbreak_prompt = self.jailbreak_template_name

        self.data = []
        for r in rows:
            
            if 'strongreject' in self.dataset_name:
                q = (r.get("prompt") or "").strip()
            else: 
                q = (r.get("turns")[0] or "").strip()

            if jailbreak_prompt and q:
                q = jailbreak_prompt.format(prompt=q)
                
            self.data.append(
                {
                    "id": int(r.get("id")) if r.get("id") is not None else len(self.data),
                    "prompt": q,
                    "source": (r.get("source") or "Unknown").strip(),
                    "category": (r.get("category") or "Unknown").strip(),
                    "meta": {"raw": dict(r)},
                }
            )

