import os
import logging
import pandas as pd
from typing import Any, Dict, List, Optional, Union

try:
    from datasets import load_dataset, load_from_disk
except ImportError:
    load_dataset = None
    load_from_disk = None

from uni_eval.registry import DATASETS
from uni_eval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

SUPPORTED_DATASETS = [f"diamonds-seed{i}" for i in range(8)] + ["function_correctness"]
HF_REPO_MAP = {
    **{f"diamonds-seed{i}": f"redwoodresearch/diamonds-seed{i}" for i in range(8)},
    "function_correctness": "redwoodresearch/function_correctness"
}

def _maybe_apply_proxy():
    """Temporarily apply PROXY_URL to HF-related env vars, 复用参考文件的代理逻辑"""
    proxy_url = os.getenv("PROXY_URL")
    keys = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]
    old = {k: os.environ.get(k) for k in keys}
    if proxy_url:
        for k in keys:
            os.environ[k] = proxy_url
    return old

def _restore_proxy(old: Dict[str, Optional[str]]):
    """Restore original proxy env vars"""
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

@DATASETS.register_module()
class ReasonUnderPressureDataset(BaseDataset):
    """
    Dataset adapter for Reason-Under-Pressure task (DIAMONDS & function_correctness).
    Adapted from VERL inference evaluation script, fully support diamonds-seed0~7 and function_correctness dataset.
    
    Loading Priority (align with do_not_answer.py):
    1. Local HF dataset (load_from_disk)
    2. Local parquet files (single file / directory)
    3. Hugging Face Hub (redwoodresearch/xxx)
    
    Output item schema (unified for uni_eval, compatible with metric calculation):
      - id: int, unique sample index
      - prompt: str, model input prompt (core field required by BaseDataset)
      - reference: list, ground truth measurements (core field for metric compute)
      - difficulty: int, sample difficulty level
      - is_correct: Union[bool, None], latent variable from VERL dataset
      - text: str, original sample text content
      - n_measurements: int, number of ground truth measurements
      - meta: dict, raw original data from local/HF dataset
    """

    def __init__(
        self,
        path: str,
        dataset_name: str = "diamonds-seed0",                                            
        split: str = "val",
        **kwargs
    ):
                           
        self.dataset_name = dataset_name
        self.split = split

        if self.dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset_name: {self.dataset_name}. "
                f"Supported datasets: {SUPPORTED_DATASETS}"
            )

        self.hf_repo = HF_REPO_MAP[self.dataset_name]
        super().__init__(path=path, **kwargs)

    def _load_local(self) -> Optional[List[Dict[str, Any]]]:
        """
        Local data loader, align with do_not_answer.py's loading priority:
        1. Local HF dataset (load_from_disk)
        2. Local parquet files (single file / directory)
        Return None if all local loading methods fail
        """
        if not self.path or not os.path.exists(self.path):
            logger.warning(f"Local dataset path {self.path} is invalid or not exists")
            return None

        if load_from_disk is not None:
            try:
                ds = load_from_disk(self.path)
                                                             
                if hasattr(ds, "keys") and self.split in ds:
                    rows = list(ds[self.split])
                else:
                    rows = list(ds)
                logger.info(f"Loaded {len(rows)} samples from local HF dataset: {self.path}")
                return rows
            except Exception as e:
                logger.warning(f"Failed to load_from_disk from {self.path}: {str(e)}")

        parquet_target = os.path.join(self.path, f"{self.split}.parquet")
                                      
        if os.path.isfile(parquet_target) and parquet_target.endswith(".parquet"):
            try:
                df = pd.read_parquet(parquet_target)
                rows = df.to_dict("records")
                logger.info(f"Loaded {len(rows)} samples from single parquet: {parquet_target}")
                return rows
            except Exception as e:
                logger.warning(f"Failed to load parquet file {parquet_target}: {str(e)}")

        parquet_files = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith(".parquet")]
        if parquet_files:
            try:
                df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
                rows = df.to_dict("records")
                logger.info(f"Loaded {len(rows)} samples from parquet dir: {self.path}")
                return rows
            except Exception as e:
                logger.warning(f"Failed to load parquet dir {self.path}: {str(e)}")

        return None

    def load(self) -> None:
        """
        Mandatory override load() method for BaseDataset.
        Full data loading pipeline, align with do_not_answer.py: local first, then HF hub.
        Format data to uni_eval standard schema.
        """
        if load_dataset is None:
            raise ImportError("Please install 'datasets' library: pip install datasets")

        rows = self._load_local()

        if rows is None:
            logger.info(f"Local loading failed, try to load from HF Hub: {self.hf_repo}")
            old = _maybe_apply_proxy()
            try:
                ds = load_dataset(self.hf_repo, split=self.split)
                rows = list(ds)
                logger.info(f"Loaded {len(rows)} samples from HF Hub: {self.hf_repo} (split={self.split})")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load dataset from both local and HF Hub! "
                    f"Local path: {self.path}, HF repo: {self.hf_repo}, error: {str(e)}"
                )
            finally:
                _restore_proxy(old)

        data = []
        for idx, raw_row in enumerate(rows):
            sample = {
                "id": idx,  
                "prompt": raw_row.get("text", "").strip(),                   
                
                "difficulty": raw_row.get("difficulty", -1),
                "is_correct": raw_row.get("is_correct", None),
                "is_clean": raw_row.get("is_clean", None),
                "measurements": raw_row.get("measurements", []),
                "n_measurements": len(raw_row.get("measurements", [])),

                "meta": {
                    "dataset_name": self.dataset_name,
                    "split": self.split,
                    "source": "local" if rows is not None else "hf_hub",
                    "hf_repo": self.hf_repo
                }
            }
            data.append(sample)
            
        self.data = data 
        
        logger.info(
            f"Successfully loaded ReasonUnderPressure dataset | "
            f"dataset={self.dataset_name}, split={self.split}, total_samples={len(self.data)}, "
            f"source={self.data[0]['meta']['source']}"
        )