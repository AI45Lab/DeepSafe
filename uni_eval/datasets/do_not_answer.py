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
class DoNotAnswerDataset(BaseDataset):
    """
    Do-Not-Answer dataset adapter.

    Primary source: HuggingFace `LibrAI/do-not-answer` (train split, 939 rows).
    Optional: load from local files under `path` if available (parquet folder / single parquet / HF load_from_disk).

    Output item schema (flattened for metric convenience):
      - id: int
      - prompt: str (dataset 'question')
      - risk_area: str
      - harm_type: str (dataset 'types_of_harm')
      - specific_harms: str
      - meta.raw: original HF row (dict-like)
    """

    def __init__(self, path: str, split: str = "train", **kwargs):
        self.split = split
        super().__init__(path=path, **kwargs)

    def _load_local(self) -> Optional[List[Dict[str, Any]]]:
        if not self.path or self.path == "HF":
            return None

        if not os.path.exists(self.path):
            return None

        if load_dataset is None:
            raise ImportError("Please install 'datasets' library: pip install datasets")

        def _collect_parquet_files(root: str) -> List[str]:
            files: List[str] = []
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if fn.endswith(".parquet"):
                        files.append(os.path.join(dirpath, fn))
            return sorted(files)

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

        if not parquet_files and os.path.isdir(self.path):
            parquet_files = _collect_parquet_files(self.path)

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
                ds = load_dataset("LibrAI/do-not-answer", split=self.split)
                rows = list(ds)
            finally:
                _restore_proxy(old)

        if rows is None:
            raise RuntimeError("Failed to load Do-Not-Answer dataset (both local and HF).")

        self.data = []
        for r in rows:
                                                                                    
            q = (r.get("question") or "").strip()
            self.data.append(
                {
                    "id": int(r.get("id")) if r.get("id") is not None else len(self.data),
                    "prompt": q,
                    "risk_area": (r.get("risk_area") or "Unknown").strip(),
                    "harm_type": (r.get("types_of_harm") or "Unknown").strip(),
                    "specific_harms": (r.get("specific_harms") or "Unknown").strip(),
                    "ground_truth": None,
                    "meta": {"raw": dict(r)},
                }
            )

