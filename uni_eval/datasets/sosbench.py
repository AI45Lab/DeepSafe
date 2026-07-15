import logging
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from uni_eval.registry import DATASETS
from uni_eval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)


@DATASETS.register_module()
class SOSBenchDataset(BaseDataset):
    """SOS-Bench dataset for scientific over-safety evaluation.

    Loads from a directory containing parquet file(s) (e.g. train-00000-of-00001.parquet).

    Each row has the following fields:
      goal          : str  — the harmful instruction
      original_term : str  — the sensitive concept being substituted
      subject       : str  — scientific domain (e.g. "biology")

    Output item schema:
      id            : int  — row index
      prompt        : str  — goal (used for model generation)
      original_term : str
      subject       : str  — scientific domain
      meta.raw      : dict — original row
    """

    def load(self) -> None:
        if pd is None:
            raise ImportError("Please install pandas: pip install pandas pyarrow")

        import os
        import glob

        path = getattr(self, "path", None)
        if not path:
            raise ValueError("SOSBenchDataset: path must be set")

        # Accept either a single parquet file or a directory containing parquet files
        if os.path.isfile(path) and path.endswith(".parquet"):
            parquet_files = [path]
        elif os.path.isdir(path):
            parquet_files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        else:
            raise ValueError(
                f"SOSBenchDataset: path must be a parquet file or directory, got: {path}"
            )

        if not parquet_files:
            raise FileNotFoundError(
                f"SOSBenchDataset: no parquet files found in {path}"
            )

        dfs = []
        for fpath in parquet_files:
            try:
                dfs.append(pd.read_parquet(fpath))
            except Exception as e:
                logger.warning("SOSBenchDataset: failed to read %s: %s", fpath, e)

        if not dfs:
            raise RuntimeError(f"SOSBenchDataset: could not load any parquet from {path}")

        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        self.data = []
        for idx, row in df.iterrows():
            goal = str(row.get("goal", "")).strip()
            if not goal:
                logger.warning("SOSBenchDataset: empty goal at row %d, skipping", idx)
                continue

            record: Dict[str, Any] = {
                "id": int(idx),
                "prompt": goal,
                "original_term": str(row.get("original_term", "")),
                "subject": str(row.get("subject", "")),
                "ground_truth": None,
                "meta": {"raw": row.to_dict()},
            }
            self.data.append(record)

        logger.info(
            "SOSBenchDataset: loaded %d items from %d file(s) at %s",
            len(self.data),
            len(parquet_files),
            path,
        )
