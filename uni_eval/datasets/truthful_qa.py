import logging
import os
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from uni_eval.registry import DATASETS
from uni_eval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

@DATASETS.register_module()
class TruthfulQADataset(BaseDataset):
    """
    Dataset loader for TruthfulQA (Multiple Choice Task).
    Supports:
    - Local JSON file (recommended for offline / cluster runs)
    - HuggingFace dataset "truthful_qa/multiple_choice" as fallback
    - Optional local CSV for Category mapping when using HuggingFace
    """

    @staticmethod
    def _split_semicolon_list(s: Any) -> List[str]:
        if s is None:
            return []
        if isinstance(s, list):
            return [str(x).strip() for x in s if str(x).strip()]
        text = str(s).strip()
        if not text:
            return []
                                                                   
        parts = [p.strip() for p in text.split(";")]
        return [p for p in parts if p]

    @staticmethod
    def _build_mc_targets_from_local_row(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Convert a local TruthfulQA row into (mc1_targets, mc2_targets) compatible with our evaluator/metric.

        Local schema example:
          - Question
          - Best Answer
          - Correct Answers (semicolon separated)
          - Incorrect Answers (semicolon separated)
        """
        best = str(row.get("Best Answer", "") or "").strip()
        correct = TruthfulQADataset._split_semicolon_list(row.get("Correct Answers", ""))
        incorrect = TruthfulQADataset._split_semicolon_list(row.get("Incorrect Answers", ""))

        def uniq(xs: List[str]) -> List[str]:
            seen = set()
            out: List[str] = []
            for x in xs:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        incorrect_u = uniq([x for x in incorrect if x and x != best])
        correct_u = uniq([x for x in correct if x])

        mc1_choices = ([best] if best else []) + incorrect_u
        mc1_labels = ([1] if best else []) + ([0] * len(incorrect_u))

        mc2_choices = uniq(correct_u + incorrect_u + ([best] if best else []))
        correct_set = set(correct_u + ([best] if best else []))
        mc2_labels = [1 if c in correct_set else 0 for c in mc2_choices]

        return (
            {"choices": mc1_choices, "labels": mc1_labels},
            {"choices": mc2_choices, "labels": mc2_labels},
        )

    def load(self) -> None:

        if self.path and os.path.isdir(self.path):
            data_dir = Path(self.path).expanduser().resolve()
            print(f"DEBUG: Loading TruthfulQA from local parquet dir: {data_dir}")

            category_map: Dict[str, str] = {}
            def _norm_q(s: str) -> str:
                                                                          
                return " ".join(str(s or "").strip().split())

            csv_path = str(self.kwargs.get("category_csv") or "").strip()
            if csv_path and not os.path.isfile(csv_path):
                print(f"WARNING: category_csv not found: {csv_path}")
                csv_path = ""

            if not csv_path:
                                                                                         
                local_csv = data_dir / "TruthfulQA.csv"
                if local_csv.is_file():
                    csv_path = str(local_csv)
                else:
                                                              
                    mbef_root = Path(__file__).resolve().parents[2]
                    default_csv = mbef_root / "data" / "TruthfulQA" / "TruthfulQA.csv"
                    if default_csv.is_file():
                        csv_path = str(default_csv)

            if csv_path and os.path.isfile(csv_path):
                try:
                    print(f"DEBUG: Loading category metadata from {csv_path}")
                    df = pd.read_csv(csv_path)
                    for _, row in df.iterrows():
                        q = _norm_q(row.get("Question", ""))
                        cat = str(row.get("Category", "Unknown")).strip()
                        if q:
                            category_map[q] = cat
                    print(f"DEBUG: Loaded category mapping for {len(category_map)} questions.")
                except Exception as e:
                    print(f"WARNING: Could not load category metadata from CSV: {e}")

            parquet_files = sorted([p for p in data_dir.glob("*.parquet") if p.is_file()])
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in directory: {data_dir}")

            rows: List[Dict[str, Any]] = []
            try:
                import pyarrow.parquet as pq                

                for fp in parquet_files:
                    table = pq.read_table(str(fp))
                    rows.extend(table.to_pylist())
            except Exception:
                for fp in parquet_files:
                    dfp = pd.read_parquet(str(fp))
                    rows.extend(dfp.to_dict(orient="records"))

            items: List[Dict[str, Any]] = []
            for idx, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                question = str(row.get("question", "") or "").strip()
                if not question:
                    continue
                mc1_targets = row.get("mc1_targets", {}) or {}
                mc2_targets = row.get("mc2_targets", {}) or {}
                category = category_map.get(_norm_q(question), "Unknown")
                items.append(
                    {
                        "id": idx,
                        "prompt": question,
                        "ground_truth": None,
                        "meta": {
                            "category": category,
                            "mc1_targets": mc1_targets,
                            "mc2_targets": mc2_targets,
                            "raw": row,
                        },
                    }
                )

            self.data = items
            print(f"DEBUG: Successfully loaded {len(self.data)} items from local parquet.")
            return

        if self.path and os.path.isfile(self.path) and self.path.lower().endswith(".json"):
            print(f"DEBUG: Loading TruthfulQA from local JSON: {self.path}")
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "data" in raw:
                raw = raw["data"]
            if not isinstance(raw, list):
                raise ValueError(f"Expected a JSON list in {self.path}, got {type(raw)}")

            items: List[Dict[str, Any]] = []
            for idx, row in enumerate(raw):
                if not isinstance(row, dict):
                    continue
                                                                    
                question = str(row.get("question") or row.get("Question") or "").strip()
                if not question:
                    continue

                category = str(row.get("category") or row.get("Category") or "Unknown").strip()

                if "mc1_targets" in row and "mc2_targets" in row:
                    mc1_targets = row.get("mc1_targets") or {}
                    mc2_targets = row.get("mc2_targets") or {}
                else:
                    mc1_targets, mc2_targets = self._build_mc_targets_from_local_row(row)

                items.append(
                    {
                        "id": idx,
                        "prompt": question,
                        "ground_truth": None,
                        "meta": {
                            "category": category,
                            "mc1_targets": mc1_targets,
                            "mc2_targets": mc2_targets,
                            "raw": row,
                        },
                    }
                )

            self.data = items
            print(f"DEBUG: Successfully loaded {len(self.data)} items from local JSON.")
            return

        category_map = {}
        if self.path and os.path.isfile(self.path) and self.path.lower().endswith(".csv"):
            try:
                print(f"DEBUG: Loading category metadata from {self.path}")
                                                                  
                df = pd.read_csv(self.path)
                                                                                             
                for _, row in df.iterrows():
                    q = str(row.get("Question", "")).strip()
                    cat = str(row.get("Category", "Unknown")).strip()
                    if q:
                        category_map[q] = cat
                print(f"DEBUG: Loaded category mapping for {len(category_map)} questions.")
            except Exception as e:
                print(f"WARNING: Could not load category metadata from CSV: {e}")

        default_local_json = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "truthful_qa.json")
        default_local_json = os.path.abspath(default_local_json)
        if os.path.isfile(default_local_json):
            print(f"DEBUG: Offline mode: using bundled local JSON: {default_local_json}")
            with open(default_local_json, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "data" in raw:
                raw = raw["data"]
            if not isinstance(raw, list):
                raise ValueError(f"Expected a JSON list in {default_local_json}, got {type(raw)}")

            items: List[Dict[str, Any]] = []
            for idx, row in enumerate(raw):
                if not isinstance(row, dict):
                    continue
                question = str(row.get("question") or row.get("Question") or "").strip()
                if not question:
                    continue
                category = category_map.get(question) or str(row.get("category") or row.get("Category") or "Unknown").strip()
                mc1_targets, mc2_targets = self._build_mc_targets_from_local_row(row)
                items.append(
                    {
                        "id": idx,
                        "prompt": question,
                        "ground_truth": None,
                        "meta": {
                            "category": category,
                            "mc1_targets": mc1_targets,
                            "mc2_targets": mc2_targets,
                            "raw": row,
                        },
                    }
                )
            self.data = items
            print(f"DEBUG: Successfully loaded {len(self.data)} items from bundled local JSON.")
            return

        if load_dataset is None:
            raise ImportError("Please install 'datasets' library: pip install datasets")

        print("DEBUG: Loading primary TruthfulQA data from HuggingFace...")

        proxy_url = os.getenv("PROXY_URL")
        old_proxies = {k: os.environ.get(k) for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]}
        if proxy_url:
            for k in old_proxies.keys(): os.environ[k] = proxy_url

        try:
            hf_data = load_dataset("truthful_qa", "multiple_choice", split="validation")
            self.data = []
            for idx, item in enumerate(hf_data):
                question = item.get("question", "").strip()

                category = category_map.get(question)
                if not category:
                    category = item.get("category", "Unknown")
                
                self.data.append({
                    "id": idx,
                    "prompt": question,
                    "ground_truth": None,
                    "meta": {
                        "category": category,
                        "mc1_targets": item.get("mc1_targets", {}),
                        "mc2_targets": item.get("mc2_targets", {}),
                        "raw": item
                    }
                })
        finally:
                             
            for k, v in old_proxies.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v

        print(f"DEBUG: Successfully loaded {len(self.data)} items with category enrichment.")
