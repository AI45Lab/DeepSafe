from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

class BaseDataset(ABC):

    REQUIRED_FIELDS: Tuple[str, ...] = ()

    def __init__(
        self,
        path: str,
        *,
        strict: bool = True,
        **kwargs,
    ):
        """
        Base class for uni_eval datasets.

        Contract (recommended output item schema used by runners/evaluators):
          - id: unique identifier (str/int)
          - prompt: str  (used for generation stage)
          - prediction: Optional[str] (model output, when available)
          - ground_truth: Optional[Any]
          - meta: dict (task-specific extra info)

        For simple user-provided datasets, we additionally support JSONL records
        containing at least:
          - id
          - question
          - response

        Use `JsonlDataset` below to load such files without writing a custom loader.
        """
        self.path = path
        self.strict = bool(strict)
        self.data: List[Dict[str, Any]] = []

        self.kwargs = dict(kwargs)
        self.load()
        self._apply_limit()

    @abstractmethod
    def load(self):
        pass

    def _apply_limit(self):
        """
        Apply random sampling limit if 'limit' or 'max_samples' is provided in kwargs.
        Expects self.data to be populated by self.load().
        """
                                                                                              
        limit = self.kwargs.get("limit")
        if limit is None:
            limit = self.kwargs.get("max_samples")
        if limit is None:
            limit = getattr(self, "limit", None)
        if limit is None:
            limit = getattr(self, "max_samples", None)

        try:
            limit = int(limit or 0)
        except (ValueError, TypeError):
            limit = 0

        if limit > 0 and self.data:
                                                       
            seed = self.kwargs.get("seed")
            if seed is None:
                seed = getattr(self, "seed", 42)
            
            try:
                seed = int(seed or 42)
            except (ValueError, TypeError):
                seed = 42

            if limit < len(self.data):
                import random
                rng = random.Random(seed)
                rng.shuffle(self.data)
                self.data = self.data[:limit]
                print(f"[{self.__class__.__name__}] Random-sampled {len(self.data)} samples (limit={limit}, seed={seed}).")
            else:
                print(f"[{self.__class__.__name__}] Requested limit={limit} >= dataset size={len(self.data)}; using full dataset.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.data)

    @staticmethod
    def _read_jsonl(path: str) -> List[Dict[str, Any]]:
        if not path:
            raise ValueError("Empty path for JSONL dataset")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"JSONL file not found: {path}")

        out: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    raise ValueError(f"Invalid JSON at {path}:{ln}: {e}") from e
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected JSON object at {path}:{ln}, got {type(obj)}")
                out.append(obj)
        return out

    @classmethod
    def _validate_required_fields(
        cls,
        obj: Dict[str, Any],
        *,
        required_fields: Optional[Sequence[str]] = None,
        strict: bool = True,
        where: str = "",
    ) -> bool:
        req = tuple(required_fields) if required_fields is not None else tuple(getattr(cls, "REQUIRED_FIELDS", ()))
        if not req:
            return True

        missing = [k for k in req if k not in obj]
        if missing:
            msg = f"Missing required fields {missing}{(' at ' + where) if where else ''}"
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            return False

        bad = []
        for k in req:
            v = obj.get(k)
            if v is None:
                bad.append(k)
            elif isinstance(v, str) and not v.strip():
                bad.append(k)
        if bad:
            msg = f"Empty/None required fields {bad}{(' at ' + where) if where else ''}"
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            return False

        return True

from uni_eval.registry import DATASETS                                                      

@DATASETS.register_module()
class JsonlDataset(BaseDataset):
    """
    Minimal JSONL dataset for user uploads.

    Input: one JSON object per line with at least:
      - id
      - question
      - response

    Output: standardized records with:
      - id
      - prompt (alias of question)
      - prediction (alias of response)
      - question / response kept for convenience
      - meta.raw keeps the original JSON object
    """

    REQUIRED_FIELDS: Tuple[str, ...] = ("id", "question", "response")

    def __init__(
        self,
        path: str,
        *,
        id_key: str = "id",
        question_key: str = "question",
        response_key: str = "response",
        ensure_unique_id: bool = True,
        strict: bool = True,
        **kwargs,
    ):
        self.id_key = id_key
        self.question_key = question_key
        self.response_key = response_key
        self.ensure_unique_id = bool(ensure_unique_id)
        super().__init__(path=path, strict=strict, **kwargs)

    def load(self) -> None:
        raw = self._read_jsonl(self.path)

        seen = set()
        out: List[Dict[str, Any]] = []
        for i, obj in enumerate(raw):
            where = f"{self.path}:{i+1}"
                                                                     
            self._validate_required_fields(
                obj,
                required_fields=(self.id_key, self.question_key, self.response_key),
                strict=self.strict,
                where=where,
            )

            rid = obj.get(self.id_key)
            question = obj.get(self.question_key)
            response = obj.get(self.response_key)

            if isinstance(question, str):
                question = question.strip()
            if isinstance(response, str):
                response = response.strip()

            if self.ensure_unique_id:
                sid = str(rid)
                if sid in seen:
                    raise ValueError(f"Duplicate id={rid!r} at {where}")
                seen.add(sid)

            out.append(
                {
                    "id": rid,
                    "question": question,
                    "response": response,
                                                                       
                    "prompt": question,
                    "prediction": response,
                    "ground_truth": None,
                    "meta": {"raw": obj},
                }
            )

        self.data = out

