import json
import logging
import os
from typing import Any, Dict, List

from uni_eval.datasets.base import BaseDataset
from uni_eval.registry import DATASETS

logger = logging.getLogger(__name__)

# Mapping from JSON filename stem to human-readable domain name
_DOMAIN_MAP = {
    "bio": "biology",
    "chem": "chemistry",
    "is": "information_security",
    "material": "materials_science",
    "med": "medicine",
    "phy": "physics",
}


@DATASETS.register_module()
class SafeScientistDataset(BaseDataset):
    """Safe-Scientist dataset for scientific safety evaluation.

    Loads from a directory containing per-domain JSON files:
      bio.json, chem.json, is.json, material.json, med.json, phy.json

    Each JSON file is a list of objects with fields:
      Task, Task Description, Prompt, Risk Type, sourceUrl

    Output item schema:
      id              : "{domain}_{index}"
      prompt          : Prompt field (used for model generation)
      task            : Task field
      task_description: Task Description field
      risk_type       : Risk Type field
      source_url      : sourceUrl field
      domain          : human-readable domain name (e.g. "biology")
    """

    def __init__(self, path: str, domains: List[str] = None, **kwargs):
        """
        Args:
            path:    Path to the directory containing domain JSON files.
            domains: Optional list of domain stems to load (e.g. ["bio", "chem"]).
                     Defaults to all available domains.
        """
        self._filter_domains = domains
        super().__init__(path, **kwargs)

    def load(self) -> None:
        root = getattr(self, "path", None)
        if not root or not os.path.isdir(root):
            logger.error("SafeScientistDataset: path is not a directory: %s", root)
            self.data = []
            return

        all_records: List[Dict[str, Any]] = []

        stems = list(_DOMAIN_MAP.keys())
        if self._filter_domains:
            stems = [s for s in stems if s in self._filter_domains]

        for stem in stems:
            fpath = os.path.join(root, f"{stem}.json")
            if not os.path.isfile(fpath):
                logger.warning("SafeScientistDataset: file not found: %s", fpath)
                continue

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    items = json.load(f)
            except Exception as e:
                logger.error("SafeScientistDataset: failed to load %s: %s", fpath, e)
                continue

            if not isinstance(items, list):
                logger.error("SafeScientistDataset: expected list in %s, got %s", fpath, type(items))
                continue

            domain = _DOMAIN_MAP.get(stem, stem)
            for idx, raw in enumerate(items):
                if not isinstance(raw, dict):
                    logger.warning("SafeScientistDataset: skipping non-dict entry at %s[%d]", fpath, idx)
                    continue

                prompt = str(raw.get("Prompt", "")).strip()
                if not prompt:
                    logger.warning("SafeScientistDataset: empty Prompt at %s[%d], skipping", fpath, idx)
                    continue

                record: Dict[str, Any] = {
                    "id": f"{stem}_{idx}",
                    "prompt": prompt,
                    "task": str(raw.get("Task", "")),
                    "task_description": str(raw.get("Task Description", "")),
                    "risk_type": str(raw.get("Risk Type", "")),
                    "source_url": str(raw.get("sourceUrl", "")),
                    "domain": domain,
                    "domain_stem": stem,
                }
                all_records.append(record)

        self.data = all_records
        logger.info(
            "SafeScientistDataset: loaded %d items from %d domain files at %s",
            len(self.data),
            len(stems),
            root,
        )
