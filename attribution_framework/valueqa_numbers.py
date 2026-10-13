"""Lossless numeric normalization used by the audited framework."""
import copy
from .valueqa.prompts_quality import QUALITY_KEYS


def canonical_numbers(output, stage):
    """Lossless JSON type repair only. Never infer or replace a grade/reason."""
    result = copy.deepcopy(output)
    changes = []

    def integer(container, key, maximum):
        value = container.get(key)
        if (isinstance(value, str) and value.isascii() and value.isdecimal()
                and str(int(value)) == value and 0 <= int(value) <= maximum):
            container[key] = int(value)
            changes.append(key + ": numeric string to integer")

    if not isinstance(result, dict):
        return result, changes
    if stage in ("agent1", "agent1_recheck") or (stage == "agent3" and "label" in result):
        integer(result, "confidence", 100)
    elif stage == "agent3" and isinstance(result.get("scores"), dict):
        for key in QUALITY_KEYS:
            if isinstance(result["scores"].get(key), dict):
                integer(result["scores"][key], "score", 5)
    return result, changes
