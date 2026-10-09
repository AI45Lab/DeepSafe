"""Input loading, stable hashes, and triplet-safe selection."""

import hashlib
import json
import random
from pathlib import Path


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def choose_samples(rows, split, seed, limit, order="triplet"):
    groups = sorted({str(row["triplet_id"]) for row in rows})
    random.Random(seed).shuffle(groups)
    train_end = int(len(groups) * .70)
    dev_end = train_end + int(len(groups) * .15)
    parts = {"train": groups[:train_end], "dev": groups[train_end:dev_end],
             "test": groups[dev_end:], "all": groups}
    order_by_group = {group: index for index, group in enumerate(groups)}
    selected = [row for row in rows if str(row["triplet_id"]) in set(parts[split])]
    if order == "triplet":
        selected.sort(key=lambda row: (order_by_group[str(row["triplet_id"])], str(row["sample_id"])))
    elif order != "file":
        raise ValueError("Unknown sample order: " + order)
    return selected[:limit] if limit else selected


