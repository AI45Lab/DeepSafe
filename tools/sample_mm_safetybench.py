                      

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Tuple

import pandas as pd

def _list_parquets(src_root: str, categories: List[str] | None, splits: List[str] | None) -> List[Tuple[str, str, str]]:
    data_root = os.path.join(src_root, "data")
    out: List[Tuple[str, str, str]] = []
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"MM-SafetyBench data dir not found: {data_root}")

    for cat in sorted(os.listdir(data_root)):
        if categories and cat not in categories:
            continue
        cat_dir = os.path.join(data_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        for fn in sorted(os.listdir(cat_dir)):
            if not fn.endswith(".parquet"):
                continue
            split = fn[: -len(".parquet")]
            if splits and split not in splits:
                continue
            out.append((cat, split, os.path.join(cat_dir, fn)))
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="MM-SafetyBench root (contains data/)")
    ap.add_argument("--dst_root", required=True, help="Output root (will create data/)")
    ap.add_argument("--n", type=int, default=100, help="Total samples to write")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", type=str, default="", help="Comma-separated splits to include (optional)")
    ap.add_argument("--categories", type=str, default="", help="Comma-separated categories to include (optional)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()] or None
    categories = [c.strip() for c in args.categories.split(",") if c.strip()] or None

    parquets = _list_parquets(args.src_root, categories, splits)
    if not parquets:
        raise RuntimeError("No parquet files found for the given filters.")

    rows: List[Dict] = []
    row_origin: List[Tuple[str, str]] = []
    for cat, split, path in parquets:
        df = pd.read_parquet(path)
        if df.empty:
            continue
        for _, r in df.iterrows():
            rows.append(r.to_dict())
            row_origin.append((cat, split))

    if not rows:
        raise RuntimeError("No rows loaded from parquet files.")

    n = min(args.n, len(rows))
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    idxs = idxs[:n]
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for i in idxs:
        key = row_origin[i]
        grouped.setdefault(key, []).append(rows[i])

    for (cat, split), rs in grouped.items():
        out_dir = os.path.join(args.dst_root, "data", cat)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{split}.parquet")
        out_df = pd.DataFrame(rs)
        out_df.to_parquet(out_path, index=False)
    os.makedirs(args.dst_root, exist_ok=True)
    with open(os.path.join(args.dst_root, "SAMPLE_MANIFEST.txt"), "w", encoding="utf-8") as f:
        f.write(f"src_root={args.src_root}\n")
        f.write(f"n={n}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"splits={splits or 'ALL'}\n")
        f.write(f"categories={categories or 'ALL'}\n")
        f.write("files:\n")
        for (cat, split), rs in sorted(grouped.items()):
            f.write(f"  - data/{cat}/{split}.parquet: {len(rs)} rows\n")

    print(f"Wrote sample dataset to: {args.dst_root} (n={n})")

if __name__ == "__main__":
    main()

