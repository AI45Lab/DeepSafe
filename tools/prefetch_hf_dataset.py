                      
"""
Prefetch a HuggingFace `datasets` dataset into a persistent cache directory.

Why:
- Some cluster jobs cannot reliably access huggingface.co (timeouts).
- If you download once into a shared cache directory, later runs can reuse it offline.

Example:
  python3 tools/prefetch_hf_dataset.py \
    --dataset truthful_qa \
    --config multiple_choice \
    --split validation \
    --cache_dir /mnt/shared-storage-user/zhangbo1/hf_cache

Optional: also persist as a local json file (offline-first workflow)
  python3 tools/prefetch_hf_dataset.py ... --save_json /mnt/shared-storage-user/zhangbo1/MBEF/data/truthful_qa_hf.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="HF dataset name, e.g. truthful_qa")
    p.add_argument("--config", default="", help="HF dataset config name, e.g. multiple_choice")
    p.add_argument("--split", default="validation", help="Split name, e.g. validation")
    p.add_argument("--cache_dir", required=True, help="Directory to store HF cache (persistent)")
    p.add_argument("--save_to_disk", default="", help="Optional: datasets.Dataset.save_to_disk() path")
    p.add_argument("--save_json", default="", help="Optional: write split to a JSON file")
    p.add_argument("--limit", type=int, default=0, help="Optional: only prefetch first N rows")
    return p.parse_args()

def main() -> int:
    args = _parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))

    try:
        from datasets import load_dataset                
    except Exception as e:
        raise SystemExit(f"ERROR: missing dependency 'datasets'. Install it first. err={e!r}")

    name = args.dataset
    cfg = args.config if args.config else None
    split = args.split

    print(f"[prefetch] dataset={name!r} config={cfg!r} split={split!r}")
    print(f"[prefetch] cache_dir={str(cache_dir)}")
    print(f"[prefetch] HF_HOME={os.environ.get('HF_HOME')}")
    print(f"[prefetch] HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE')}")

    ds = load_dataset(name, cfg, split=split, cache_dir=str(cache_dir))
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    _ = ds[0] if len(ds) > 0 else None
    print(f"[prefetch] loaded rows={len(ds)}")

    if args.save_to_disk:
        out = Path(args.save_to_disk).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(out))
        print(f"[prefetch] saved to disk: {out}")

    if args.save_json:
        outj = Path(args.save_json).expanduser().resolve()
        outj.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = [dict(r) for r in ds]
        with outj.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"[prefetch] saved json: {outj} (rows={len(rows)})")

    print("[prefetch] done")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

