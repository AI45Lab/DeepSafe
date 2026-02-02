                      
"""
示例：# llava-hf/llava-1.5-7b-hf
  python /mnt/shared-storage-user/zhangbo1/MBEF/tools/model_h.py models--Qwen--Qwen-7B-Chat
  python /mnt/shared-storage-user/zhangbo1/MBEF/tools/model_h.py models--mistralai--Mistral-Small-24B-Instruct-2501
  如缓存没有,modelscope下载    LibrAI/longformer-harmful-ro
  modelscope download --model LibrAI/longformer-harmful-ro --local_dir LibrAI/longformer-harmful-ro
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

DEFAULT_HF_CACHE_TO_DIR = "/mnt/shared-storage-user/zhangbo1/tools/hf_cache_to_dir.py"
DEFAULT_REMOTE_PREFIX = "h-ceph:ailab-public-shared/hf_hub"
DEFAULT_LOCAL_ROOT = "/mnt/shared-storage-user/ai4good2-share/models/"

def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)

def _parse_model_cache_name(model_cache_name: str) -> tuple[str, str]:
    """
    models--Org--RepoName  -> (org, repo)
    """
    parts = model_cache_name.split("--")
    if len(parts) < 3 or parts[0] != "models":
        raise ValueError(
            f"模型名格式不对：{model_cache_name}\n"
            "期望格式：models--<Org>--<Repo>"
        )
    org = parts[1].strip()
    repo = "--".join(parts[2:]).strip()
    if not org or not repo:
        raise ValueError(f"模型名格式不对：{model_cache_name}")
    return org, repo

def main() -> int:
    p = argparse.ArgumentParser(description="只传入 model_name（models--Org--Repo），自动下载+转换。")
    p.add_argument("model_name", help="例如：models--Qwen--Qwen1.5-7B-Chat")

    args = p.parse_args()

    org, repo = _parse_model_cache_name(args.model_name)
    local_root = Path(DEFAULT_LOCAL_ROOT)
    cache_dir = local_root / org / args.model_name
    out_dir = local_root / org / repo
    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    src = f"{DEFAULT_REMOTE_PREFIX.rstrip('/')}/{args.model_name}"

    rclone_cmd = [
        "rclone",
        "copy",
        "--progress",
        "--transfers",
        "200",
        "--checkers",
        "200",
        "--links",
        src,
        str(cache_dir),
    ]
    _run(rclone_cmd)

    hf_cache_to_dir = Path(DEFAULT_HF_CACHE_TO_DIR)
    if not hf_cache_to_dir.exists():
        raise FileNotFoundError(f"hf_cache_to_dir.py not found: {hf_cache_to_dir}")

    convert_cmd = [
        sys.executable,
        str(hf_cache_to_dir),
        str(cache_dir),
        str(out_dir),
        "--delete-original",
        "--overwrite",
    ]
    _run(convert_cmd)
    print(f"模型已下载并转换完成: {out_dir}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

