"""Resume official pinned safetensors via HTTP ranges and verify official SHA256."""
import concurrent.futures as cf
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
REPO = "Qwen/Qwen3.5-4B"
MODEL = ROOT / "models_cache" / "models--Qwen--Qwen3.5-4B" / "snapshots" / REVISION
OUT = ROOT / "results/qwen_deepseek_flash_1533_20261011"
CHUNK = 8 * 1024**2


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    raise RuntimeError("Progress file is busy")


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(data)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL.mkdir(parents=True, exist_ok=True)
    metadata = requests.get(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true",
                            timeout=30).json()
    if metadata.get("sha") != REVISION:
        raise RuntimeError("Official revision metadata mismatch")
    weights = [entry for entry in metadata["siblings"] if entry["rfilename"].endswith(".safetensors")]
    state_lock = threading.Lock()
    stopped = threading.Event()
    began = time.monotonic()
    state = {"status": "downloading", "revision": REVISION, "files": [], "error": None}
    jobs = []
    start_bytes = 0
    for weight in weights:
        name, size = weight["rfilename"], weight["size"]
        sha = weight["lfs"]["sha256"]
        target = MODEL / name
        partial = target.with_name(name + ".partial")
        ledger = target.with_name(name + ".ranges.jsonl")
        done = set()
        if target.exists() and target.stat().st_size == size and file_sha(target) == sha:
            item = {"name": name, "total_bytes": size, "downloaded_bytes": size,
                    "sha256": sha, "status": "verified"}
            state["files"].append(item)
            start_bytes += size
            continue
        if partial.exists() and partial.stat().st_size == size and ledger.exists():
            for line in ledger.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("sha256") == sha:
                    done.add(record["chunk"])
        else:
            with partial.open("wb") as stream:
                stream.truncate(size)
        count = math.ceil(size / CHUNK)
        done = {chunk for chunk in done if 0 <= chunk < count}
        downloaded = sum(min(CHUNK, size - chunk * CHUNK) for chunk in done)
        item = {"name": name, "total_bytes": size, "downloaded_bytes": downloaded,
                "sha256": sha, "status": "downloading"}
        state["files"].append(item)
        start_bytes += downloaded
        origin = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
        response = requests.head(origin, allow_redirects=True, timeout=(15,45))
        response.raise_for_status()
        signed = response.url  # Credentials in signed CDN URLs are never logged or written.
        item_lock = threading.Lock()
        for chunk in range(count):
            if chunk not in done:
                jobs.append((signed, partial, ledger, item, item_lock, chunk, size, sha))

    def publish():
        with state_lock:
            state["total_bytes"] = sum(item["total_bytes"] for item in state["files"])
            state["downloaded_bytes"] = sum(item["downloaded_bytes"] for item in state["files"])
            elapsed = time.monotonic() - began
            state["elapsed_seconds"] = round(elapsed, 2)
            state["bytes_per_second"] = (state["downloaded_bytes"] - start_bytes) / max(elapsed, 1)
            state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            atomic_json(OUT / "download_progress.json", state)

    def fetch(job):
        url, partial, ledger, item, lock, chunk, size, sha = job
        start, end = chunk * CHUNK, min(size, (chunk + 1) * CHUNK) - 1
        for attempt in range(4):
            if stopped.is_set():
                return
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                                  stream=True, timeout=(15,45)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise RuntimeError("CDN did not return the requested byte range")
                    data = b"".join(response.iter_content(chunk_size=256 * 1024))
                if len(data) != end - start + 1:
                    raise RuntimeError("Incomplete range response")
                with lock:
                    with partial.open("r+b") as stream:
                        stream.seek(start); stream.write(data); stream.flush()
                    with ledger.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"chunk": chunk, "sha256": sha}) + "\n")
                        stream.flush(); os.fsync(stream.fileno())
                    with state_lock:
                        item["downloaded_bytes"] += len(data)
                return
            except Exception as exc:
                if attempt == 3:
                    # HTTP exception strings can contain signed URLs; keep only the exception type.
                    raise RuntimeError(f"Range {chunk} failed ({type(exc).__name__})") from None
                time.sleep(2**attempt)

    publish()
    print(f"Official revision verified; resuming {len(jobs)} ranges with 8 workers", flush=True)
    try:
        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            pending = {pool.submit(fetch, job) for job in jobs}
            try:
                while pending:
                    done, pending = cf.wait(pending, timeout=2, return_when=cf.FIRST_COMPLETED)
                    for future in done:
                        future.result()
                    publish()
            except BaseException:
                stopped.set()
                for future in pending:
                    future.cancel()
                raise
        state["status"] = "verifying"
        publish()
        for item in state["files"]:
            if item["status"] == "verified":
                continue
            target = MODEL / item["name"]
            partial = target.with_name(item["name"] + ".partial")
            item["status"] = "verifying"
            publish()
            if file_sha(partial) != item["sha256"]:
                raise RuntimeError("Official SHA256 verification failed: " + item["name"])
            os.replace(partial, target)
            item["status"] = "verified"
            publish()
        state["status"] = "complete"
        publish()
        atomic_json(MODEL / "download_verified.json", state)
        print("All official model shards verified: " + str(MODEL), flush=True)
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        publish()
        raise


if __name__ == "__main__":
    main()
