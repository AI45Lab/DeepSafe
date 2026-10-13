"""Standard-library data, identity, locking and durable checkpoint utilities."""
import hashlib
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path

LABELS = ("high", "medium", "low")
FIELDS = ("question", "target_comment", "answer", "original_context")
ROOT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def model_directory_identity(path, include_weights=True):
    """Hash actual inference artifacts, including index-listed shards. No weight loading."""
    root=Path(path).resolve()
    patterns=("config.json","generation_config.json","tokenizer*","chat_template*",
              "special_tokens_map.json","added_tokens.json","vocab*","merges*",
              "*.model","preprocessor_config.json","processor_config.json")
    files={p for pattern in patterns for p in root.glob(pattern) if p.is_file()}
    if include_weights:
        for pattern in ("*.safetensors","pytorch_model*.bin","model*.bin","*.index.json"):
            files.update(p for p in root.glob(pattern) if p.is_file())
        for index in tuple(p for p in files if p.name.endswith(".index.json")):
            for name in set(read_json(index).get("weight_map",{}).values()):
                relative=Path(name)
                shard=root/relative
                # HF snapshots legitimately symlink their shards to cache blobs.
                if relative.is_absolute() or ".." in relative.parts or not shard.is_file():
                    raise ValueError("Missing or out-of-directory model shard: "+str(shard))
                files.add(shard)
        if not any(p.suffix in (".safetensors",".bin") for p in files):
            raise ValueError("No model weight artifacts in: "+str(root))
    if not files:
        raise ValueError("No inference artifacts in: "+str(root))
    return {str(p.relative_to(root)).replace("\\","/"):file_hash(p) for p in sorted(files)}


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError(f"Invalid JSONL line {number}: {path}") from exc
    return rows


def load_samples(path):
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("sample_id"), (str, int)):
            raise ValueError("Each row needs sample_id")
        sid = str(row["sample_id"])
        if not sid or sid in seen:
            raise ValueError("Empty/duplicate sample_id: " + sid)
        seen.add(sid)
        if any(not isinstance(row.get(k), str) or not row[k].strip()
               for k in ("question", "answer", "original_context")):
            raise ValueError("Question, answer and original_context require nonempty text: " + sid)
        if "target_comment" not in row or (row["target_comment"] is not None
                                            and not isinstance(row["target_comment"], str)):
            raise ValueError("target_comment must be text or null for document QA: " + sid)
    if not rows:
        raise ValueError("Empty dataset")
    return rows


def blind(row):
    return {k: row[k] for k in FIELDS}


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def parse_json(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    # Reasoning, if explicitly enabled, is usage-accounted but not treated as evidence.
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        return None


@contextmanager
def output_lock(directory):
    """An OS lock is automatically released on process death; no stale lock deletion."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Output directory is in use by another process") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Checkpoint:
    def __init__(self, path, identity, input_hashes):
        self.path, self.identity, self.input_hashes = Path(path), identity, input_hashes
        self.events, self.latest = [], {}
        self._lock = threading.RLock()
        if self.path.exists():
            with self.path.open("rb+") as stream:
                raw = stream.read()
                offset = 0
                for part in raw.splitlines(keepends=True):
                    try:
                        event = json.loads(part.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError):
                        if offset + len(part) == len(raw) and not part.endswith(b"\n"):
                            stream.seek(offset)
                            stream.truncate()
                            break
                        raise ValueError("Corrupt checkpoint; non-final line cannot be discarded")
                    sid = event.get("sample_id")
                    if (event.get("identity") != identity or sid not in input_hashes
                            or event.get("input_hash") != input_hashes[sid]):
                        raise ValueError("Checkpoint data/model/prompt mismatch; use another output directory")
                    self.events.append(event)
                    self.latest[(sid, event["stage"])] = event
                    offset += len(part)
                else:
                    if raw and not raw.endswith(b"\n"):
                        stream.seek(0, os.SEEK_END)
                        stream.write(b"\n")

    def get(self, sid, stage):
        with self._lock:
            return self.latest.get((str(sid), stage))

    def record(self, sid, stage, **value):
        sid = str(sid)
        event = {"sample_id": sid, "stage": stage, "identity": self.identity,
                 "input_hash": self.input_hashes[sid], **value}
        # One durable append at a time, including the in-memory publication.
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.events.append(event)
            self.latest[(sid, stage)] = event
        return event
