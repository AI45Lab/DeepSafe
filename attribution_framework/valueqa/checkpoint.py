"""Durable, per-agent JSONL checkpoints."""

import json
import os
import threading


class Checkpoint:
    def __init__(self, path, run_hash, input_hashes):
        self.path, self.run_hash, self.input_hashes = path, run_hash, input_hashes
        self.lock = threading.Lock()
        self.events = {}
        self.all_events = []
        if path.exists():
            with path.open("rb+") as stream:
                raw = stream.read()
                chunks = raw.splitlines(keepends=True)
                offset = 0
                for number, chunk in enumerate(chunks, 1):
                    line_start = offset
                    try:
                        event = json.loads(chunk.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        if number == len(chunks) and not chunk.endswith(b"\n"):
                            # A process may have stopped midway through the final append.
                            stream.seek(offset)
                            stream.truncate()
                            break
                        raise ValueError("Invalid checkpoint line " + str(number)) from exc
                    offset += len(chunk)
                    try:
                        sid = str(event["sample_id"])
                        if (sid not in input_hashes or event["input_sha256"] != input_hashes[sid] or
                                event["run_sha256"] != run_hash):
                            raise ValueError("Checkpoint data/config mismatch at line " + str(number))
                        self.events.setdefault(sid, {})[event["stage"]] = event
                        self.all_events.append(event)
                    except (KeyError, TypeError) as exc:
                        if number == len(chunks) and not chunk.endswith(b"\n"):
                            stream.seek(line_start)
                            stream.truncate()
                            break
                        raise ValueError("Invalid checkpoint line " + str(number)) from exc
                if chunks and offset == len(raw) and not raw.endswith(b"\n"):
                    stream.seek(0, os.SEEK_END)
                    stream.write(b"\n")

    def record(self, sid, stage, result):
        event = {"sample_id": sid, "stage": stage, "input_sha256": self.input_hashes[sid],
                 "run_sha256": self.run_hash, **result}
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.events.setdefault(sid, {})[stage] = event
            self.all_events.append(event)
        return event


