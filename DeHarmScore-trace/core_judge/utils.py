from __future__ import annotations

import json
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import yaml

from .schemas import AppConfig


_TRACE_LOCAL = threading.local()


def _load_input_file(file_path: Path) -> dict[str, Any]:
    if file_path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported input file format: {file_path}. Input files must use .json.")
    loaded = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Input file must contain a mapping object: {file_path}")
    return loaded


def _resolve_input_config(raw_input: Any, config_dir: Path) -> dict[str, Any]:
    if isinstance(raw_input, str):
        return _load_input_file((config_dir / raw_input).resolve())
    if isinstance(raw_input, dict):
        input_path = raw_input.get("path") or raw_input.get("file") or raw_input.get("input_file")
        if input_path is not None:
            if not isinstance(input_path, str):
                raise ValueError("'input.path' must be a string file path.")
            return _load_input_file((config_dir / input_path).resolve())
        return raw_input
    raise ValueError("Missing or invalid 'input' section in YAML config.")


def _resolve_search_paths(raw_config: dict[str, Any], config_dir: Path) -> None:
    search_data = raw_config.get("search")
    if not isinstance(search_data, dict):
        return
    for key in ("artifact_dir", "cache_dir"):
        value = search_data.get(key)
        if value not in (None, ""):
            search_data[key] = str((config_dir / str(value)).resolve())


def load_config(config_path: str | Path) -> AppConfig:
    config_file = Path(config_path)
    raw_config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_config, dict):
        raise ValueError("Config YAML must contain a mapping object.")
    if "input" in raw_config:
        raw_input = raw_config.get("input")
        if raw_input not in (None, ""):
            raw_config["input"] = _resolve_input_config(raw_input, config_file.parent)
    _resolve_search_paths(raw_config, config_file.parent)
    return AppConfig.from_dict(raw_config)


@dataclass
class RunReporter:
    show_progress: bool = True
    show_timing: bool = True
    prefix: str = ""
    output_lock: Lock | None = None
    timings: dict[str, float] = field(default_factory=dict)
    _counter: int = 0

    def log(self, message: str) -> None:
        if self.show_progress:
            self._emit(message)

    def run_stage(self, key: str, label: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._counter += 1
        prefix = f"[{self._counter}] {label}"
        if self.show_progress:
            self._emit(f"{prefix}...")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        self.timings[key] = elapsed
        if self.show_progress:
            suffix = f" done in {elapsed:.2f}s" if self.show_timing else " done"
            self._emit(f"{prefix}{suffix}")
        return result

    def _emit(self, message: str) -> None:
        lines = message.splitlines() or [message]
        rendered = "\n".join(f"{self.prefix} {line}" if self.prefix else line for line in lines)
        if self.output_lock is None:
            print(rendered, file=sys.stderr, flush=True)
            return
        with self.output_lock:
            print(rendered, file=sys.stderr, flush=True)


@dataclass
class PromptTraceWriter:
    output_path: Path
    output_lock: Lock = field(default_factory=Lock)

    def write(self, payload: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False)
        with self.output_lock:
            with self.output_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


@contextmanager
def trace_context(writer: PromptTraceWriter | None):
    previous = getattr(_TRACE_LOCAL, "writer", None)
    _TRACE_LOCAL.writer = writer
    try:
        yield
    finally:
        _TRACE_LOCAL.writer = previous


def trace_prompt_response(stage: str, prompt: str, response: str, **metadata: Any) -> None:
    writer = getattr(_TRACE_LOCAL, "writer", None)
    if writer is None:
        return
    payload = {
        "stage": stage,
        "prompt": prompt,
        "response": response,
    }
    if metadata:
        payload["metadata"] = metadata
    writer.write(payload)


def strip_json_fence(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


__all__ = [
    "PromptTraceWriter",
    "RunReporter",
    "load_config",
    "strip_json_fence",
    "trace_context",
    "trace_prompt_response",
]
