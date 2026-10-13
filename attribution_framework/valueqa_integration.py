"""DeepSafe model transport and resumable execution for the audited ValueQA methods."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from . import single_llm
from .valueqa.checkpoint import Checkpoint
from .valueqa.data import sha
from .valueqa.pipeline import run_samples
from .valueqa.pipeline_evidence import process_sample
from .valueqa.prompts_evidence import load_rules, prompt, PROMPT_VERSION
from .valueqa.prompts import INPUT_KEYS, parse_json
from .valueqa.prompts_quality import valid_scores
from .valueqa.prompts_review import valid_review
from .valueqa.adjudication import valid_adjudication
from .valueqa_numbers import canonical_numbers

ROOT = Path(__file__).resolve().parent


class RecordedAPIError(RuntimeError):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def engine_sha256():
    files = [Path(__file__), ROOT / "single_llm.py", ROOT / "valueqa_numbers.py",
             *sorted((ROOT / "valueqa").glob("*.py")), *sorted((ROOT / "valueqa_rules").iterdir())]
    return sha({str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files if path.is_file()})


def model_info(model):
    return {"model": getattr(model, "model_name", type(model).__name__),
            "api_base": str(getattr(model, "api_base", "")),
            "parameters": getattr(model, "default_gen_kwargs", {})}


def normalize_framework_output(raw):
    """Repair mechanical formatting only; preserve labels, reasons and quoted evidence."""
    parsed = parse_json(raw)
    repairs, numeric_repairs = [], []
    if isinstance(parsed, dict) and "verdict" in parsed:
        normalized = dict(parsed)
        if set(normalized) == {"verdict", "reason"}:
            normalized["issues"] = []
            repairs.append("issues: missing field defaulted to empty array; low still requires localization")
        validator = valid_review
    else:
        stage = "agent3" if isinstance(parsed, dict) and "scores" in parsed else "agent1"
        normalized, numeric_repairs = canonical_numbers(parsed, stage)
        repairs.extend(numeric_repairs)
        validator = (lambda value: valid_scores(value, 5)) if stage == "agent3" else valid_adjudication
    if not validator(normalized):
        return raw, {"output_schema_valid": False}
    metadata = {"output_schema_valid": True}
    reason = normalized.get("reason")
    if isinstance(reason, str) and len(reason.strip()) > 100:
        metadata["output_warnings"] = [{"field": "reason", "actual_characters": len(reason.strip()),
                                        "recommended_max_characters": 100, "action": "preserved in full"}]
    if repairs:
        metadata.update(normalized_output=normalized, output_repairs=repairs)
    if numeric_repairs:
        metadata["numeric_type_repairs"] = numeric_repairs
    return json.dumps(normalized, ensure_ascii=False) if repairs else raw, metadata


class APIRequester:
    def __init__(self, model, directory, identity, method):
        if not hasattr(model, "client"):
            raise TypeError("ValueQA requires a DeepSafe APIModel with a chat-completions client")
        if getattr(model.client, "max_retries", 0) != 0:
            raise ValueError("Set model.max_retries=0; ValueQA records and controls request retries itself")
        self.model, self.directory, self.identity, self.method = model, directory, identity, method
        self.lock = threading.Lock()

    def __call__(self, messages):
        return self.send(messages)

    def send(self, messages, sample_id=None):
        settings = dict(getattr(self.model, "default_gen_kwargs", {}))
        extra = dict(settings.pop("extra_body", {}) or {})
        temperature, maximum = settings.pop("temperature", 0), settings.pop("max_tokens", 2048)
        extra.update({k: settings.pop(k) for k in list(settings) if k in ("thinking", "enable_thinking")})
        if set(extra) & {"model", "messages", "temperature", "max_tokens"}:
            raise ValueError("Extra API parameters cannot overwrite the model-input contract")
        options = {"model": self.model.model_name, "messages": messages,
                   "temperature": temperature, "max_tokens": maximum, **settings}
        payload = {**options, **extra}
        event = {"run_sha256": self.identity, "sample_id": sample_id, "method": self.method,
                 "started_utc": datetime.now(timezone.utc).isoformat(), "messages": messages,
                 "messages_sha256": sha(messages), "request_sha256": sha(payload),
                 "endpoint": self.model.api_base.rstrip("/") + "/chat/completions",
                 "requested_model": self.model.model_name}
        began = time.monotonic()
        try:
            response = self.model.client.chat.completions.create(**options, extra_body=extra)
            raw = response.choices[0].message.content
            raw = raw if isinstance(raw, str) else ""
            usage = response.usage
            tokens = {"prompt": getattr(usage, "prompt_tokens", 0) or 0,
                      "completion": getattr(usage, "completion_tokens", 0) or 0}
            event.update(status="response_received", raw_output=raw, response_model=response.model,
                         finish_reason=response.choices[0].finish_reason,
                         usage=usage.model_dump() if usage is not None else {})
            if hasattr(response, "model_dump_json"):
                event["raw_response"] = response.model_dump_json()
            text = raw
            if self.method == "framework":
                text, normalization = normalize_framework_output(raw)
                event.update(normalization)
                if normalization.get("numeric_type_repairs"):
                    tokens["numeric_type_repairs"] = len(normalization["numeric_type_repairs"])
            return text, tokens
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            error = type(exc).__name__ + ": " + str(exc)
            key = getattr(self.model.client, "api_key", None)
            if isinstance(key, str) and key:
                error = error.replace(key, "[redacted]")
            event.update(status="request_failed", http_status=status, error=error)
            if status in (401, 402, 403, 404):
                (self.directory / ".pause").touch()
            raise RecordedAPIError(error, status) from None
        finally:
            event["elapsed_seconds"] = round(time.monotonic() - began, 3)
            with self.lock, (self.directory / "api_audit.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())


class ValueQAOrchestrator:
    def __init__(self, model, method, state_dir, workers=10, attempts=2, recovery_passes=3):
        if method not in ("framework", "single_llm"):
            raise ValueError("Unknown ValueQA method")
        if min(workers, attempts, recovery_passes) < 1:
            raise ValueError("workers, attempts and recovery_passes must be positive")
        self.model, self.method, self.directory = model, method, Path(state_dir)
        model_concurrency = int(getattr(model, "concurrency", workers))
        if model_concurrency < 1:
            raise ValueError("Model concurrency must be positive")
        self.workers = min(int(workers), model_concurrency)
        self.attempts, self.recovery_passes = int(attempts), int(recovery_passes)

    def classify(self, samples):
        samples = list(samples)
        ids = [str(row["sample_id"]) for row in samples]
        if not samples or len(set(ids)) != len(ids):
            raise ValueError("ValueQA requires a nonempty dataset with unique IDs")
        for row in samples:
            single_llm.messages(row)
            if "label" in row or "annotations" in row:
                raise ValueError("Reference labels and spans cannot enter the classifier")
        hashes = {str(row["sample_id"]): sha(row) for row in samples}
        manifest = {"method": self.method, "engine_sha256": engine_sha256(), "model": model_info(self.model),
                    "input_sha256": hashes, "selected_ids": ids, "workers": self.workers,
                    "attempts": self.attempts, "recovery_passes": self.recovery_passes,
                    "prompt_version": PROMPT_VERSION if self.method == "framework" else "single_llm_one_response_v1"}
        self.directory.mkdir(parents=True, exist_ok=True)
        with single_llm.directory_lock(self.directory):
            path = self.directory / "experiment_manifest.json"
            if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Code, inputs or model settings changed; use a new state_dir")
            single_llm.atomic_json(path, manifest)
            identity = sha(manifest)
            requester = APIRequester(self.model, self.directory, identity, self.method)
            self.run_info = {"method": "valueqa_" + self.method, "engine_sha256": manifest["engine_sha256"],
                             "model": self.model.model_name, "api_base": self.model.api_base,
                             "prompt_version": manifest["prompt_version"], "run_sha256": identity,
                             "parameters": model_info(self.model)["parameters"]}
            if self.method == "framework":
                return self._framework(samples, hashes, identity, requester)
            return self._single(samples, requester)

    def _framework(self, samples, hashes, identity, requester):
        rules = load_rules(ROOT / "valueqa_rules")
        for row in samples:
            prompt(row, rules, "agent1")
        checkpoint = Checkpoint(self.directory / "checkpoint.jsonl", identity, hashes)
        for _ in range(self.recovery_passes):
            if (self.directory / ".pause").exists():
                break
            _, _, interrupted = run_samples(samples, rules, requester, checkpoint, self.directory / ".pause",
                                             self.workers, self.attempts, prompt, process_sample)
            if not interrupted:
                break
        return {str(row["sample_id"]): {"final": checkpoint.events.get(str(row["sample_id"]), {}).get("final"),
                                        "stages": checkpoint.events.get(str(row["sample_id"]), {})} for row in samples}

    def _single(self, samples, requester):
        audit = self.directory / "api_audit.jsonl"
        received = {}
        if audit.exists():
            with audit.open("r+b") as stream:
                while True:
                    position, line = stream.tell(), stream.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        stream.truncate(position)
                        break
                    event = json.loads(line)
                    if event["run_sha256"] != requester.identity:
                        raise ValueError("Single-LLM audit run identity mismatch")
                    if event["status"] == "response_received":
                        sid = str(event["sample_id"])
                        if sid in received:
                            raise ValueError("Multiple returned responses for " + sid)
                        received[sid] = event["raw_output"]

        def request(row):
            sid = str(row["sample_id"])
            for attempt in range(self.attempts):
                if (self.directory / ".pause").exists():
                    return sid, None
                try:
                    raw, _ = requester.send(single_llm.messages(row), sid)
                    return sid, raw  # Keep every returned response, including invalid JSON.
                except Exception as exc:
                    status = getattr(exc, "status_code", None)
                    if status is not None and 400 <= status < 500 and status != 429:
                        break
                    if attempt + 1 < self.attempts:
                        time.sleep(min(2 ** attempt, 4))
            return sid, None

        pending = [row for row in samples if str(row["sample_id"]) not in received]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for sid, raw in pool.map(request, pending):
                if raw is not None:
                    received[sid] = raw
        return {str(row["sample_id"]): {"raw_output": received.get(str(row["sample_id"])),
                                        "response_received": str(row["sample_id"]) in received} for row in samples}
