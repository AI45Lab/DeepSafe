"""Qwen initial checkpoint -> independent evidence review -> deterministic acceptance gate."""
import argparse
import concurrent.futures as futures
import importlib.metadata
import json
import signal
import subprocess
import sys
import time
import threading
from pathlib import Path

if __package__:
    from .backend import APIBackend, BackendError, LocalBackend, ReviewAPIBackend, ReviewVLLMBackend
    from .common import (Checkpoint, LABELS, ROOT, blind, digest, file_hash, load_samples,
                        model_directory_identity, output_lock, parse_json, read_json, read_jsonl, write_json, write_jsonl)
    from .evaluate import report
    from .gate import decide, gate_identity, validate_policy
    from .prompts import review_messages
    from .boundary_contract import bind_single_review
    from .routing import thresholds, route_initial, select_review
else:
    from backend import APIBackend, BackendError, LocalBackend, ReviewAPIBackend, ReviewVLLMBackend
    from common import (Checkpoint, LABELS, ROOT, blind, digest, file_hash, load_samples,
                        model_directory_identity, output_lock, parse_json, read_json, read_jsonl, write_json, write_jsonl)
    from evaluate import report
    from gate import decide, gate_identity, validate_policy
    from prompts import review_messages
    from boundary_contract import bind_single_review
    from routing import thresholds, route_initial, select_review


def path_from_config(value, config_path):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def load_config(path, overrides=None):
    path = Path(path).resolve()
    config = read_json(path)
    import os
    for key, variable in (("base_model","QWEN_BASE_PATH"),("adapter_path","QWEN_LORA_PATH"),
                          ("review_vllm_python","VLLM_PYTHON"),
                          ("review_api_base","REVIEW_API_BASE"),("review_api_model","REVIEW_API_MODEL"),
                          ("review_local_model","REVIEW_MODEL_PATH")):
        if os.environ.get(variable):
            config[key] = os.environ[variable]
            # Explicit model directories supplied through the environment are
            # relative to the invoking shell; JSON paths are config-relative.
            if key in ("base_model", "review_local_model") and Path(config[key]).is_dir():
                config[key] = str(Path(config[key]).resolve())
    config.update({k:v for k,v in (overrides or {}).items() if v is not None})
    config.setdefault("review_backend","api")
    config["review_confidence_thresholds"] = thresholds(config)
    config.setdefault("api_transport", "requests")
    config.setdefault("review_api_transport", "requests")
    if any(config[key] not in ("requests", "urllib", "curl") for key in ("api_transport", "review_api_transport")):
        raise ValueError("HTTP transport must be requests, urllib or curl")
    config.setdefault("review_api_workers", 5)
    if type(config["review_api_workers"]) is not int or config["review_api_workers"] < 1:
        raise ValueError("review_api_workers must be a positive integer")
    if config["review_backend"] not in ("vllm","api"):
        raise ValueError("Independent review requires a different model: review_backend must be api or vllm")
    if config["review_use_adapter"]:
        raise ValueError("The classification LoRA is for initial classification only, not evidence review")
    if config["review_backend"]=="api":
        from urllib.parse import urlparse
        endpoint=urlparse(config.get("review_api_base",""))
        if (endpoint.scheme not in ("http","https") or not endpoint.hostname
                or not isinstance(config.get("review_api_model"),str)
                or not config["review_api_model"].strip()
                or "CHANGE_ME" in config["review_api_model"]):
            raise ValueError("API review requires review_api_base and review_api_model")
        if not isinstance(config.get("review_api_extra_body"),dict):
            raise ValueError("review_api_extra_body must be an object")
        if set(config["review_api_extra_body"]) & {"model","messages","temperature","max_tokens"}:
            raise ValueError("review_api_extra_body cannot override standard request fields")
        if config["review_api_extra_body"].get("stream", False) is not False:
            raise ValueError("Streaming responses are unsupported; omit stream or set stream=false")
        timeout=config.get("review_api_timeout")
        if type(timeout) not in (int,float) or not 0 < timeout < float("inf"):
            raise ValueError("review_api_timeout must be positive and finite")
    if config["review_backend"]=="vllm":
        if sys.platform != "linux":
            raise ValueError("Offline vLLM requires Linux (including WSL2); use API review on native Windows")
        reviewer=config.get("review_local_model")
        if not isinstance(reviewer,str) or not reviewer.strip() or "CHANGE_ME" in reviewer:
            raise ValueError("vLLM review requires an explicit review_local_model different from the initial Qwen")
        for key in ("review_vllm_startup_timeout","review_vllm_tensor_parallel_size"):
            if type(config.get(key)) is not int or config[key]<1:
                raise ValueError("Positive integer required: "+key)
        utilization=config.get("review_vllm_gpu_memory_utilization")
        if type(utilization) not in (int,float) or not 0 < utilization <= 1:
            raise ValueError("review_vllm_gpu_memory_utilization must be in (0,1]")
        if type(config.get("review_vllm_enforce_eager")) is not bool:
            raise ValueError("review_vllm_enforce_eager must be boolean")
    for key in ("adapter_path", "cache_dir", "input", "output_dir", "references"):
        if config.get(key):
            config[key] = str(path_from_config(config[key], path))
    for key in ("review_local_model","review_local_tokenizer"):
        if config.get(key) and path_from_config(config[key],path).is_dir():
            config[key]=str(path_from_config(config[key],path))
    adapter = Path(config["adapter_path"])
    if not (adapter / "adapter_config.json").is_file() and (adapter / "best_adapter").is_dir():
        adapter = adapter / "best_adapter"
        config["adapter_path"] = str(adapter)
    for name in ("adapter_config.json", "adapter_model.safetensors", "classification_schema.json",
                 "training_config.json", "tokenizer.json", "chat_template.jinja"):
        if not (adapter / name).is_file():
            raise ValueError("Missing adapter artifact: " + str(adapter / name))
    if path_from_config(config["base_model"], path).is_dir():
        config["base_model"] = str(path_from_config(config["base_model"], path))
    training = read_json(adapter / "training_config.json")
    ac = read_json(adapter / "adapter_config.json")
    if not Path(config["base_model"]).is_dir() and config["base_model"] != ac["base_model_name_or_path"]:
        raise ValueError("Base model differs from adapter training base")
    if config["base_revision"] != training["model_revision"]:
        raise ValueError("Base revision differs from adapter training revision")
    def model_key(value):
        if not isinstance(value,str):
            return None
        candidate=path_from_config(value,path)
        if candidate.is_dir():
            return os.path.normcase(str(candidate.resolve()))
        return value.strip().replace("\\", "/").rstrip("/").casefold()
    reviewer=config["review_api_model"] if config["review_backend"]=="api" else config["review_local_model"]
    initial_models=[config["base_model"],ac["base_model_name_or_path"],config["adapter_path"]]
    if config["backend"]=="api":
        initial_models.append(config.get("api_model"))
    if model_key(reviewer) in {model_key(value) for value in initial_models if value}:
        raise ValueError("Review model must differ from the initial Qwen model (model ID/path matches)")
    if config["backend"] not in ("transformers", "api"):
        raise ValueError("backend must be transformers or api")
    if config["mode"] not in ("observe", "experimental", "calibrated"):
        raise ValueError("Invalid mode")
    for k in ("initial_attempts", "review_attempts", "review_max_input_tokens", "review_max_new_tokens", "review_max_time_seconds"):
        if type(config[k]) is not int or config[k] < 1:
            raise ValueError("Positive integer required: " + k)
    if config["initial_attempts"] > 2 or config["review_attempts"] != 1:
        raise ValueError("Initial attempts <=2; review attempts must be 1 (no automatic review loops)")
    for k in ("local_files_only", "review_enable_thinking", "review_use_adapter"):
        if type(config[k]) is not bool:
            raise ValueError("Boolean required: " + k)
    return config


def engine_identity(config):
    # Gate mode, policy, references, output path and per-run cap do not change model tasks.
    excluded = {"mode", "input", "references", "output_dir", "policy", "review_api_workers"}
    inactive_prefixes = []
    if config["backend"] == "transformers":
        inactive_prefixes.append("api_")
    if config["review_backend"] == "api":
        inactive_prefixes.extend(("review_local_", "review_vllm_"))
        excluded.update({"review_enable_thinking", "review_max_input_tokens", "review_max_time_seconds"})
    else:
        inactive_prefixes.append("review_api_")
    settings = {k:v for k,v in config.items()
                if k not in excluded and not any(k.startswith(prefix) for prefix in inactive_prefixes)}
    adapter = Path(config["adapter_path"])
    names = ("adapter_config.json", "adapter_model.safetensors", "classification_schema.json",
             "training_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    artifacts = {name:file_hash(adapter/name) for name in names}
    local_artifacts={}
    hashed={}
    roles=[("base_model",config["base_model"],True)]
    if config["review_backend"]=="vllm":
        reviewer=config["review_local_model"]
        tokenizer=config.get("review_local_tokenizer") or reviewer
        roles.extend((("review_model",reviewer,True),("review_tokenizer",tokenizer,False)))
    for role,value,weights in roles:
        if Path(value).is_dir():
            key=(str(Path(value).resolve()),weights)
            if key not in hashed:
                hashed[key]=model_directory_identity(value,include_weights=weights)
            local_artifacts[role]=hashed[key]
    source = {name:file_hash(ROOT/name) for name in
              ("common.py", "backend.py", "initial_protocol.py", "prompts.py", "review_prompt.txt",
               "boundary_contract.py", "routing.py",
               "classification_prompt_strict.txt", "run.py")}
    if config["review_backend"] == "vllm":
        source["vllm_worker.py"]=file_hash(ROOT/"vllm_worker.py")
    packages = {}
    package_names = []
    if config["backend"] == "transformers":
        package_names.extend(("torch", "transformers", "peft", "accelerate"))
        if config["quantization"] == "4bit":
            package_names.append("bitsandbytes")
    if config["review_backend"] == "vllm" and not config.get("review_vllm_python"):
        package_names.extend(("vllm", "torch", "transformers", "tokenizers"))
    for name in dict.fromkeys(package_names):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    review_packages=None
    if config["review_backend"]=="vllm" and config.get("review_vllm_python"):
        # Fingerprint the separate interpreter without importing CUDA or loading weights.
        probe="""import importlib.metadata as m,json,sys
values={'python':sys.version}
for name in ('vllm','torch','transformers','tokenizers'):
    try: values[name]=m.version(name)
    except m.PackageNotFoundError: values[name]=None
print(json.dumps(values))
"""
        result=subprocess.run([config["review_vllm_python"],"-c",probe],capture_output=True,
                              text=True,encoding="utf-8",timeout=20,check=True)
        review_packages=json.loads(result.stdout)
    return digest({"settings":settings, "adapter":artifacts, "source":source,
                   "packages":packages,"review_packages":review_packages,"local_models":local_artifacts})


def import_initials(path, rows, checkpoint):
    allowed = {str(r["sample_id"]) for r in rows}
    actual_hashes={str(r["sample_id"]):digest(blind(r)) for r in rows}
    source_hash=file_hash(path)
    seen = set()
    prepared = []
    # Validate the entire file before importing anything; a bad later record must not
    # leave a partially imported checkpoint that is reused on the next invocation.
    for item in read_jsonl(path):
        if not isinstance(item, dict):
            raise ValueError("Imported initial must be a prediction JSON object")
        sid = str(item.get("sample_id", ""))
        if sid not in allowed or sid in seen:
            raise ValueError("Imported initial has unknown/duplicate ID: " + sid)
        seen.add(sid)
        prediction_fields = [key for key in ("predicted_level", "predicted_label", "initial_label") if key in item]
        if not prediction_fields:
            raise ValueError("Explicit prediction field required; evaluation/reference label cannot be imported as predictions: " + sid)
        label = item[prediction_fields[0]]
        if any(item[key] != label for key in prediction_fields):
            raise ValueError("Conflicting imported prediction fields: " + sid)
        if label not in LABELS:
            raise ValueError("Imported initial label invalid: " + sid)
        if item.get("input_sha256") and item["input_sha256"]!=actual_hashes[sid]:
            raise ValueError("Imported prediction input hash mismatch: "+sid)
        old = checkpoint.get(sid, "initial")
        if old:
            if old.get("output", {}).get("label") != label:
                raise ValueError("Imported initial conflicts with saved initial: " + sid)
            continue
        # Reference/correct/annotation fields are deliberately not imported or sent to review.
        value = {"label":label, "protocol":"imported_frozen_prediction"}
        if isinstance(item.get("probabilities"), dict):
            value["probabilities"] = item["probabilities"]
        prepared.append((sid, value))
    for sid, value in prepared:
        checkpoint.record(sid, "initial", status="succeeded", attempt=0, output=value,
                          usage={}, imported_file_sha256=source_hash)


def execute_stage(checkpoint, sid, name, call, attempts, stopped, start_attempt=None):
    old = checkpoint.get(sid, name)
    if old and old["status"] in ("succeeded", "terminal_failed", "skipped"):
        return old
    attempt = old.get("attempt", 0) if old else 0
    if attempt >= attempts:
        return checkpoint.record(sid, name, status="terminal_failed", attempt=attempt,
                                 error="Interrupted or retry limit reached; stage is not rerun automatically")
    while attempt < attempts:
        if stopped():
            return None
        attempt += 1
        # Record the attempt before sending/inference, so restarts cannot reset request budgets.
        if start_attempt is not None:
            if not start_attempt(attempt):
                return None
        else:
            checkpoint.record(sid, name, status="running", attempt=attempt)
        started = time.monotonic()
        usage = {}
        try:
            output, usage = call()
            if name == "initial":
                if not isinstance(output, dict) or output.get("label") not in LABELS:
                    raise ValueError("Initial classification invalid; no fabricated fallback label")
            else:
                raw = output
                output = parse_json(raw)
                if usage.get("finish_reason") != "stop" or output is None:
                    return checkpoint.record(sid, name, status="terminal_failed", attempt=attempt,
                        raw_output=raw, output=output, usage=usage, elapsed_seconds=time.monotonic()-started,
                        error="Review truncated/timed out/non-JSON; valid initial is retained")
            return checkpoint.record(sid, name, status="succeeded", attempt=attempt, output=output,
                usage=usage, elapsed_seconds=time.monotonic()-started,
                **({"raw_output":raw} if name != "initial" else {}))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            retryable = bool(getattr(exc, "retryable", False)) and attempt < attempts
            event = checkpoint.record(sid, name, status="retryable" if retryable else "terminal_failed",
                attempt=attempt, error=f"{type(exc).__name__}: {exc}", usage=usage,
                usage_unknown=not bool(usage),
                service_unavailable=bool(getattr(exc, "service_unavailable", False)),
                elapsed_seconds=time.monotonic()-started)
            if not retryable:
                return event
            time.sleep(1)
    return checkpoint.get(sid, name)


def perform_review(row, schema, reviewer, checkpoint, config, stopped, claim=None):
    """One durable review call, following confidence routing; failures keep initial."""
    sid = str(row["sample_id"])
    if not select_review(checkpoint, sid, config):
        return checkpoint.get(sid, "review")
    def evaluate_answer():
        text, usage = reviewer.review(review_messages(row, schema))
        raw = parse_json(text)
        if isinstance(raw, dict) and usage.get("finish_reason") == "stop":
            text = json.dumps(bind_single_review(raw, row), ensure_ascii=False)
        return text, usage

    if stopped():
        return None
    if hasattr(reviewer, "ensure_available"):
        try:
            reviewer.ensure_available()
        except BackendError:
            return None
    return execute_stage(checkpoint, sid, "review", evaluate_answer,
        config["review_attempts"], stopped, start_attempt=claim("review") if claim else None)


def process_rows(rows, schema, backend_factory, checkpoint, config, stopped, limit=None, imported_only=False):
    backend, processed = None, 0
    for row in rows:
        if stopped():
            break
        sid = str(row["sample_id"])
        first, second = checkpoint.get(sid, "initial"), checkpoint.get(sid, "review")
        for name, event, budget in (("initial",first,config["initial_attempts"]),
                                    ("review",second,config["review_attempts"])):
            if event and event["status"] == "running" and event.get("attempt",0) >= budget:
                checkpoint.record(sid,name,status="terminal_failed",attempt=event["attempt"],
                    interrupted_usage_unknown=True,error="Previously interrupted attempt exhausted its stage budget")
        first, second = checkpoint.get(sid, "initial"), checkpoint.get(sid, "review")
        done = (first and first["status"] == "terminal_failed") or (
            first and first["status"] == "succeeded" and second and second["status"] in ("succeeded", "terminal_failed", "skipped"))
        if done:
            continue
        if limit is not None and processed >= limit:
            break
        if imported_only and not (first and first["status"] == "succeeded"):
            continue
        if first and first["status"] == "succeeded" and not select_review(checkpoint, sid, config):
            continue
        if backend is None:
            backend = backend_factory()
        first = execute_stage(checkpoint, sid, "initial", lambda:backend.initial(row),
                              config["initial_attempts"], stopped)
        if first and first["status"] == "succeeded" and not stopped():
            second=perform_review(row, schema, backend, checkpoint, config, stopped)
        if first and first["status"]=="terminal_failed" or second and second["status"]=="terminal_failed":
            backend.clear_after_error()
        processed += 1
        print(f"Processed pending sample {processed}: {sid}", flush=True)
    return processed


def process_api_reviews(rows, schema, reviewer, checkpoint, config, stopped):
    """Bound pending futures and requests; paused/unscheduled samples consume no attempt."""
    workers = config.get("review_api_workers", 5)
    halted = threading.Event()
    dispatch_lock = threading.Lock()

    def review_one(row):
        sid = str(row["sample_id"])

        def claim(stage):
            def begin(attempt):
                # Both stages claim only their actual request, never queued work.
                with dispatch_lock:
                    if halted.is_set() or stopped():
                        return False
                    if hasattr(reviewer, "ensure_available"):
                        try:
                            reviewer.ensure_available()
                        except BackendError:
                            halted.set()
                            return False
                    checkpoint.record(sid, stage, status="running", attempt=attempt)
                    return True
            return begin

        event = perform_review(row, schema, reviewer, checkpoint, config,
                               lambda: stopped() or halted.is_set(), claim)
        if event and event["status"] == "terminal_failed":
            reviewer.clear_after_error()
        if (event and event.get("service_unavailable")) or getattr(reviewer, "healthy", True) is False:
            halted.set()
        return sid, event

    pending = {}
    iterator = iter(rows)
    exhausted = False
    with futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="evidence_review") as pool:
        try:
            while pending or not exhausted:
                if stopped():
                    halted.set()
                while not exhausted and not halted.is_set() and len(pending) < workers:
                    try:
                        row = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    # No dataset-sized queue: at most workers active/queued tasks exist.
                    sid = str(row["sample_id"])
                    if not select_review(checkpoint, sid, config):
                        continue
                    pending[pool.submit(review_one, row)] = sid
                if not pending:
                    break
                if halted.is_set():
                    for task in pending:
                        task.cancel()
                done, _ = futures.wait(pending, timeout=0.2, return_when=futures.FIRST_COMPLETED)
                for task in done:
                    pending.pop(task)
                    if not task.cancelled():
                        sid, event = task.result()
                        if event:
                            print("Review processed: " + sid, flush=True)
        except BaseException:
            halted.set()
            for task in pending:
                task.cancel()
            # The pool drains started calls before callers publish frozen outputs.
            raise
    if halted.is_set() and not stopped():
        print("Reviewer unavailable; stopped new reviews and saved started calls.", flush=True)


def process_phased(rows,schema,initial_factory,review_factory,checkpoint,config,stopped,limit=None,imported_only=False):
    """Complete/cache initial tasks, close that model, then start the selected reviewer."""
    selected=[]
    for row in rows:
        sid=str(row["sample_id"])
        for name,budget in (("initial",config["initial_attempts"]), ("review",config["review_attempts"])):
            old=checkpoint.get(sid,name)
            if old and old["status"]=="running" and old.get("attempt",0)>=budget:
                checkpoint.record(sid,name,status="terminal_failed",attempt=old["attempt"],
                    interrupted_usage_unknown=True,error="Previously interrupted attempt exhausted its budget")
        first,second=checkpoint.get(sid,"initial"),checkpoint.get(sid,"review")
        if first and first["status"]=="terminal_failed": continue
        if first and first["status"]=="succeeded" and second and second["status"] in ("succeeded","terminal_failed","skipped"):
            continue
        if imported_only and not (first and first["status"]=="succeeded"): continue
        if first and first["status"] == "succeeded" and not select_review(checkpoint, sid, config): continue
        selected.append(row)
    if limit is not None: selected=selected[:limit]
    classifier=None
    try:
        for row in selected:
            if stopped(): break
            sid=str(row["sample_id"])
            old=checkpoint.get(sid,"initial")
            if old and old["status"]=="succeeded": continue
            if classifier is None: classifier=initial_factory()
            result=execute_stage(checkpoint,sid,"initial",lambda:classifier.initial(row),
                config["initial_attempts"],stopped)
            if result and result["status"]=="terminal_failed": classifier.clear_after_error()
            print("Initial saved: "+sid,flush=True)
    finally:
        if classifier is not None: classifier.close()
        classifier=None
    reviewer=None
    try:
        if config.get("review_backend", "api") == "api" and config.get("review_api_workers", 5) > 1:
            pending_rows = []
            for row in selected:
                sid = str(row["sample_id"])
                first, second = checkpoint.get(sid, "initial"), checkpoint.get(sid, "review")
                if (first and first["status"] == "succeeded"
                        and not (second and second["status"] in ("succeeded", "terminal_failed", "skipped"))
                        and select_review(checkpoint, sid, config)):
                    pending_rows.append(row)
            if pending_rows and not stopped():
                reviewer = review_factory()
                process_api_reviews(pending_rows, schema, reviewer, checkpoint, config, stopped)
            return len(selected)
        for row in selected:
            if stopped(): break
            sid=str(row["sample_id"])
            first=checkpoint.get(sid,"initial")
            if not first or first["status"]!="succeeded": continue
            if not select_review(checkpoint, sid, config): continue
            if reviewer is None: reviewer=review_factory()
            # A worker already dead between requests has not attempted this sample.
            # Leave it pending so restarting can review it without redoing initial tasks.
            if hasattr(reviewer,"ensure_available"):
                try:
                    reviewer.ensure_available()
                except BackendError as exc:
                    print(str(exc)+"; stopping before a new review attempt.",flush=True)
                    break
            result=perform_review(row, schema, reviewer, checkpoint, config, stopped)
            if result and result["status"]=="terminal_failed": reviewer.clear_after_error()
            print("Review processed: "+sid,flush=True)
            if getattr(reviewer,"healthy",True) is False:
                print("Reviewer unavailable; stopping new reviews. Initial predictions are preserved.",flush=True)
                break
    finally:
        if reviewer is not None: reviewer.close()
    return len(selected)


def freeze_outputs(rows, checkpoint, mode, policy=None, config=None):
    outputs = []
    for row in rows:
        sid = str(row["sample_id"])
        first, second = checkpoint.get(sid, "initial"), checkpoint.get(sid, "review")
        initial = first.get("output") if first and first["status"] == "succeeded" else None
        review = second.get("output") if second and second["status"] == "succeeded" else None
        decision = decide(initial, review, row, mode, policy)
        route = route_initial(initial, config)
        if initial is not None and not route["review_required"]:
            decision.update(final_label=initial["label"], experimental_label=initial["label"],
                            accepted=False, decision_reason="confidence_sufficient_keep_initial",
                            structure_errors=[], review_structure_valid=False)
        outputs.append({"sample_id":sid, **decision,
            **route,
            "initial_status":first["status"] if first else "pending",
            "review_status":("skipped" if initial is not None and not route["review_required"]
                             else second["status"] if second else "pending"),
            "initial":initial, "review":review,
            "review_error":second.get("error") if second else None})
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Explicit current configuration file")
    parser.add_argument("--base-model", help="Server base model directory (or pinned HF model ID)")
    parser.add_argument("--adapter-path", help="Server best_adapter directory, or training output containing it")
    parser.add_argument("--quantization", choices=("none","4bit"))
    parser.add_argument("--review-backend",choices=("api","vllm"))
    parser.add_argument("--review-api-base")
    parser.add_argument("--review-api-model")
    parser.add_argument("--review-workers", type=int, help="Maximum concurrent API reviews; current configuration uses 5; offline vLLM stays serial")
    parser.add_argument("--review-local-model")
    parser.add_argument("--vllm-python",help="Separate offline-vLLM Python executable")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--no-reference",action="store_true",help="Do not open a separate reference file")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("observe", "experimental", "calibrated"))
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--limit", type=int, help="Cap pending samples this invocation; successful stages are reused")
    parser.add_argument("--dry-run", action="store_true", help="Validate files/prompts; do not load/download/call a model")
    parser.add_argument("--redecide-only", action="store_true", help="Reapply gate to checkpoints; no model calls")
    parser.add_argument("--initial-predictions", type=Path, help="Import a frozen initial prediction JSONL by sample_id")
    parser.add_argument("--imported-only", action="store_true", help="Review only imported/saved initial predictions")
    args = parser.parse_args(argv)
    for key in ("base_model", "review_local_model"):
        value = getattr(args, key)
        if value and Path(value).is_dir():
            setattr(args, key, str(Path(value).resolve()))
    config = load_config(args.config, {"base_model":args.base_model,"adapter_path":args.adapter_path,
        "quantization":args.quantization,"review_backend":args.review_backend,
        "review_api_base":args.review_api_base,"review_api_model":args.review_api_model,
        "review_api_workers":args.review_workers,
        "review_local_model":args.review_local_model,"review_vllm_python":args.vllm_python})
    for key in ("input", "references", "output_dir", "mode"):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value.resolve()) if isinstance(value, Path) else value
    if args.no_reference:
        config["references"]=None
    if config.get("references") and not Path(config["references"]).is_file():
        raise ValueError("Reference file is missing; supply --references or --no-reference")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    rows = load_samples(config["input"])
    engine = engine_identity(config)
    hashes = {str(r["sample_id"]):digest(blind(r)) for r in rows}
    identity = digest({"engine":engine, "inputs":hashes})
    schema = read_json(Path(config["adapter_path"])/"classification_schema.json")["question_schema"]
    policy = read_json(args.policy) if args.policy else None
    gate_id=gate_identity()
    if config["mode"] == "calibrated":
        validate_policy(policy, engine, gate_id)
    if args.dry_run:
        print(json.dumps({"samples":len(rows), "engine_identity":engine,"gate_identity":gate_id, "adapter_path":config["adapter_path"],
            "base_model":config["base_model"], "revision":config["base_revision"], "quantization":config["quantization"],
            "review_uses_adapter":config["review_use_adapter"], "mode":config["mode"],
            "review_backend":config["review_backend"],
            "review_api_workers":config["review_api_workers"] if config["review_backend"] == "api" else 1,
            "review_prompt_fields":list(blind(rows[0])), "models_loaded":False}, ensure_ascii=False, indent=2))
        return 0
    out = Path(config["output_dir"])
    if args.initial_predictions and not args.initial_predictions.is_file():
        raise ValueError("Initial prediction file not found: " + str(args.initial_predictions))
    if (args.imported_only and not args.redecide_only and not args.initial_predictions
            and not (out / "checkpoint.jsonl").is_file()):
        parser.error("A fresh --imported-only run requires --initial-predictions <prediction.jsonl>")
    pause = out/".pause"
    interrupted = False
    began = time.monotonic()
    with output_lock(out):
        checkpoint = Checkpoint(out/"checkpoint.jsonl", identity, hashes)
        manifest = {"engine_identity":engine, "run_identity":identity, "config":config,
                    "input_hashes":hashes, "gate_policy":policy,"gate_identity":gate_id,
                    "classification_note":("4-bit differs from training BF16; remeasure baseline ACC"
                        if config["quantization"]=="4bit" else "Pinned training label protocol; remeasure baseline under this runtime")}
        write_json(out/"run_manifest.json", manifest)
        if args.initial_predictions:
            import_initials(args.initial_predictions, rows, checkpoint)
        try:
            if not args.redecide_only:
                factory = lambda: LocalBackend(config) if config["backend"] == "transformers" else APIBackend(config)
                review_factory=(lambda:ReviewVLLMBackend(config,out/"vllm_worker.log")) if config["review_backend"]=="vllm" else (lambda:ReviewAPIBackend(config))
                process_phased(rows,schema,factory,review_factory,checkpoint,config,pause.exists,args.limit,args.imported_only)
        except KeyboardInterrupt:
            interrupted = True
            print("Interrupted. Saved stages remain recoverable; resume with the same command.", flush=True)
        finally:
            # Freeze predictions first, then open scoring-only reference labels.
            outputs = freeze_outputs(rows, checkpoint, config["mode"], policy, config)
            write_jsonl(out/"predictions.jsonl", outputs)
            write_jsonl(out/"initial_predictions.jsonl",[
                {"sample_id":o["sample_id"],"predicted_level":o["initial_label"],
                 "probabilities":o["initial"].get("probabilities"),"input_sha256":hashes[o["sample_id"]]}
                for o in outputs if o["initial_label"] in LABELS])
            summary, trial, final = report(rows, outputs, config.get("references"))
            summary.update(engine_identity=engine,gate_identity=gate_id, mode=config["mode"],
                invocation_elapsed_seconds=time.monotonic()-began,
                recorded_inference_seconds=sum(e.get("elapsed_seconds",0) for e in checkpoint.events),
                logical_attempts=sum(e["status"]=="running" for e in checkpoint.events),
                unknown_interrupted_attempts=sum(e["status"]=="running" or e.get("interrupted_usage_unknown",False)
                    for e in checkpoint.latest.values()),
                unknown_usage_attempts=sum(e.get("usage_unknown",False) for e in checkpoint.events),
                usage={key:sum(e.get("usage",{}).get(key,0) or 0 for e in checkpoint.events)
                       for key in ("prompt_tokens", "completion_tokens", "initial_forward_passes")})
            write_json(out/"metrics.json", summary)
            write_jsonl(out/"experimental_changes.jsonl", trial)
            write_jsonl(out/"final_changes.jsonl", final)
            write_json(out/"run_state.json", {"interrupted":interrupted, "paused":pause.exists(),
                "terminal_initial_failures":sum(o["initial_status"]=="terminal_failed" for o in outputs),
                "pending_samples":sum(o["initial_status"] in ("pending","running","retryable") or
                    (o["initial_status"]=="succeeded" and o["review_status"] in ("pending","running","retryable")) for o in outputs)})
            for key in ("initial_label", "experimental_label", "final_label"):
                print(key, summary["scores"][key], flush=True)
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
