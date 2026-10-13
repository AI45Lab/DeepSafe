"""DeepSafe adapter for the shared Qwen checkpoint, review and acceptance implementation."""
from pathlib import Path
from contextlib import contextmanager, ExitStack
import os
import threading

from qwen_evidence_review.common import (Checkpoint, LABELS, blind, digest, file_hash,
                                       output_lock, read_json, write_json, write_jsonl)
from qwen_evidence_review.gate import decide, gate_identity, validate_policy
from qwen_evidence_review.run import (engine_identity, freeze_outputs, import_initials,
                                     load_config, process_phased)
from uni_eval.evaluators.base import BaseEvaluator
from uni_eval.registry import EVALUATORS

ROOT = Path(__file__).resolve().parents[2]
METHOD = "qwen_evidence_review"


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def bridge_identity(config):
    # Include the integration contract, without making metric/report edits invalidate inference.
    files = ("uni_eval/evaluators/qwen_evidence.py", "uni_eval/models/qwen_evidence.py",
             "uni_eval/datasets/qwen_evidence.py")
    return digest({"qwen_engine": engine_identity(config),
                   "bridge": {name: file_hash(ROOT / name) for name in files}})


@EVALUATORS.register_module()
class QwenEvidenceEvaluator(BaseEvaluator):
    def __init__(self, config_path=None, runtime=None,
                 state_dir="results/qwen_evidence/state", limit=0, policy_path="",
                 initial_predictions="", imported_only=False, dry_run=False,
                 use_precomputed_predictions=False, **kwargs):
        if not config_path:
            raise ValueError("Specify evaluator.config_path explicitly; no default framework configuration is used")
        self.config_path = project_path(config_path)
        self.runtime = dict(runtime or {})
        self.state_dir = project_path(state_dir)
        if isinstance(limit, bool) or str(limit).strip() != str(int(limit)) or int(limit) < 0:
            raise ValueError("evaluator.limit must be an integer >=0 (0 means all pending samples)")
        self.limit = int(limit) or None
        self.policy_path = project_path(policy_path) if policy_path else None
        self.initial_predictions = project_path(initial_predictions) if initial_predictions else None
        self.imported_only = bool(imported_only)
        self.dry_run = bool(dry_run)
        self.use_precomputed_predictions = bool(use_precomputed_predictions)
        if (not self.use_precomputed_predictions and self.initial_predictions
                and not self.initial_predictions.is_file()):
            raise ValueError("Initial prediction file not found: " + str(self.initial_predictions)
                             + "; specify --evaluator.initial_predictions <prediction.jsonl>")
        if (not self.use_precomputed_predictions and self.imported_only and not self.initial_predictions
                and not (self.state_dir / "checkpoint.jsonl").is_file()):
            raise ValueError("A fresh imported-only run requires an explicit initial prediction file; "
                             "specify --evaluator.initial_predictions <prediction.jsonl>")
        self._publication_lock_owner = None

    @contextmanager
    def publication_lock(self, output_dir=None, predictions_path=None):
        """Keep state, final snapshots and reports in the same publication transaction."""
        owner = threading.get_ident()
        if self._publication_lock_owner == owner:
            yield
            return
        directories = [self.state_dir]
        if output_dir is not None:
            directories.append(Path(output_dir))
        if predictions_path is not None:
            directories.append(Path(predictions_path).parent)
        unique = {os.path.normcase(str(path.resolve())): path.resolve() for path in directories}
        with ExitStack() as stack:
            for key in sorted(unique):
                stack.enter_context(output_lock(unique[key]))
            self._publication_lock_owner = owner
            try:
                yield
            finally:
                self._publication_lock_owner = None

    def prepare(self, dataset):
        # Runtime has one owner (the evaluator); identical settings are used for gen and offline eval.
        config = load_config(self.config_path, self.runtime)
        if config["backend"] != "transformers":
            raise ValueError("DeepSafe Qwen initial classification requires the trained Transformers LoRA logits protocol")
        if config["review_use_adapter"]:
            raise ValueError("The class-token LoRA must be disabled for independent evidence review")
        rows, seen = [], set()
        for item in dataset:
            inputs = item["qwen_input"]
            sid = str(item["id"])
            if str(inputs["sample_id"]) != sid or sid in seen:
                raise ValueError("Duplicate or mismatched Qwen sample ID: " + sid)
            seen.add(sid)
            rows.append({"sample_id": sid, **blind(inputs)})
        if not rows:
            raise ValueError("Empty Qwen evaluation dataset")
        engine = bridge_identity(config)
        hashes = {r["sample_id"]: digest(blind(r)) for r in rows}
        identity = digest({"engine": engine, "inputs": hashes})
        gate = gate_identity()
        policy = read_json(self.policy_path) if self.policy_path else None
        if config["mode"] == "calibrated":
            validate_policy(policy, engine, gate)
        info = {"method": METHOD, "engine_identity": engine, "run_identity": identity,
                "gate_identity": gate, "mode": config["mode"],
                "policy_sha256": digest(policy) if policy else None,
                "review_backend": config["review_backend"], "quantization": config["quantization"]}
        schema = read_json(Path(config["adapter_path"]) / "classification_schema.json")["question_schema"]
        return config, rows, hashes, identity, policy, info, schema

    @staticmethod
    def pack(dataset, outputs, info):
        return [{**item, "prediction": out["final_label"], "attribution": out,
                 "run_info": dict(info)} for item, out in zip(dataset, outputs)]

    def generate_predictions(self, model, dataset):
        dataset = list(dataset)
        config, rows, hashes, identity, policy, info, schema = self.prepare(dataset)
        if self.dry_run:
            workers = config["review_api_workers"] if config["review_backend"] == "api" else 1
            print(f"Qwen dry-run: {len(rows)} samples; review_workers={workers}; no model loading or API calls; engine={info['engine_identity']}")
            # A validation run cannot overwrite a genuine inference snapshot.
            raise SystemExit(0)
        if not callable(getattr(model, "initial_backend", None)):
            raise TypeError("QwenEvidenceEvaluator requires QwenEvidenceModel")
        with self.publication_lock():
            checkpoint = Checkpoint(self.state_dir / "checkpoint.jsonl", identity, hashes)
            write_json(self.state_dir / "run_manifest.json",
                       {"run_info": info, "input_hashes": hashes, "gate_policy": policy,
                        "config": config})
            if self.initial_predictions:
                import_initials(self.initial_predictions, rows, checkpoint)
            try:
                # Both review options use another model, after classifier stages are saved and closed.
                process_phased(rows, schema, lambda: model.initial_backend(config),
                               lambda: model.review_backend(config, self.state_dir),
                               checkpoint, config, (self.state_dir / ".pause").exists,
                               self.limit, self.imported_only)
            finally:
                # Even interruption/startup failure leaves saved stages and frozen initials recoverable.
                outputs = freeze_outputs(rows, checkpoint, config["mode"], policy, config)
                write_jsonl(self.state_dir / "predictions.jsonl", outputs)
                write_jsonl(self.state_dir / "initial_predictions.jsonl",
                            [{"sample_id": o["sample_id"], "predicted_level": o["initial_label"],
                              "probabilities": o["initial"].get("probabilities"),
                              "input_sha256": hashes[o["sample_id"]]}
                             for o in outputs if o["initial_label"] in LABELS])
        return self.pack(dataset, outputs, info)

    def evaluate(self, model, dataset, **kwargs):
        dataset = list(dataset)
        if not self.use_precomputed_predictions:
            return self.generate_predictions(model, dataset)
        config, rows, _, _, policy, info, _ = self.prepare(dataset)
        outputs = []
        for item, row in zip(dataset, rows):
            old_info = item.get("run_info", {})
            if any(old_info.get(key) != info[key] for key in ("method", "engine_identity", "run_identity")):
                raise ValueError("Qwen prediction input/runtime/code mismatch; use matching settings or regenerate")
            out = item.get("attribution")
            if not isinstance(out, dict) or str(out.get("sample_id")) != row["sample_id"]:
                raise ValueError("Missing/mismatched frozen Qwen evidence for " + row["sample_id"])
            initial = out.get("initial") if out.get("initial_status") == "succeeded" else None
            review = out.get("review") if out.get("review_status") == "succeeded" else None
            # Reapply current gates; old candidate/final labels are never trusted as model evidence.
            outputs.append({**out, **decide(initial, review, row, config["mode"], policy)})
        return self.pack(dataset, outputs, info)
