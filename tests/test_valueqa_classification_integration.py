"""Integration contracts exercised through real registry/runner/CLI, without CUDA weights."""
import contextlib
import copy
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import yaml
import uni_eval
uni_eval.load_all_modules("value")
from uni_eval.registry import DATASETS, EVALUATORS, METRICS, MODELS, RUNNERS
from uni_eval.models.valueqa_classification import ValueQAClassificationModel
from valueqa_classification_review.common import output_lock, read_json, read_jsonl, write_json, write_jsonl
from valueqa_classification_review.boundary_test_support import certify_review, stage_response, visible_sample

ROOT = Path(__file__).resolve().parents[1]
ROW = {"sample_id": "one", "original_context": "台风导致会议延期，最终改到周五。",
       "target_comment": "会议改到周五。", "question": "为什么延期，改到哪一天？", "answer": "因为台风。"}


def evidence():
    material = [{"field": "original_context", "text": ROW["original_context"]}]
    answer = [{"field": "answer", "text": ROW["answer"]}]
    value = {"label": "medium", "reason": "原因正确，遗漏日期", "review_complete": True,
            "requirements_complete": True, "residual_core_check_complete": True,
            "core": {"status": "correct", "reason": "原因正确", "answer_evidence": answer,
                     "material_evidence": material},
            "requirements": [
                {"id": "R1", "point": "原因", "status": "met", "reason": "原因已表达",
                 "question_evidence": [{"field": "question", "text": "为什么延期"}],
                 "material_evidence": material, "answer_evidence": answer},
                {"id": "R2", "point": "日期", "status": "missing", "reason": "遗漏必要日期",
                 "question_evidence": [{"field": "question", "text": "改到哪一天"}],
                 "material_evidence": material, "answer_evidence": []}],
            "issues": [{"requirement_id": "R2", "kind": "local", "status": "confirmed",
                        "reason": "遗漏日期", "missing_point": "周五", "answer_evidence": [],
                        "material_evidence": material}]}
    return certify_review(value, ROW)


class FakeBackend:
    def __init__(self, calls, role="shared", fail=False):
        self.calls, self.role, self.fail = calls, role, fail
        calls.append(("open", role))

    def initial(self, row):
        self.calls.append(("initial", copy.deepcopy(row)))
        return {"label": "high"}, {"initial_forward_passes": 1}

    def review(self, messages):
        self.calls.append(("review", copy.deepcopy(messages)))
        if self.fail:
            raise RuntimeError("mock review failure")
        return json.dumps(stage_response(messages, evidence()), ensure_ascii=False), {"finish_reason": "stop"}

    def clear_after_error(self):
        self.calls.append(("clear", self.role))

    def close(self):
        self.calls.append(("close", self.role))


class QwenIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        adapter = self.directory / "adapter"
        adapter.mkdir()
        config = read_json(ROOT / "valueqa_classification_review/config.current.server.json")
        config.update(adapter_path=str(adapter), review_backend="api", local_files_only=True,
                      review_api_base="http://127.0.0.1:1/v1", review_api_model="mock-review", review_api_workers=1,
                      review_api_warmup_models=False, quantization="none")
        write_json(adapter / "adapter_config.json", {"base_model_name_or_path": config["base_model"]})
        write_json(adapter / "training_config.json", {"model_revision": config["base_revision"]})
        write_json(adapter / "classification_schema.json", {"question_schema": {"instructions": "核对材料",
            "criteria": {"high": "充分", "medium": "局部不足", "low": "中心误读"}}})
        for name in ("adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            (adapter / name).write_text("mock-only", encoding="utf-8")
        self.config_path = self.directory / "runtime.json"
        write_json(self.config_path, config)
        self.samples = self.directory / "samples.jsonl"
        self.references = self.directory / "references.jsonl"
        write_jsonl(self.samples, [{**ROW, "label": "medium"}, {**ROW, "sample_id": "two", "label": "medium"}])
        write_jsonl(self.references, [{"sample_id": sid, "label": "medium", "annotations": []} for sid in ("one", "two")])
        self.cfg = {"module_profile": "value", "model": {"type": "ValueQAClassificationModel"},
                    "dataset": {"type": "ValueQAClassificationDataset", "path": str(self.samples), "references_path": str(self.references)},
                    "evaluator": {"type": "ValueQAClassificationEvaluator", "config_path": str(self.config_path),
                                  "state_dir": str(self.directory / "state"), "runtime": {"mode": "observe"}},
                    "metrics": [{"type": "ValueQAClassificationMetric"}], "summarizer": {"type": "StandardSummarizer"},
                    "runner": {"type": "LocalRunner", "stage": "all", "prediction_policy": "strict",
                               "output_dir": str(self.directory / "results"), "use_evaluator_gen": True,
                               "badcase_unknown_samples": 0}}

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self):
        return DATASETS.build(self.cfg["dataset"])

    def evaluator(self, **kwargs):
        return EVALUATORS.build({**self.cfg["evaluator"], **kwargs})

    def infer(self, **kwargs):
        calls = []
        model = ValueQAClassificationModel()
        fail = kwargs.pop("fail", False)
        with patch.object(model, "initial_backend", side_effect=lambda cfg: FakeBackend(calls, "initial")), \
             patch.object(model, "review_backend", side_effect=lambda cfg, path: FakeBackend(calls, "review", fail=fail)):
            results = self.evaluator(**kwargs).evaluate(model, self.dataset())
        return results, calls

    def run_runner(self, cfg):
        with contextlib.redirect_stdout(io.StringIO()):
            RUNNERS.build(cfg["runner"]).run(cfg)

    def test_registry_construction_is_lazy(self):
        with patch("valueqa_classification_review.backend.LocalBackend", side_effect=AssertionError("eager load")):
            MODELS.build(self.cfg["model"])
            self.evaluator()
            METRICS.build({"type": "ValueQAClassificationMetric"})

    def test_environment_adapter_overrides_defaults_and_explicit_path_overrides_environment(self):
        adapter=self.directory/"adapter"
        explicit_adapter=self.directory/"explicit_adapter"
        shutil.copytree(adapter,explicit_adapter)
        for name in ("api", "api_linux", "vllm", "cached_1533"):
            cfg=yaml.safe_load((ROOT/f"configs/eval_tasks/valueqa_classification_{name}.yaml").read_text(encoding="utf-8"))
            runtime=dict(cfg["evaluator"]["runtime"])
            runtime.update(review_api_base="http://127.0.0.1:1/v1", review_api_model="mock-review",
                           review_local_model="mock-local-review")
            settings={"config_path":cfg["evaluator"]["config_path"],"runtime":runtime}
            with patch.dict(os.environ,{"QWEN_LORA_PATH":str(adapter),"QWEN_BASE_PATH":"","VLLM_PYTHON":""}), \
                 patch("valueqa_classification_review.run.sys.platform", "linux"):
                config=self.evaluator(**settings).prepare(list(self.dataset()))[0]
                self.assertEqual(config["adapter_path"],str(adapter))
                runtime["adapter_path"]=str(explicit_adapter)
                config=self.evaluator(**settings).prepare(list(self.dataset()))[0]
                self.assertEqual(config["adapter_path"],str(explicit_adapter))

    def test_evaluation_input_import_does_not_load_model_or_record_initials(self):
        model=ValueQAClassificationModel()
        with patch.object(model,"initial_backend",side_effect=AssertionError("must not load a model")):
            with self.assertRaisesRegex(ValueError,"Explicit prediction field required"):
                self.evaluator(initial_predictions=str(self.samples),imported_only=True).evaluate(model,self.dataset())
        checkpoint=self.directory/"state/checkpoint.jsonl"
        self.assertFalse(checkpoint.exists())

    def test_no_gold_or_initial_leak_and_observe_gate(self):
        results, calls = self.infer()
        self.assertEqual([r["prediction"] for r in results], ["high", "high"])
        self.assertEqual(results[0]["attribution"]["experimental_label"], "medium")
        for name, value in calls:
            if name == "initial":
                self.assertEqual(set(value), {"sample_id", "question", "answer", "target_comment", "original_context"})
            if name == "review":
                text = json.dumps(value, ensure_ascii=False)
                self.assertNotIn("ground_truth", text)
                self.assertNotIn("initial_label", text)
                self.assertNotIn("sample_id", text)
        self.assertEqual(calls[-1], ("close", "review"))

    def test_limit_then_resume_then_no_model_calls(self):
        results, calls = self.infer(limit=1)
        self.assertEqual(results[1]["attribution"]["initial_status"], "pending")
        scores = METRICS.build({"type": "ValueQAClassificationMetric"}).compute(results)
        self.assertEqual(scores["initial_n"], 1)
        self.assertEqual(scores["candidate_n"], 1)
        results, calls = self.infer()
        self.assertEqual([v["sample_id"] for name, v in calls if name == "initial"], ["two"])
        _, calls = self.infer()
        self.assertEqual(calls, [])

    def test_runner_parallel_ten_preserves_order_and_reuses_completed_checkpoints(self):
        rows = [{**ROW, "sample_id": str(i), "answer": ROW["answer"] + f" 样本{i}", "label": "medium"}
                for i in range(30)]
        write_jsonl(self.samples, rows)
        write_jsonl(self.references, [{"sample_id": str(i), "label": "medium", "annotations": []}
                                     for i in range(30)])
        cfg = copy.deepcopy(self.cfg)
        cfg["evaluator"]["runtime"]["review_api_workers"] = 10
        lock, barrier = threading.Lock(), threading.Barrier(10)
        state = {"active": 0, "peak": 0, "calls": 0, "closed": False}
        class Reviewer:
            def review(self, messages):
                with lock:
                    index = state["calls"]
                    state["calls"] += 1
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                try:
                    if index < 10: barrier.wait(timeout=5)
                    visible = visible_sample(messages)
                    value = evidence()
                    value["reason"] = visible.get("answer", "requirements")
                    return json.dumps(stage_response(messages, value)), {"finish_reason": "stop"}
                finally:
                    with lock: state["active"] -= 1
            def clear_after_error(self): pass
            def close(self):
                if state["active"]: raise AssertionError("Reviewer closed before in-flight work finished")
                state["closed"] = True
        with patch.object(ValueQAClassificationModel, "initial_backend", side_effect=lambda cfg: FakeBackend([])), \
             patch.object(ValueQAClassificationModel, "review_backend", side_effect=lambda cfg, path: Reviewer()):
            self.run_runner(cfg)
        self.assertEqual((state["calls"], state["peak"], state["closed"]), (30, 10, True))
        outputs = read_jsonl(self.directory / "state/predictions.jsonl")
        self.assertEqual([r["sample_id"] for r in outputs], [str(i) for i in range(30)])
        self.assertEqual([r["review"]["reason"] for r in outputs], [r["answer"] for r in rows])
        cfg["evaluator"]["runtime"]["review_api_workers"] = 3
        with patch.object(ValueQAClassificationModel, "initial_backend", side_effect=AssertionError("recomputed initial")), \
             patch.object(ValueQAClassificationModel, "review_backend", side_effect=AssertionError("repeated review")):
            self.run_runner(cfg)

    def test_failure_falls_back_and_is_not_retried(self):
        results, _ = self.infer(fail=True)
        self.assertEqual(results[0]["prediction"], "high")
        self.assertEqual(results[0]["attribution"]["review_status"], "terminal_failed")
        _, calls = self.infer()
        self.assertEqual(calls, [])

    def test_phased_release_before_review(self):
        for selected in ("vllm", "api"):
            calls = []
            model = ValueQAClassificationModel()
            runtime = {"review_backend": selected, "review_api_model": "mock",
                       "review_local_model": "mock-local-review"}
            with patch("valueqa_classification_review.run.sys.platform", "linux"), \
                 patch.object(model, "initial_backend", side_effect=lambda cfg: FakeBackend(calls, "initial")), \
                 patch.object(model, "review_backend", side_effect=lambda cfg, path: FakeBackend(calls, "review")):
                self.evaluator(runtime=runtime, state_dir=str(self.directory / selected)).evaluate(model, self.dataset())
            self.assertLess(calls.index(("close", "initial")), calls.index(("open", "review")))
            self.assertEqual(calls[-1], ("close", "review"))

    def test_factory_routes_review_backend(self):
        model = ValueQAClassificationModel()
        for selection, name in (("api", "ReviewAPIBackend"), ("vllm", "ReviewVLLMBackend")):
            with patch("valueqa_classification_review.backend." + name) as factory:
                model.review_backend({"review_backend": selection}, self.directory)
                self.assertEqual(factory.call_count, 1)
        with patch("valueqa_classification_review.backend.LocalBackend", side_effect=AssertionError("same-model fallback")):
            with self.assertRaisesRegex(ValueError, "api or vllm"):
                model.review_backend({"review_backend": "transformers"}, self.directory)

    def test_native_windows_vllm_rejected_before_initial_loading(self):
        model = ValueQAClassificationModel()
        with patch("valueqa_classification_review.run.sys.platform", "win32"), \
             patch.object(model, "initial_backend", side_effect=AssertionError("initial model loaded")):
            with self.assertRaisesRegex(ValueError, "requires Linux"):
                self.evaluator(runtime={"review_backend": "vllm", "review_local_model": "different-reviewer"}).evaluate(
                    model, self.dataset())
        self.assertFalse((self.directory / "state/checkpoint.jsonl").exists())

    def test_delayed_root_publication_blocks_concurrent_run_and_preserves_latest_snapshot(self):
        from uni_eval.runners.local_runner import _write_predictions_jsonl
        for stage in ("all", "gen"):
            with self.subTest(stage=stage):
                cfg = copy.deepcopy(self.cfg)
                cfg["runner"].update(stage=stage, output_dir=str(self.directory / stage / "results"))
                cfg["evaluator"].update(state_dir=str(self.directory / stage / "state"), limit=1)
                at_publish, release = threading.Event(), threading.Event()
                errors = []
                def delayed_write(path, rows):
                    at_publish.set()
                    if not release.wait(timeout=10):
                        raise RuntimeError("test publication barrier timed out")
                    return _write_predictions_jsonl(path, rows)
                def first_run():
                    try:
                        RUNNERS.build(cfg["runner"]).run(cfg)
                    except BaseException as exc:
                        errors.append(exc)
                with patch.object(ValueQAClassificationModel, "initial_backend", side_effect=lambda cfg: FakeBackend([])), \
                     patch.object(ValueQAClassificationModel, "review_backend", side_effect=lambda cfg, path: FakeBackend([])), \
                     patch("uni_eval.runners.local_runner._write_predictions_jsonl", side_effect=delayed_write), \
                     patch("builtins.print"):
                    thread = threading.Thread(target=first_run)
                    thread.start()
                    try:
                        self.assertTrue(at_publish.wait(timeout=10))
                        later = copy.deepcopy(cfg); later["evaluator"]["limit"] = 0
                        with self.assertRaisesRegex(RuntimeError, "in use"):
                            RUNNERS.build(later["runner"]).run(later)
                        state = read_jsonl(Path(cfg["evaluator"]["state_dir"]) / "predictions.jsonl")
                        self.assertEqual(state[1]["initial_status"], "pending")
                    finally:
                        release.set(); thread.join(timeout=10)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(errors, [])
                    RUNNERS.build(later["runner"]).run(later)
                snapshot = read_jsonl(Path(cfg["runner"]["output_dir"]) / "predictions.jsonl")
                self.assertTrue(all(row["attribution"]["review_status"] == "succeeded" for row in snapshot))

    def test_publication_lock_covers_reports_and_custom_prediction_directory(self):
        from uni_eval.summarizers.standard import StandardSummarizer
        cfg = copy.deepcopy(self.cfg)
        predictions = self.directory / "custom/predictions.jsonl"
        cfg["runner"]["predictions_path"] = str(predictions)
        original = StandardSummarizer.summarize
        def summarize(instance, results, directory):
            for path in (self.directory / "state", self.directory / "results", predictions.parent):
                with self.assertRaisesRegex(RuntimeError, "in use"):
                    with output_lock(path): pass
            return original(instance, results, directory)
        with patch.object(ValueQAClassificationModel, "initial_backend", side_effect=lambda cfg: FakeBackend([])), \
             patch.object(ValueQAClassificationModel, "review_backend", side_effect=lambda cfg, path: FakeBackend([])), \
             patch.object(StandardSummarizer, "summarize", summarize):
            self.run_runner(cfg)
        for path in (self.directory / "state", self.directory / "results", predictions.parent):
            with output_lock(path): pass

    def test_shared_output_and_state_lock_is_deduplicated_and_released_after_failure(self):
        evaluator = self.evaluator()
        state = self.directory / "state"
        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            with evaluator.publication_lock(state, state / "predictions.jsonl"):
                with evaluator.publication_lock():
                    raise RuntimeError("intentional failure")
        with output_lock(state): pass
        cfg = copy.deepcopy(self.cfg)
        cfg["runner"]["output_dir"] = str(state)
        with patch.object(ValueQAClassificationModel, "initial_backend", side_effect=lambda cfg: FakeBackend([])), \
             patch.object(ValueQAClassificationModel, "review_backend", side_effect=lambda cfg, path: FakeBackend([])):
            self.run_runner(cfg)
        self.assertTrue((state / "result.json").exists())

    def test_runner_all_saves_snapshot_and_offline_redecides(self):
        with patch.object(ValueQAClassificationModel, "initial_backend", side_effect=lambda cfg: FakeBackend([])), \
             patch.object(ValueQAClassificationModel, "review_backend", side_effect=lambda cfg, path: FakeBackend([])):
            self.run_runner(self.cfg)
        snapshot = self.directory / "results/predictions.jsonl"
        self.assertTrue(snapshot.is_file())
        cfg = copy.deepcopy(self.cfg)
        cfg["runner"].update(stage="eval", output_dir=str(self.directory / "offline"), predictions_path=str(snapshot))
        cfg["evaluator"]["runtime"]["mode"] = "experimental"
        with patch("valueqa_classification_review.backend.LocalBackend", side_effect=AssertionError("offline load")):
            self.run_runner(cfg)
        result = read_json(self.directory / "offline/result.json")
        self.assertEqual(result["metrics"]["initial_acc"], 0)
        self.assertEqual(result["metrics"]["final_acc"], 1)
        self.assertEqual(result["metrics"]["final_changes_corrected_error"], 2)
        self.assertEqual(read_jsonl(snapshot)[0]["prediction"], "high")

    def test_offline_rejects_changed_engine(self):
        results, _ = self.infer()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.evaluator(runtime={"quantization": "4bit"}, use_precomputed_predictions=True).evaluate(None, results)

    def test_offline_input_change_is_rejected(self):
        results, _ = self.infer()
        results[0]["qwen_input"]["answer"] += "改动"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.evaluator(use_precomputed_predictions=True).evaluate(None, results)

    def test_calibrated_policy_identity_and_allowed_transition(self):
        results, _ = self.infer()
        info = results[0]["run_info"]
        policy = {"schema_version": 2, "engine_identity": info["engine_identity"], "gate_identity": info["gate_identity"],
                  "validated": True, "enabled_transitions": ["high->medium"],
                  "validation_protocol": "mock-independent-test", "confirmation_dataset_sha256": "mock-hash"}
        path = self.directory / "policy.json"
        write_json(path, policy)
        evaluator = self.evaluator(runtime={"mode": "calibrated"}, policy_path=str(path), use_precomputed_predictions=True)
        self.assertEqual(evaluator.evaluate(None, results)[0]["prediction"], "medium")
        policy["gate_identity"] = "stale"
        write_json(path, policy)
        with self.assertRaisesRegex(ValueError, "matching engine/gate"):
            evaluator.evaluate(None, results)

    def test_dry_run_does_not_overwrite_outputs(self):
        results, _ = self.infer()
        old = (self.directory / "state/predictions.jsonl").read_bytes()
        with self.assertRaises(SystemExit) as result:
            self.evaluator(dry_run=True).generate_predictions(ValueQAClassificationModel(), self.dataset())
        self.assertEqual(result.exception.code, 0)
        self.assertEqual(old, (self.directory / "state/predictions.jsonl").read_bytes())

    def test_public_cli_api_review_then_offline_eval(self):
        requests = []
        warmups = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                warmups.append(self.path)
                if self.path != "/v1/models":
                    self.send_error(404)
                    return
                payload = json.dumps({"object": "list", "data": [{"id": "mock-review"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                payload = json.dumps({"choices": [{"message": {"content": json.dumps(stage_response(requests[-1]["messages"], evidence()), ensure_ascii=False)},
                                                   "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 20}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        imported = self.directory / "initial.jsonl"
        write_jsonl(imported, [{"sample_id": sid, "predicted_level": "high"} for sid in ("one", "two")])
        cfg = copy.deepcopy(self.cfg)
        cfg["evaluator"].update(initial_predictions=str(imported), imported_only=True,
            runtime={"review_backend": "api", "review_api_model": "mock-review",
                     "review_api_warmup_models": True,
                     "review_api_base": f"http://127.0.0.1:{server.server_port}/v1"})
        path = self.directory / "cli.yaml"
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        try:
            first = subprocess.run([sys.executable, "tools/run.py", str(path)], cwd=ROOT, capture_output=True, timeout=30)
            self.assertEqual(first.returncode, 0, first.stderr.decode(errors="replace"))
            self.assertEqual(len(requests), 2)
            self.assertEqual(warmups, ["/v1/models"])
            self.assertEqual(requests[0]["model"], "mock-review")
            self.assertNotIn("predicted_level", json.dumps(requests[0]))
            second = subprocess.run([sys.executable, "tools/run.py", str(path), "--runner.stage", "eval",
                                     "--evaluator.runtime.mode", "experimental"], cwd=ROOT, capture_output=True, timeout=30)
            self.assertEqual(second.returncode, 0, second.stderr.decode(errors="replace"))
            self.assertEqual(len(requests), 2)
            self.assertEqual(warmups, ["/v1/models"])
            self.assertEqual(read_json(self.directory / "results/result.json")["metrics"]["final_acc"], 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_frozen_dataset_loads_and_configs_register(self):
        for name in ("vllm", "api"):
            cfg = yaml.safe_load((ROOT / f"configs/eval_tasks/valueqa_classification_{name}.yaml").read_text(encoding="utf-8"))
            self.assertEqual(cfg["evaluator"]["runtime"]["review_backend"], name)
            self.assertEqual(cfg["evaluator"]["runtime"]["mode"], "experimental")
            MODELS.build(cfg["model"])
            self.assertEqual(len(DATASETS.build(cfg["dataset"])), 1533)

    def test_delivery_fields_preserved_and_labels_excluded_from_model_input(self):
        samples = read_jsonl(ROOT / "data/valueqa/evaluation_samples.jsonl")
        references = read_jsonl(ROOT / "data/valueqa/reference_annotations.jsonl")
        expected = ("sample_id", "triplet_id", "display_number", "source_id", "evaluation_dimension",
                    "primary_metric", "original_context", "target_comment", "question", "answer", "label")
        cfg = yaml.safe_load((ROOT / "configs/eval_tasks/valueqa_classification_api.yaml").read_text(encoding="utf-8"))
        dataset = DATASETS.build(cfg["dataset"])
        for sample, reference, record in zip(samples, references, dataset):
            self.assertEqual(tuple(sample), expected)
            self.assertEqual(tuple(reference), ("sample_id", "triplet_id", "label", "annotations"))
            self.assertEqual(sample["sample_id"], reference["sample_id"])
            self.assertEqual(sample["triplet_id"], reference["triplet_id"])
            self.assertEqual(sample["label"], reference["label"])
            self.assertEqual({key: record[key] for key in expected}, sample)
            self.assertEqual(record["ground_truth_annotations"], reference["annotations"])
            self.assertEqual(set(record["qwen_input"]),
                             {"sample_id", "question", "answer", "target_comment", "original_context"})


if __name__ == "__main__":
    unittest.main()
