"""Offline integration, routing, persistence and public-runner coverage."""

import copy
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import yaml

import uni_eval
uni_eval.load_all_modules("value")
from uni_eval.registry import DATASETS, EVALUATORS, METRICS, MODELS, RUNNERS
from uni_eval.models.base import BaseModel
from attribution_framework.valueqa_integration import engine_sha256, normalize_framework_output
from uni_eval.models.valueqa_service import select_service

ROOT = Path(__file__).resolve().parents[1]


def baseline(level="high", spans=None):
    return json.dumps({"answer_is_grounded": "pass", "confidence": 90, "decisive_reason": "材料支持",
                       "missing_or_wrong_points": [], "predicted_level": level,
                       "problem_spans": spans or [], "question_coverage": "complete"}, ensure_ascii=False)


class StatusError(Exception):
    def __init__(self, status, text):
        super().__init__(text)
        self.status_code = status


class FakeClient:
    max_retries = 0
    api_key = "fake-valueqa-test-secret"

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **options):
        self.calls.append(options)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, model_dump=lambda: {"prompt_tokens": 1, "completion_tokens": 1})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response), finish_reason="stop")],
                               model="mock", usage=usage, model_dump_json=lambda: json.dumps({"raw_output": response}))


@MODELS.register_module()
class ValueQATestAPIModel(BaseModel):
    def __init__(self, responses=None, **kwargs):
        self.model_name, self.api_base = "mock", "https://api.invalid/v1"
        self.default_gen_kwargs = {"temperature": 0, "max_tokens": 2048, "thinking": {"type": "disabled"}}
        self.client = FakeClient(responses or [])

    def generate(self, prompts, **kwargs):
        raise AssertionError("ValueQA generation must use its evaluator adapter")


class ValueQAIntegrationTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "results").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="valueqa-test-", dir=str(ROOT / "results"))
        self.directory = Path(self.temporary.name).resolve()
        self.directory.relative_to((ROOT / "results").resolve())
        original = json.loads((ROOT / "data/valueqa/evaluation_samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.row = {**original, "sample_id": "example", "answer": "唯一错误片段", "label": "low"}
        self.write_data()

    def tearDown(self):
        self.directory.relative_to((ROOT / "results").resolve())
        self.temporary.cleanup()

    def write_data(self):
        self.input = self.directory / "samples.jsonl"
        self.references = self.directory / "references.jsonl"
        self.input.write_text(json.dumps(self.row, ensure_ascii=False) + "\n", encoding="utf-8")
        ref = {"sample_id": self.row["sample_id"], "label": self.row["label"],
               "annotations": [{"start": 0, "end": len(self.row["answer"]), "text": self.row["answer"]}] if self.row["label"] == "low" else []}
        self.references.write_text(json.dumps(ref, ensure_ascii=False) + "\n", encoding="utf-8")

    def dataset(self):
        return DATASETS.build({"type": "ValueQADataset", "path": str(self.input), "references_path": str(self.references)})

    def evaluator(self, method="single_llm", **kwargs):
        return EVALUATORS.build({"type": "ValueQASingleLLMEvaluator" if method == "single_llm" else "ValueQAFrameworkEvaluator",
                                 "state_dir": str(self.directory / method), "workers": 1, "attempts": 2, **kwargs})

    def test_one_response_joint_classification_and_localization(self):
        model = ValueQATestAPIModel([baseline("low", [self.row["answer"]])])
        results = self.evaluator().evaluate(model, self.dataset())
        scores = METRICS.build({"type": "ValueQAMetric"}).compute(results)
        self.assertEqual(len(model.client.calls), 1)
        self.assertEqual(scores["accuracy"], 1)
        self.assertEqual(scores["low_span_exact_f1"], 1)
        self.assertEqual(scores["low_span_char_f1"], 1)
        call = model.client.calls[0]
        self.assertNotIn("top_p", call)
        self.assertEqual(call["extra_body"], {"thinking": {"type": "disabled"}})
        text = call["messages"][1]["content"].split("\n\n输入：\n", 1)[1]
        data, _ = json.JSONDecoder().raw_decode(text)
        self.assertEqual(set(data), {"候选答案", "相关原文", "问题", "需要理解的目标评论"})
        self.assertNotIn("label", self.dataset()[0]["valueqa_input"])

    def test_malformed_response_is_preserved_and_not_reasked(self):
        model = ValueQATestAPIModel(["not JSON"])
        results = self.evaluator().evaluate(model, self.dataset())
        self.assertEqual(len(model.client.calls), 1)
        retry_model = ValueQATestAPIModel([])
        repeated = self.evaluator().evaluate(retry_model, self.dataset())
        self.assertEqual(len(retry_model.client.calls), 0)
        self.assertEqual(repeated[0]["prediction"], "not JSON")
        scores = METRICS.build({"type": "ValueQAMetric"}).compute(results)
        self.assertEqual(scores["total_samples"], 1)
        self.assertEqual(scores["invalid_predictions"], 1)

    def test_connection_failure_retries_without_changing_prompt(self):
        model = ValueQATestAPIModel([StatusError(503, "temporary"), baseline()])
        results = self.evaluator().evaluate(model, self.dataset())
        self.assertEqual(len(model.client.calls), 2)
        self.assertEqual(model.client.calls[0], model.client.calls[1])
        self.assertFalse(results[0]["classification_error"])

    def test_authentication_failure_pauses_and_redacts_key(self):
        model = ValueQATestAPIModel([StatusError(401, "rejected " + FakeClient.api_key)])
        results = self.evaluator().evaluate(model, self.dataset())
        self.assertEqual(len(model.client.calls), 1)
        self.assertTrue(results[0]["classification_error"])
        self.assertTrue((self.directory / "single_llm/.pause").exists())
        self.assertNotIn(FakeClient.api_key, (self.directory / "single_llm/api_audit.jsonl").read_text(encoding="utf-8"))

    def test_supported_review_does_not_trigger_low_confidence(self):
        first = json.dumps({"label": "high", "confidence": 1, "reason": "依据"}, ensure_ascii=False)
        review = json.dumps({"verdict": "supported", "reason": "支持", "issues": []}, ensure_ascii=False)
        model = ValueQATestAPIModel([first, review])
        results = self.evaluator("framework").evaluate(model, self.dataset())
        self.assertEqual(len(model.client.calls), 2)
        stages = results[0]["attribution"]["stages"]
        self.assertNotIn("agent3", stages)
        self.assertNotIn("agent1_recheck", stages)
        again = ValueQATestAPIModel([])
        self.evaluator("framework").evaluate(again, self.dataset())
        self.assertEqual(len(again.client.calls), 0)

    def test_agent3_direct_downgrade_and_low_locator(self):
        first = {"label": "high", "confidence": 95, "reason": "初判"}
        review = {"verdict": "contradicted", "reason": "中心误读", "issues": []}
        third = {"label": "low", "confidence": 95, "reason": "中心误读", "promotion_checks": []}
        location = {"issues": [{"span": self.row["answer"], "type": "distortion", "reason": "误读"}]}
        model = ValueQATestAPIModel([json.dumps(value, ensure_ascii=False) for value in (first, review, third, location)])
        results = self.evaluator("framework").evaluate(model, self.dataset())
        self.assertEqual(len(model.client.calls), 4)
        self.assertEqual(json.loads(results[0]["prediction"])["label"], "low")
        self.assertNotIn("agent1_recheck", results[0]["attribution"]["stages"])
        scores = METRICS.build({"type": "ValueQAMetric", "method": "framework"}).compute(results)
        self.assertEqual(scores["accuracy"], 1)
        self.assertEqual(scores["low_span_exact_f1"], 1)

    def test_agent3_promotion_without_evidence_keeps_initial(self):
        responses = [{"label": "medium", "confidence": 90, "reason": "缺失关键关系"},
                     {"verdict": "unclear", "reason": "待核验", "issues": []},
                     {"label": "high", "confidence": 99, "reason": "完整", "promotion_checks": []}]
        model = ValueQATestAPIModel([json.dumps(value, ensure_ascii=False) for value in responses])
        results = self.evaluator("framework").evaluate(model, self.dataset())
        self.assertEqual(len(model.client.calls), 3)
        self.assertEqual(json.loads(results[0]["prediction"])["label"], "medium")
        self.assertEqual(results[0]["attribution"]["final"]["change_guard"], "promotion_blocked_insufficient_evidence")

    def test_framework_authentication_error_does_not_leak_key_to_checkpoint(self):
        model = ValueQATestAPIModel([StatusError(401, "rejected " + FakeClient.api_key)])
        results = self.evaluator("framework").evaluate(model, self.dataset())
        self.assertTrue(results[0]["classification_error"])
        for name in ("api_audit.jsonl", "checkpoint.jsonl"):
            self.assertNotIn(FakeClient.api_key, (self.directory / "framework" / name).read_text(encoding="utf-8"))

    def test_long_reason_and_string_confidence_are_preserved_and_audited(self):
        reason = "题内关键关系仍未解释。" * 15
        first = {"label": "medium", "confidence": "72", "reason": reason}
        second = {"verdict": "supported", "reason": "支持初判", "issues": []}
        model = ValueQATestAPIModel([json.dumps(value, ensure_ascii=False) for value in (first, second)])
        results = self.evaluator("framework").evaluate(model, self.dataset())
        self.assertEqual(len(model.client.calls), 2)
        final = json.loads(results[0]["prediction"])
        self.assertEqual(final["label"], "medium")
        self.assertEqual(final["confidence"], 72)
        self.assertEqual(final["reason"], reason)
        events = [json.loads(line) for line in (self.directory / "framework/api_audit.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(json.loads(events[0]["raw_output"]), first)
        self.assertEqual(events[0]["normalized_output"]["reason"], reason)
        self.assertEqual(events[0]["output_warnings"][0]["actual_characters"], len(reason))
        self.assertEqual(events[0]["numeric_type_repairs"], ["confidence: numeric string to integer"])

    def test_missing_review_issues_on_low_requires_real_locator_output(self):
        first = {"label": "low", "confidence": 90, "reason": "中心误读"}
        second = {"verdict": "supported", "reason": "支持初判"}
        located = {"issues": [{"span": self.row["answer"], "type": "distortion", "reason": "误读"}]}
        model = ValueQATestAPIModel([json.dumps(value, ensure_ascii=False) for value in (first, second, located)])
        results = self.evaluator("framework").evaluate(model, self.dataset())
        self.assertEqual(len(model.client.calls), 3)
        self.assertIn("locator", results[0]["attribution"]["stages"])
        self.assertNotIn("agent3", results[0]["attribution"]["stages"])
        self.assertEqual(json.loads(results[0]["prediction"])["issues"][0]["span"], self.row["answer"])
        events = [json.loads(line) for line in (self.directory / "framework/api_audit.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertNotIn("issues", json.loads(events[1]["raw_output"]))
        self.assertEqual(events[1]["normalized_output"]["issues"], [])
        self.assertTrue(events[1]["output_repairs"])

    def test_missing_review_issues_on_dispute_still_routes_to_agent3(self):
        for verdict in ("contradicted", "unclear"):
            first = {"label": "high", "confidence": 90, "reason": "初判"}
            second = {"verdict": verdict, "reason": "待核实"}
            third = {"label": "low", "confidence": 88, "reason": "原文支持中心误读。" * 15, "promotion_checks": []}
            located = {"issues": [{"span": self.row["answer"], "type": "distortion", "reason": "误读"}]}
            model = ValueQATestAPIModel([json.dumps(value, ensure_ascii=False) for value in (first, second, third, located)])
            evaluator = EVALUATORS.build({"type": "ValueQAFrameworkEvaluator", "state_dir": str(self.directory / verdict), "workers": 1})
            results = evaluator.evaluate(model, self.dataset())
            self.assertEqual(len(model.client.calls), 4)
            final = results[0]["attribution"]["final"]
            self.assertEqual(final["decision_source"], "agent3")
            self.assertEqual(final["final"]["label"], "low")
            self.assertEqual(final["final"]["reason"], third["reason"])
            self.assertIsNone(final["recheck"])

    def test_mechanical_repairs_do_not_infer_semantic_fields_or_coerce_bad_types(self):
        good = {"label": "medium", "confidence": 72, "reason": "关键关系不足"}
        cases = [{**good, "label": "unknown"}, {**good, "reason": " "},
                 {**good, "extra": "unexpected"}, {"confidence": 72, "reason": "缺少等级"},
                 {"verdict": "unknown", "reason": "未知意见"},
                 {"verdict": "supported", "reason": "支持", "issues": None}]
        cases.extend({**good, "confidence": value} for value in (True, "72.5", "-1", "101", " 72", "72.0", "abc"))
        for value in cases:
            with self.subTest(value=value):
                raw = json.dumps(value, ensure_ascii=False)
                text, metadata = normalize_framework_output(raw)
                self.assertEqual(text, raw)
                self.assertFalse(metadata["output_schema_valid"])
                self.assertNotIn("normalized_output", metadata)

    def test_format_repair_does_not_relax_promotion_evidence(self):
        first = {"label": "medium", "confidence": 90, "reason": "缺少关键关系"}
        second = {"verdict": "unclear", "reason": "待核实"}
        third = {"label": "high", "confidence": "99", "reason": "声称已覆盖必要要求。" * 15, "promotion_checks": []}
        model = ValueQATestAPIModel([json.dumps(value, ensure_ascii=False) for value in (first, second, third)])
        results = self.evaluator("framework").evaluate(model, self.dataset())
        self.assertEqual(len(model.client.calls), 3)
        final = results[0]["attribution"]["final"]
        self.assertEqual(final["final"]["label"], "medium")
        self.assertEqual(final["change_guard"], "promotion_blocked_insufficient_evidence")

    def test_framework_format_repair_does_not_change_single_llm_contract(self):
        response = json.loads(baseline())
        response["decisive_reason"] = "材料支持当前解释。" * 20
        raw = json.dumps(response, ensure_ascii=False)
        model = ValueQATestAPIModel([raw])
        results = self.evaluator().evaluate(model, self.dataset())
        self.assertEqual(len(model.client.calls), 1)
        self.assertEqual(results[0]["prediction"], raw)
        scores = METRICS.build({"type": "ValueQAMetric"}).compute(results)
        self.assertEqual(scores["contract_valid_predictions"], 0)
        event = json.loads((self.directory / "single_llm/api_audit.jsonl").read_text(encoding="utf-8"))
        self.assertNotIn("output_repairs", event)

    def test_model_settings_change_rejects_checkpoint_reuse(self):
        self.evaluator().evaluate(ValueQATestAPIModel([baseline()]), self.dataset())
        model = ValueQATestAPIModel([])
        model.default_gen_kwargs["max_tokens"] = 4096
        with self.assertRaisesRegex(ValueError, "changed"):
            self.evaluator().evaluate(model, self.dataset())

    def test_model_concurrency_cap_limits_simultaneous_requests(self):
        dataset = list(self.dataset())
        dataset.append({**copy.deepcopy(dataset[0]), "id": "second",
                        "valueqa_input": {**dataset[0]["valueqa_input"], "sample_id": "second"}})
        model = ValueQATestAPIModel([baseline(), baseline()])
        model.concurrency = 1
        active, peak = [0], [0]
        lock = threading.Lock()
        create = model.client.create

        def observed(**kwargs):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            try:
                time.sleep(0.03)
                return create(**kwargs)
            finally:
                with lock:
                    active[0] -= 1

        model.client.chat.completions.create = observed
        evaluator = EVALUATORS.build({"type": "ValueQASingleLLMEvaluator", "state_dir": str(self.directory / "cap"), "workers": 2})
        results = evaluator.evaluate(model, dataset)
        self.assertEqual(len(results), 2)
        self.assertEqual(peak[0], 1)

    def test_public_generation_and_offline_scoring_and_input_guard(self):
        config = {"model": {"type": "ValueQATestAPIModel", "responses": [baseline("low", [self.row["answer"]])]},
                  "dataset": {"type": "ValueQADataset", "path": str(self.input), "references_path": str(self.references)},
                  "evaluator": {"type": "ValueQASingleLLMEvaluator", "state_dir": str(self.directory / "state"), "workers": 1},
                  "metrics": [{"type": "ValueQAMetric"}], "summarizer": {"type": "StandardSummarizer"},
                  "runner": {"type": "LocalRunner", "prediction_policy": "strict", "use_evaluator_gen": True,
                             "stage": "gen", "output_dir": str(self.directory / "public"), "badcase_unknown_samples": 0}}
        RUNNERS.build(config["runner"]).run(config)
        config["runner"]["stage"] = "eval"
        RUNNERS.build(config["runner"]).run(config)
        self.row["evaluation_dimension"] = "changed"
        self.write_data()
        with self.assertRaisesRegex(ValueError, "input mismatch"):
            RUNNERS.build(config["runner"]).run(config)

    def test_task_configs_do_not_select_a_connection_or_model(self):
        for name in ("value_classification", "value_attribution"):
            config = yaml.safe_load((ROOT / "configs/eval_tasks" / (name + ".yaml")).read_text(encoding="utf-8"))
            model = config["model"]
            self.assertIsNone(model["connection"])
            for connection in ("api", "local"):
                self.assertIsNone(model[connection]["model_name"])
                self.assertIsNone(model[connection]["api_base"])
            with patch("uni_eval.models.api.OpenAI") as client:
                with self.assertRaisesRegex(ValueError, "model.connection"):
                    MODELS.build(model)
                client.assert_not_called()

    def test_selector_only_initializes_the_selected_service(self):
        api = {"model_name": "chosen-api-model", "api_base": "https://api.invalid/v1", "api_key": "ENV",
               "temperature": 0, "max_tokens": 2048, "thinking": {"type": "disabled"}}
        local = {"model_name": "chosen-local-model", "api_base": "http://127.0.0.1:11434/v1",
                 "temperature": 0, "max_tokens": 2048}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "private-test-key"}), patch("uni_eval.models.api.OpenAI") as client:
            remote = MODELS.build({"type": "ValueQAServiceModel", "connection": "api", "api": api})
            self.assertEqual(remote.model_name, "chosen-api-model")
            self.assertEqual(client.call_args.kwargs["api_key"], "private-test-key")
            self.assertEqual(client.call_args.kwargs["base_url"], api["api_base"])
            self.assertEqual(remote.concurrency, 10)
            client.reset_mock()
            model = MODELS.build({"type": "ValueQAServiceModel", "connection": "local", "api": api, "local": local})
            client.assert_called_once()
            self.assertEqual(client.call_args.kwargs["api_key"], "EMPTY")
            self.assertEqual(client.call_args.kwargs["base_url"], local["api_base"])
            self.assertEqual(model.model_name, "chosen-local-model")
            self.assertEqual(model.concurrency, 1)
            self.assertNotIn("thinking", model.default_gen_kwargs)

    def test_service_configuration_errors_are_reported_before_requests(self):
        good = {"model_name": "configured-model", "api_base": "http://127.0.0.1:8000/v1"}
        cases = [(None, good, "model.connection"),
                 ("local", {**good, "model_name": None}, "model.local.model_name"),
                 ("local", {**good, "api_base": None}, "model.local.api_base"),
                 ("local", {**good, "api_base": "not-an-address"}, "HTTP"),
                 ("local", {**good, "max_retries": 3}, "max_retries"),
                 ("local", {**good, "concurrency": 0}, "concurrency"),
                 ("local", {**good, "mode": "completion"}, "chat/completions")]
        for connection, selected, message in cases:
            with self.subTest(connection=connection, message=message):
                with patch("uni_eval.models.api.OpenAI") as client:
                    with self.assertRaisesRegex(ValueError, message):
                        MODELS.build({"type": "ValueQAServiceModel", "connection": connection, "local": selected})
                    client.assert_not_called()

    def test_inactive_service_settings_are_optional(self):
        api = {"model_name": "only-api", "api_base": "https://api.invalid/v1"}
        local = {"model_name": "only-local", "api_base": "http://127.0.0.1:8000/v1"}
        self.assertEqual(select_service("api", api, {"model_name": None})["model_name"], "only-api")
        self.assertEqual(select_service("local", {"api_base": None}, local)["model_name"], "only-local")

    def test_both_replacement_yaml_tasks_over_local_http(self):
        calls = []
        fail = [False]

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append(payload)
                text = payload["messages"][-1]["content"]
                context, _ = json.JSONDecoder().raw_decode(text.rsplit("\n\n输入：\n", 1)[1])
                answer = context.get("answer", context.get("候选答案"))
                label = {"A": "high", "B": "medium", "C": "low"}[answer]
                if "predicted_level" in text:
                    output = baseline(label, [answer] if label == "low" else [])
                elif "verdict" in text:
                    output = json.dumps({"verdict": "supported", "reason": "材料支持",
                                         "issues": [{"span": answer, "type": "distortion", "reason": "误读"}] if label == "low" else []}, ensure_ascii=False)
                else:
                    output = json.dumps({"label": label, "confidence": 90, "reason": "核对材料"}, ensure_ascii=False)
                if fail[0]:
                    output = "invalid JSON"
                body = json.dumps({"id": "local", "object": "chat.completion", "created": 0,
                                   "model": "local-test-model", "choices": [{"index": 0,
                                   "message": {"role": "assistant", "content": output}, "finish_reason": "stop"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        records = [{**self.row, "sample_id": str(index), "answer": answer, "label": label}
                   for index, (answer, label) in enumerate(zip("ABC", ("high", "medium", "low")))]
        self.input.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        refs = [{"sample_id": row["sample_id"], "label": row["label"],
                 "annotations": [{"start": 0, "end": 1, "text": "C"}] if row["label"] == "low" else []} for row in records]
        self.references.write_text("".join(json.dumps(row) + "\n" for row in refs), encoding="utf-8")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            env = {**os.environ, "OPENAI_API_KEY": "LOCAL_VALUEQA_TEST_SECRET", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
            for name, connection in ((name, connection) for name in ("value_classification", "value_attribution")
                                      for connection in ("api", "local")):
                fail[0] = False
                config = yaml.safe_load((ROOT / "configs/eval_tasks" / (name + ".yaml")).read_text(encoding="utf-8"))
                config["model"]["connection"] = connection
                config["model"][connection].update(model_name=connection + "-test-model", api_base=f"http://127.0.0.1:{server.server_port}/v1")
                if connection == "api":
                    config["model"]["api"]["thinking"] = {"type": "disabled"}
                config["dataset"].update(path=str(self.input), references_path=str(self.references))
                config["evaluator"].update(state_dir=str(self.directory / (name + "-state")), workers=1)
                config["runner"]["output_dir"] = str(self.directory / name)
                config_path = self.directory / (name + ".yaml")
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                for stage in ("all", "gen", "eval"):
                    before = len(calls)
                    command = subprocess.run([sys.executable, "-B", str(ROOT / "tools/run.py"), str(config_path), "--runner.stage", stage],
                                             cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
                    self.assertEqual(command.returncode, 0, command.stdout + command.stderr)
                    if stage in ("gen", "eval"):
                        self.assertEqual(len(calls), before)
                result = json.loads((self.directory / name / connection / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(result["metrics"]["accuracy"], 1)
                self.assertEqual(result["metrics"]["low_span_exact_f1"], 1)
                self.assertNotIn("LOCAL_VALUEQA_TEST_SECRET", json.dumps(result))
                self.assertEqual(result["details"][0]["run_info"]["model"], connection + "-test-model")
                fail[0] = True
                config["evaluator"]["state_dir"] = str(self.directory / (name + "-failed-state"))
                config["runner"]["output_dir"] = str(self.directory / (name + "-failed"))
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                command = subprocess.run([sys.executable, "-B", str(ROOT / "tools/run.py"), str(config_path)],
                                         cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
                self.assertEqual(command.returncode, 0, command.stdout + command.stderr)
                result = json.loads((self.directory / (name + "-failed") / connection / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(result["metrics"]["total_samples"], 3)
                self.assertEqual(result["metrics"]["invalid_predictions"], 3)
                self.assertEqual(result["metrics"]["accuracy"], 0)
            self.assertTrue(calls)
            self.assertTrue(all("top_p" not in payload for payload in calls))
            self.assertTrue(all(payload.get("thinking") == {"type": "disabled"} for payload in calls if payload["model"] == "api-test-model"))
            self.assertTrue(all("thinking" not in payload for payload in calls if payload["model"] == "local-test-model"))
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
