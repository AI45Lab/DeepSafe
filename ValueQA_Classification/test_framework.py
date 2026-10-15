"""Deterministic gates, failure recovery and real mock-HTTP transport; no GPU/model downloads."""
import ast
import copy
import json
import os
import sys
from types import SimpleNamespace
import tempfile
import threading
import unittest
import contextlib
import io
import subprocess
import zipfile
from urllib.error import URLError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

if __package__:
    from . import backend as backend_module, check_env as check_env_module
    from . import build_server_package as package_module
    from . import gate as gate_module, run as run_module
    from .backend import APIBackend, BackendError, ReviewAPIBackend, ReviewVLLMBackend
    from .common import (Checkpoint, ROOT, blind, digest, load_samples, model_directory_identity,
                         parse_json, read_json, read_jsonl, file_hash, write_json, write_jsonl)
    from .evaluate import report
    from .gate import decide, gate_identity, validate_policy, validate_review
    from .initial_protocol import system_prompt, prompt_ids, prepare_samples
    from .prompts import review_messages
    from .boundary_test_support import certify_review, stage_response, visible_sample
    from .run import (engine_identity, execute_stage, freeze_outputs, import_initials,
                      load_config, main, process_rows, process_phased)
else:
    import backend as backend_module
    import build_server_package as package_module
    import check_env as check_env_module
    import gate as gate_module
    import run as run_module
    from backend import APIBackend, BackendError, ReviewAPIBackend, ReviewVLLMBackend
    from common import (Checkpoint, ROOT, blind, digest, load_samples, model_directory_identity,
                        parse_json, read_json, read_jsonl, file_hash, write_json, write_jsonl)
    from evaluate import report
    from gate import decide, gate_identity, validate_policy, validate_review
    from initial_protocol import system_prompt, prompt_ids, prepare_samples
    from prompts import review_messages
    from boundary_test_support import certify_review, stage_response, visible_sample
    from run import (engine_identity, execute_stage, freeze_outputs, import_initials,
                     load_config, main, process_rows, process_phased)


ROW = {"sample_id":"one", "original_context":"台风导致会议延期，最终改到周五。",
       "target_comment":"会议改到周五。", "question":"为什么延期，改到哪一天？", "answer":"因为台风。"}
SCHEMA = {"instructions":"核对材料。", "criteria":{"high":"充分", "medium":"局部不足", "low":"中心误读"}}


def q(field, text):
    return {"field":field, "text":text}


def fixture_config(directory):
    """Portable dummy files are used only by HTTP mocks; no model is loaded."""
    config=read_json(ROOT/"config.current.server.json")
    config.update(review_backend="api",review_api_base="http://127.0.0.1:1/v1",
                  review_api_model="mock-review",review_api_workers=1,
                  api_transport="urllib",review_api_transport="urllib",review_api_warmup_models=False,
                  quantization="none",mode="observe")
    adapter=Path(directory)/"mock_adapter"
    adapter.mkdir(parents=True,exist_ok=True)
    write_json(adapter/"adapter_config.json",{"base_model_name_or_path":config["base_model"]})
    write_json(adapter/"training_config.json",{"model_revision":config["base_revision"]})
    write_json(adapter/"classification_schema.json",{"question_schema":SCHEMA})
    for name in ["adapter_model.safetensors","tokenizer.json","tokenizer_config.json","chat_template.jinja"]:
        (adapter/name).write_text("mock-only",encoding="utf-8")
    config["adapter_path"]=str(adapter)
    return config


def medium_review():
    material = [q("original_context","台风导致会议延期，最终改到周五。")]
    value = {"label":"medium", "reason":"原因正确但遗漏日期", "review_complete":True,
        "requirements_complete":True, "residual_core_check_complete":True,
        "core":{"status":"correct", "reason":"延期原因正确", "answer_evidence":[q("answer","因为台风。")], "material_evidence":material},
        "requirements":[
            {"id":"R1", "point":"延期原因", "status":"met", "reason":"原因已表达", "question_evidence":[q("question","为什么延期")], "material_evidence":material,"answer_evidence":[q("answer","因为台风。") ]},
            {"id":"R2", "point":"新日期", "status":"missing", "reason":"题目要求而答案遗漏", "question_evidence":[q("question","改到哪一天")], "material_evidence":material,"answer_evidence":[]}],
        "issues":[{"requirement_id":"R2", "kind":"local", "status":"confirmed", "reason":"日期必要而未表达", "missing_point":"周五", "answer_evidence":[], "material_evidence":material}]}
    return certify_review(value, ROW)


def complete_review(row):
    value = medium_review()
    value.update(label="high", reason="完整", issues=[])
    value["core"]["answer_evidence"] = [q("answer",row["answer"])]
    for requirement in value["requirements"]:
        requirement.update(status="met", answer_evidence=[q("answer",row["answer"])])
    return certify_review(value, row)


class GateTests(unittest.TestCase):
    def test_valid_local_omission_and_observe_mode(self):
        value = medium_review()
        self.assertEqual(validate_review(value,ROW)[0], [])
        observed = decide({"label":"high"},value,ROW,"observe")
        self.assertEqual((observed["final_label"],observed["experimental_label"]),("high","medium"))
        self.assertFalse(observed["accepted"])
        self.assertEqual(decide({"label":"high"},value,ROW,"experimental")["final_label"],"medium")

    def test_fake_quote_cannot_override(self):
        value = medium_review()
        value["core"]["material_evidence"] = [q("original_context","会议改到周一。")]
        self.assertEqual(decide({"label":"high"},value,ROW,"experimental")["final_label"],"high")

    def test_high_requires_positive_complete_checks(self):
        row = {**ROW,"answer":"因为台风，改到周五。"}
        value = complete_review(row)
        self.assertEqual(decide({"label":"low"},value,row,"experimental")["final_label"],"high")
        value["requirements"][0]["answer_evidence"] = []
        self.assertEqual(decide({"label":"low"},value,row,"experimental")["final_label"],"low")

    def test_low_needs_core_misreading_not_low_score(self):
        value = medium_review()
        value["label"] = "low"
        self.assertTrue(validate_review(value,ROW)[0])

    def test_uncertain_and_incomplete_are_rejected(self):
        for change in [dict(review_complete=False),dict(requirements_complete=False),dict(residual_core_check_complete=False)]:
            value = {**medium_review(),**change}
            self.assertEqual(decide({"label":"high"},value,ROW,"experimental")["final_label"],"high")
        value = medium_review()
        value["requirements"][1]["status"]="uncertain"
        self.assertTrue(validate_review(value,ROW)[0])

    def test_malformed_model_fields_do_not_crash(self):
        value=medium_review()
        value["issues"][0]["requirement_id"] = []
        self.assertTrue(validate_review(value,ROW)[0])
        value=medium_review()
        value["core"]["answer_evidence"][0]["field"]=[]
        self.assertTrue(validate_review(value,ROW)[0])

    def test_initial_failure_does_not_invent_medium(self):
        self.assertIsNone(decide(None,medium_review(),ROW,"experimental")["final_label"])

    def test_policy_must_match_and_be_validated(self):
        policy={"schema_version":2,"engine_identity":"engine","gate_identity":gate_identity(),"validated":True,
                "enabled_transitions":["high->medium"],"validation_protocol":"independent confirmation",
                "confirmation_dataset_sha256":"actual-hash"}
        validate_policy(policy,"engine")
        self.assertTrue(decide({"label":"high"},medium_review(),ROW,"calibrated",policy)["accepted"])
        policy["enabled_transitions"]=[]
        self.assertFalse(decide({"label":"high"},medium_review(),ROW,"calibrated",policy)["accepted"])
        policy["validated"]=False
        with self.assertRaises(ValueError): validate_policy(policy,"engine")


class WorkflowTests(unittest.TestCase):
    def checkpoint(self,path):
        return Checkpoint(path,"identity",{"one":digest(blind(ROW))})

    def test_training_protocol_source_and_full_window_coverage(self):
        config=read_json(ROOT/"config.current.local.json") if (ROOT/"config.current.local.json").exists() else {}
        source=Path(config.get("adapter_path","__absent__")).parent/"code_snapshot"/"common.py"
        if source.exists():
            original=source.read_text(encoding="utf-8")
            copied=(ROOT/"initial_protocol.py").read_text(encoding="utf-8")
            original_functions={n.name:ast.get_source_segment(original,n) for n in ast.parse(original).body if isinstance(n,ast.FunctionDef)}
            for node in ast.parse(copied).body:
                if isinstance(node,ast.FunctionDef):
                    self.assertEqual(ast.get_source_segment(copied,node),original_functions[node.name])
        class Tokenizer:
            eos_token_id=1
            pad_token_id=0
            def encode(self,text,add_special_tokens=False): return {"high":[11],"medium":[12],"low":[13]}[text]
            def apply_chat_template(self,messages,**kwargs):
                user=json.loads(messages[1]["content"])
                self.assert_fields=set(user)
                return list(range(len(json.dumps(messages,ensure_ascii=False))))
            def __call__(self,text,**kwargs): return {"offset_mapping":[(i,i+1) for i in range(len(text))]}
        tok=Tokenizer()
        row={**ROW,"original_context":"材料很长。\n"*200}
        sample=prepare_samples([row],tok,{"max_length":512,"context_overlap_tokens":16},SCHEMA)[0]
        self.assertGreater(len(sample["windows"]),1)
        covered=0
        for start,end in sample["window_spans"]:
            self.assertLessEqual(start,covered)
            covered=max(covered,end)
        self.assertEqual(covered,len(row["original_context"]))
        self.assertEqual(tok.assert_fields,set(blind(row)))

    def test_pause_after_initial_then_resume_review_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"state.jsonl"
            cp=self.checkpoint(path)
            stopped=[False]
            calls=[]
            class Fake:
                def initial(self,row):
                    calls.append("initial")
                    stopped[0]=True
                    return {"label":"high"},{}
                def review(self,messages):
                    calls.append("review")
                    return json.dumps(stage_response(messages, medium_review())),{"finish_reason":"stop"}
                def clear_after_error(self): pass
            cfg={"initial_attempts":2,"review_attempts":1,"review_backend":"api","review_api_workers":1}
            process_rows([ROW],SCHEMA,Fake,cp,cfg,lambda:stopped[0])
            self.assertEqual(cp.get("one","initial")["status"],"succeeded")
            stopped[0]=False
            cp=self.checkpoint(path)
            process_rows([ROW],SCHEMA,Fake,cp,cfg,lambda:stopped[0])
            self.assertEqual(calls,["initial","review"])
            process_rows([ROW],SCHEMA,Fake,cp,cfg,lambda:False)
            self.assertEqual(calls,["initial","review"])

    def test_review_failure_is_terminal_and_keeps_initial(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp=self.checkpoint(Path(tmp)/"state.jsonl")
            cp.record("one","initial",status="succeeded",attempt=1,output={"label":"high"})
            calls=[]
            def fail():
                calls.append(1)
                raise BackendError("HTTP 400: parameter rejected")
            execute_stage(cp,"one","review",fail,1,lambda:False)
            execute_stage(cp,"one","review",fail,1,lambda:False)
            self.assertEqual(calls,[1])
            self.assertEqual(freeze_outputs([ROW],cp,"experimental")[0]["final_label"],"high")

    def test_truncated_output_keeps_initial_and_counts_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp=self.checkpoint(Path(tmp)/"state.jsonl")
            event=execute_stage(cp,"one","review",lambda:(json.dumps(medium_review()),{"finish_reason":"length","completion_tokens":50}),1,lambda:False)
            self.assertEqual(event["status"],"terminal_failed")
            self.assertEqual(event["usage"]["completion_tokens"],50)

    def test_trailing_checkpoint_fragment_recovered_and_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"state.jsonl"
            cp=self.checkpoint(path)
            cp.record("one","initial",status="succeeded",attempt=1,output={"label":"high"})
            with path.open("ab") as f: f.write(b'{"partial":')
            self.assertEqual(self.checkpoint(path).get("one","initial")["output"]["label"],"high")
            with self.assertRaises(ValueError): Checkpoint(path,"different",{"one":digest(blind(ROW))})

    def test_interrupted_review_never_receives_new_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp=self.checkpoint(Path(tmp)/"state.jsonl")
            cp.record("one","initial",status="succeeded",attempt=1,output={"label":"high"})
            cp.record("one","review",status="running",attempt=1)
            def factory(): self.fail("Exhausted interrupted review should not load a model")
            process_rows([ROW],SCHEMA,factory,cp,{"initial_attempts":2,"review_attempts":1},lambda:False)
            self.assertEqual(cp.get("one","review")["status"],"terminal_failed")

    def test_imported_predictions_strip_reference_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"predictions.jsonl"
            write_jsonl(path,[{"sample_id":"one","predicted_level":"high","reference_level":"low","correct":False}])
            cp=self.checkpoint(Path(tmp)/"state.jsonl")
            import_initials(path,[ROW],cp)
            self.assertNotIn("reference_level",cp.get("one","initial")["output"])

    def test_import_rejects_evaluation_and_reference_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"not_predictions.jsonl"
            for item in ({**ROW,"label":"high"},
                         {"sample_id":"one","label":"high","annotations":[]},
                         {"sample_id":"one","label":"high"}):
                write_jsonl(path,[item])
                cp=self.checkpoint(Path(tmp)/"state.jsonl")
                with self.assertRaisesRegex(ValueError,"Explicit prediction field required"):
                    import_initials(path,[ROW],cp)
                self.assertEqual(cp.events,[])

    def test_import_checks_entire_file_before_writing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=[ROW,{**ROW,"sample_id":"two"}]
            cp=Checkpoint(Path(tmp)/"cp.jsonl","id",{r["sample_id"]:digest(blind(r)) for r in rows})
            path=Path(tmp)/"initial.jsonl"
            write_jsonl(path,[{"sample_id":"one","predicted_level":"high"},
                             {"sample_id":"two","label":"low"}])
            with self.assertRaisesRegex(ValueError,"Explicit prediction field required"):
                import_initials(path,rows,cp)
            self.assertEqual(cp.events,[])
            self.assertFalse(cp.path.exists())

    def test_import_uses_explicit_prediction_and_rejects_conflicting_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"initial.jsonl"
            write_jsonl(path,[{"sample_id":"one","initial_label":"high","label":"low"}])
            cp=self.checkpoint(Path(tmp)/"state.jsonl")
            import_initials(path,[ROW],cp)
            self.assertEqual(cp.get("one","initial")["output"]["label"],"high")
            write_jsonl(path,[{"sample_id":"one","predicted_level":"high","predicted_label":"low"}])
            with self.assertRaisesRegex(ValueError,"Conflicting imported prediction fields"):
                import_initials(path,[ROW],cp)

    def test_metrics_keep_failures_but_separate_unstarted_rows(self):
        rows=[{**ROW,"label":"medium"},{**ROW,"sample_id":"two","label":"high"},
              {**ROW,"sample_id":"three","label":"high"}]
        outputs=[{"sample_id":"one","initial_label":"high","experimental_label":"medium","final_label":"high","initial_status":"succeeded"},
                 {"sample_id":"two","initial_label":None,"experimental_label":None,"final_label":None,"initial_status":"terminal_failed"},
                 {"sample_id":"three","initial_status":"pending"}]
        summary,_,_=report(rows,outputs)
        self.assertEqual(summary["evaluated_samples"],2)
        self.assertEqual(summary["scores"]["initial_label"]["acc"],0)
        self.assertEqual(summary["scores"]["experimental_label"]["acc"],0.5)

    def test_no_grade_or_reference_in_review_inputs(self):
        row={**ROW,"label":"high","reference_canary":"DO_NOT_SEND"}
        messages=review_messages(row,SCHEMA)
        self.assertNotIn("DO_NOT_SEND",json.dumps(messages))
        self.assertEqual(json.loads(messages[1]["content"].split("\n",1)[1]),blind(row))


class MockHTTPTests(unittest.TestCase):
    def test_cli_limit_resume_and_redecide_do_not_recall_models(self):
        bodies=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                bodies.append(body)
                text="high" if body["max_tokens"]==8 else json.dumps(stage_response(body["messages"], medium_review()))
                self.send_response(200);self.end_headers()
                self.wfile.write(json.dumps({"choices":[{"message":{"content":text},"finish_reason":"stop"}],
                    "usage":{"prompt_tokens":10,"completion_tokens":20}}).encode())
            def log_message(self,*args): pass
        with tempfile.TemporaryDirectory() as tmp, ThreadingHTTPServer(("127.0.0.1",0),Handler) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            root=Path(tmp)
            rows=[{**ROW,"sample_id":str(i),"label":"medium","reference_canary":"NEVER_SERIALIZE"} for i in range(3)]
            write_jsonl(root/"samples.jsonl",rows)
            write_jsonl(root/"refs.jsonl",[{"sample_id":str(i),"label":"medium"} for i in range(3)])
            config=fixture_config(root)
            config.update(backend="api",api_base=f"http://127.0.0.1:{server.server_port}/v1",
                input=str(root/"samples.jsonl"),references=str(root/"refs.jsonl"),output_dir=str(root/"out"),
                cache_dir=str(root/"cache"),api_model="classifier",review_backend="api",
                review_api_base=f"http://127.0.0.1:{server.server_port}/v1",review_api_model="base")
            write_json(root/"config.json",config)
            arguments=["--config",str(root/"config.json")]
            try:
                with patch("builtins.print"), patch.dict(os.environ,{"QWEN_BASE_PATH":"","QWEN_LORA_PATH":"","VLLM_PYTHON":""}):
                    self.assertEqual(main(arguments+["--limit","1"]),0)
                    first=read_json(root/"out"/"metrics.json")
                    self.assertEqual(first["evaluated_samples"],1)
                    self.assertEqual(len(bodies),2)
                    self.assertEqual(main(arguments),0)
                    self.assertEqual(len(bodies),6)
                    self.assertEqual([b["model"] for b in bodies],
                                     ["classifier","base","classifier","classifier","base","base"])
                    self.assertEqual(main(arguments+["--redecide-only","--mode","experimental"]),0)
                    self.assertEqual(len(bodies),6)
                    last=read_json(root/"out"/"metrics.json")
                self.assertEqual(last["scores"]["initial_label"]["acc"],0)
                self.assertEqual(last["scores"]["final_label"]["acc"],1)
                self.assertEqual(last["accepted_changes"],3)
                self.assertNotIn("NEVER_SERIALIZE",json.dumps(bodies))
            finally:
                server.shutdown();thread.join()

    def test_http_transport_finish_reason_and_error_body(self):
        bodies=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                bodies.append(body)
                if body["model"]=="reject":
                    self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"bad parameter"}')
                    return
                text="high" if body["max_tokens"]==8 else json.dumps(stage_response(body["messages"], medium_review()))
                self.send_response(200); self.send_header("Content-Type","application/json");self.end_headers()
                self.wfile.write(json.dumps({"choices":[{"message":{"content":text},"finish_reason":"stop"}],
                                             "usage":{"prompt_tokens":10,"completion_tokens":20}}).encode())
            def log_message(self,*args): pass
        with tempfile.TemporaryDirectory() as tmp, ThreadingHTTPServer(("127.0.0.1",0),Handler) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            config=fixture_config(Path(tmp))
            config.update(api_base=f"http://127.0.0.1:{server.server_port}/v1",api_model="initial",api_review_model="review")
            client=APIBackend(config)
            try:
                self.assertEqual(client.initial(ROW)[0]["label"],"high")
                text,usage=client.review(review_messages(ROW,SCHEMA))
                self.assertEqual(parse_json(text)["label"],"medium")
                self.assertEqual(usage["finish_reason"],"stop")
                self.assertEqual(json.loads(bodies[0]["messages"][1]["content"]),blind(ROW))
                config["api_model"]="reject"
                with self.assertRaisesRegex(BackendError,"bad parameter"): client.initial(ROW)
            finally:
                server.shutdown();thread.join()


class PhasedBackendTests(unittest.TestCase):
    def test_dead_worker_between_requests_keeps_unsent_reviews_pending_and_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=[{**ROW,"sample_id":str(i)} for i in range(3)]
            cp=Checkpoint(Path(tmp)/"cp.jsonl","id",{r["sample_id"]:digest(blind(r)) for r in rows})
            for row in rows: cp.record(row["sample_id"],"initial",status="succeeded",attempt=0,output={"label":"high"})
            class DiesBetweenRequests(ReviewVLLMBackend):
                def __init__(self):
                    self.healthy=True
                    self.exited=False
                    self.process=SimpleNamespace(poll=lambda:1 if self.exited else None)
                def review(self,messages):
                    self.exited=True
                    return json.dumps(stage_response(messages, medium_review())),{"finish_reason":"stop"}
                def close(self): pass
            def forbidden(): self.fail("Saved initials must not be loaded or recomputed")
            reviewer=DiesBetweenRequests()
            cfg={"initial_attempts":2,"review_attempts":1,"review_backend":"vllm"}
            process_phased(rows,SCHEMA,forbidden,lambda:reviewer,cp,cfg,lambda:False,imported_only=True)
            self.assertFalse(reviewer.healthy)
            self.assertEqual(cp.get("0","review")["status"],"succeeded")
            self.assertIsNone(cp.get("1","review"))
            self.assertIsNone(cp.get("2","review"))
            calls=[]
            class Healthy:
                def review(self,messages):
                    calls.append(1)
                    return json.dumps(stage_response(messages, medium_review())),{"finish_reason":"stop"}
                def close(self): pass
                def clear_after_error(self): pass
            cp=Checkpoint(Path(tmp)/"cp.jsonl","id",{r["sample_id"]:digest(blind(r)) for r in rows})
            process_phased(rows,SCHEMA,forbidden,Healthy,cp,cfg,lambda:False,imported_only=True)
            self.assertEqual(len(calls),2)
            self.assertTrue(all(cp.get(r["sample_id"],"review")["status"]=="succeeded" for r in rows))

    def test_send_to_dead_worker_marks_it_unhealthy(self):
        client=object.__new__(ReviewVLLMBackend)
        client.healthy=True
        client.process=SimpleNamespace(poll=lambda:1)
        with self.assertRaisesRegex(BackendError,"unavailable"): client._send({"command":"review"})
        self.assertFalse(client.healthy)

    def test_unload_before_review_and_resume_without_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=[ROW,{**ROW,"sample_id":"two"}]
            cp=Checkpoint(Path(tmp)/"cp.jsonl","id",{r["sample_id"]:digest(blind(r)) for r in rows})
            calls=[]
            class Initial:
                def initial(self,row):
                    calls.append("initial:"+row["sample_id"])
                    return {"label":"high"},{}
                def close(self): calls.append("close_initial")
                def clear_after_error(self): pass
            class Review:
                def review(self,messages):
                    calls.append("review")
                    return json.dumps(stage_response(messages, medium_review())),{"finish_reason":"stop"}
                def close(self): calls.append("close_review")
                def clear_after_error(self): pass
            cfg={"initial_attempts":2,"review_attempts":1,"review_backend":"api","review_api_workers":1}
            process_phased(rows,SCHEMA,Initial,Review,cp,cfg,lambda:False)
            self.assertEqual(calls,["initial:one","initial:two","close_initial","review","review","close_review"])
            def forbidden(): self.fail("Completed work must not reload models")
            process_phased(rows,SCHEMA,forbidden,forbidden,cp,cfg,lambda:False)

    def test_cached_initial_review_failure_keeps_initial(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp=Checkpoint(Path(tmp)/"cp.jsonl","id",{"one":digest(blind(ROW))})
            cp.record("one","initial",status="succeeded",attempt=0,output={"label":"high"})
            class Review:
                def review(self,messages): raise BackendError("evidence unavailable")
                def close(self): pass
                def clear_after_error(self): pass
            def forbidden(): self.fail("Imported-only must not load classifier")
            process_phased([ROW],SCHEMA,forbidden,Review,cp,{"initial_attempts":2,"review_attempts":1,"review_api_workers":1},
                           lambda:False,imported_only=True)
            result=freeze_outputs([ROW],cp,"experimental")[0]
            self.assertEqual(result["final_label"],"high")
            self.assertEqual(result["review_status"],"terminal_failed")

    def test_import_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"initial.jsonl"
            write_jsonl(path,[{"sample_id":"one","predicted_level":"high","input_sha256":"wrong"}])
            cp=Checkpoint(Path(tmp)/"cp.jsonl","id",{"one":digest(blind(ROW))})
            with self.assertRaisesRegex(ValueError,"hash mismatch"): import_initials(path,[ROW],cp)

    def test_configuration_bounds_and_malformed_policy(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            cfg=fixture_config(Path(tmp)); cfg.update(review_backend="vllm",review_local_model="different-reviewer")
            path=Path(tmp)/"config.json"
            for key,value in [("review_vllm_gpu_memory_utilization",1.1),
                              ("review_vllm_tensor_parallel_size",True),("review_vllm_startup_timeout",0)]:
                write_json(path,{**cfg,key:value})
                with self.assertRaises(ValueError): load_config(path)
            cfg.update(review_backend="api",review_api_model="CHANGE_ME")
            write_json(path,cfg)
            with self.assertRaises(ValueError): load_config(path)
            policy={"schema_version":1,"engine_identity":"id","validated":True,
                    "enabled_transitions":[{}]}
            with self.assertRaises(ValueError): validate_policy(policy,"id")

    def test_review_defaults_to_api_and_never_implicitly_reuses_initial_model(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            cfg=fixture_config(Path(tmp))
            cfg.pop("review_backend")
            path=Path(tmp)/"config.json"
            write_json(path,cfg)
            self.assertEqual(load_config(path)["review_backend"],"api")
            for model in ("", "CHANGE_ME", cfg["base_model"]):
                write_json(path,{**cfg,"review_api_model":model})
                with self.assertRaises(ValueError): load_config(path)
            # An unused API-initial alias must not constrain Transformers initial classification.
            write_json(path,{**cfg,"review_api_model":cfg["api_model"]})
            self.assertEqual(load_config(path)["review_api_model"],cfg["api_model"])
            write_json(path,{**cfg,"backend":"api","review_api_model":cfg["api_model"]})
            with self.assertRaisesRegex(ValueError,"must differ"): load_config(path)
            write_json(path,{**cfg,"review_backend":"transformers"})
            with self.assertRaisesRegex(ValueError,"different model"): load_config(path)
            for model in ("", "CHANGE_ME", cfg["base_model"]):
                write_json(path,{**cfg,"review_backend":"vllm","review_local_model":model})
                with self.assertRaises(ValueError): load_config(path)

    def test_review_model_environment_and_explicit_override_priority(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            cfg=fixture_config(Path(tmp))
            path=Path(tmp)/"config.json"
            write_json(path,cfg)
            values={"REVIEW_API_BASE":"http://127.0.0.1:2/v1","REVIEW_API_MODEL":"env-review",
                    "REVIEW_MODEL_PATH":"env-local-review"}
            with patch.dict(os.environ,values):
                loaded=load_config(path)
                self.assertEqual(loaded["review_api_model"],"env-review")
                self.assertEqual(loaded["review_api_base"],values["REVIEW_API_BASE"])
                loaded=load_config(path,{"review_api_model":"explicit-review"})
                self.assertEqual(loaded["review_api_model"],"explicit-review")
                # Simulate the supported platform; these configuration tests do not load vLLM.
                with patch.object(run_module.sys, "platform", "linux"):
                    loaded=load_config(path,{"review_backend":"vllm"})
                self.assertEqual(loaded["review_local_model"],"env-local-review")

    def test_local_review_cannot_reuse_classifier_through_an_equivalent_path(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            root=Path(tmp)
            cfg=fixture_config(root)
            (root/"base").mkdir()
            cfg.update(base_model="base",review_backend="vllm",review_local_model="base/../base")
            path=root/"config.json"
            write_json(path,cfg)
            with patch.object(run_module.sys, "platform", "linux"), self.assertRaisesRegex(ValueError,"must differ"):
                load_config(path)

    def test_review_api_settings_are_independent(self):
        bodies=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200);self.end_headers()
                self.wfile.write(json.dumps({"choices":[{"message":{"content":json.dumps(medium_review())},
                                                         "finish_reason":"stop"}]}).encode())
            def log_message(self,*args): pass
        with tempfile.TemporaryDirectory() as tmp, ThreadingHTTPServer(("127.0.0.1",0),Handler) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            cfg=fixture_config(Path(tmp))
            cfg.update(api_base="http://127.0.0.1:1/v1",api_model="unused-initial",
                       api_review_model="unused-shared",review_api_model="evidence-model",
                       review_api_base=f"http://127.0.0.1:{server.server_port}/v1")
            try:
                value,usage=ReviewAPIBackend(cfg).review(review_messages(ROW,SCHEMA))
                self.assertEqual(parse_json(value)["label"],"medium")
                self.assertEqual(bodies[0]["model"],"evidence-model")
                self.assertEqual(bodies[0]["max_tokens"],cfg["review_max_new_tokens"])
            finally:
                server.shutdown();thread.join()


class ParallelAPIReviewTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self, mode="normal", failure_code=402):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{**ROW, "sample_id": str(i), "answer": ROW["answer"] + f" 样本{i}"}
                    for i in range(30)]
            hashes = {r["sample_id"]: digest(blind(r)) for r in rows}
            cp = Checkpoint(root / "cp.jsonl", "test", hashes)
            for row in rows:
                cp.record(row["sample_id"], "initial", status="succeeded", attempt=0,
                          output={"label": "high"})
            pause = root / ".pause"
            state = {"mode": mode, "active": 0, "peak": 0, "requests": []}
            lock, release = threading.Lock(), threading.Event()
            barrier = threading.Barrier(10, action=pause.touch if mode == "pause" else None)

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args): pass
                def do_POST(self):
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    visible = visible_sample(body["messages"])
                    with lock:
                        index = len(state["requests"])
                        state["requests"].append(visible)
                        state["active"] += 1
                        state["peak"] = max(state["peak"], state["active"])
                    try:
                        if index < 10:
                            barrier.wait(timeout=8)
                        if state["mode"] == "unavailable" and index != 0:
                            if not release.wait(8):
                                raise RuntimeError("Mock failure was not propagated")
                        if index == 0 and state["mode"] in ("unavailable", "bad_request"):
                            self.send_response(failure_code if state["mode"] == "unavailable" else 400)
                            self.end_headers()
                            self.wfile.write(b'{"error":"controlled mock failure"}')
                        else:
                            value = medium_review()
                            value["reason"] = visible["answer"]
                            text = "{" if index == 0 and state["mode"] == "bad_json" else json.dumps(value)
                            payload = json.dumps({"choices": [{"message": {"content": text},
                                                               "finish_reason": "stop"}]}).encode()
                            self.send_response(200)
                            self.end_headers()
                            self.wfile.write(payload)
                    finally:
                        with lock:
                            state["active"] -= 1

            class Server(ThreadingHTTPServer):
                request_queue_size = 50

            class Client(ReviewAPIBackend):
                def send(self, *args):
                    try:
                        return super().send(*args)
                    except BackendError as exc:
                        if exc.service_unavailable:
                            release.set()  # Let other already-started mock calls finish.
                        raise

            with Server(("127.0.0.1", 0), Handler) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                cfg = fixture_config(root)
                cfg.update(review_api_workers=10, review_api_timeout=12,
                           review_api_base=f"http://127.0.0.1:{server.server_port}/v1")
                def forbidden(): self.fail("Frozen Qwen initials must be reused")
                def run():
                    with contextlib.redirect_stdout(io.StringIO()):
                        process_phased(rows, SCHEMA, forbidden, lambda: Client(cfg), cp, cfg,
                                       pause.exists, imported_only=True)
                try:
                    yield SimpleNamespace(root=root, rows=rows, hashes=hashes, cp=cp, pause=pause,
                                          state=state, cfg=cfg, run=run)
                finally:
                    release.set()
                    server.shutdown()
                    thread.join()

    def test_ten_real_http_requests_preserve_association_and_durable_resume(self):
        with self.fixture() as f:
            f.run()
            self.assertEqual(f.state["peak"], 10)
            self.assertEqual(len(f.state["requests"]), 30)
            loaded = Checkpoint(f.root / "cp.jsonl", "test", f.hashes)
            for row in f.rows:
                event = loaded.get(row["sample_id"], "review")
                self.assertEqual(event["status"], "succeeded")
                self.assertEqual(event["output"]["reason"], row["answer"])
            outputs = freeze_outputs(f.rows, loaded, "observe")
            self.assertEqual([r["sample_id"] for r in outputs], [str(i) for i in range(30)])
            f.run()
            self.assertEqual(len(f.state["requests"]), 30)
            self.assertTrue(all(set(r) == set(blind(ROW)) for r in f.state["requests"]))

    def test_pause_drains_ten_started_calls_and_resume_skips_them(self):
        with self.fixture("pause") as f:
            f.run()
            self.assertEqual(len(f.state["requests"]), 10)
            self.assertEqual(sum(e["stage"] == "review" and e["status"] == "running"
                                 for e in f.cp.events), 10)
            self.assertEqual(sum(f.cp.get(r["sample_id"], "review") is None for r in f.rows), 20)
            f.pause.unlink()
            f.state["mode"] = "normal"
            f.run()
            self.assertEqual(len(f.state["requests"]), 30)
            self.assertEqual(len({r["answer"] for r in f.state["requests"]}), 30)
            self.assertTrue(all(f.cp.get(r["sample_id"], "review")["status"] == "succeeded"
                                for r in f.rows))

    def test_service_failures_stop_dispatch_keep_pending_and_allow_resume(self):
        for code in (402, 429, 503):
            with self.subTest(code=code), self.fixture("unavailable", code) as f:
                f.run()
                self.assertEqual(len(f.state["requests"]), 10)
                events = [f.cp.get(r["sample_id"], "review") for r in f.rows]
                self.assertEqual(sum(e is None for e in events), 20)
                self.assertEqual(sum(bool(e and e["status"] == "terminal_failed") for e in events), 1)
                f.state["mode"] = "normal"
                f.run()
                self.assertEqual(len(f.state["requests"]), 30)
                self.assertEqual(len({r["answer"] for r in f.state["requests"]}), 30)

    def test_single_bad_request_or_non_json_does_not_stop_other_samples(self):
        for mode in ("bad_request", "bad_json"):
            with self.subTest(mode=mode), self.fixture(mode) as f:
                f.run()
                events = [f.cp.get(r["sample_id"], "review") for r in f.rows]
                self.assertEqual(len(f.state["requests"]), 30)
                self.assertEqual(sum(e["status"] == "succeeded" for e in events), 29)
                self.assertEqual(sum(e["status"] == "terminal_failed" for e in events), 1)

    def test_worker_validation_and_concurrency_independent_identity(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            cfg = fixture_config(root)
            path = root / "config.json"
            write_json(path, cfg)
            identity = engine_identity(load_config(path))
            self.assertEqual(identity, engine_identity(load_config(path, {"review_api_workers": 10})))
            for bad in (0, -1, True, "10", 1.5, None):
                cfg["review_api_workers"] = bad
                write_json(path, cfg)
                with self.subTest(value=bad), self.assertRaisesRegex(ValueError, "positive integer"):
                    load_config(path)
            cfg.pop("review_api_workers")
            write_json(path, cfg)
            self.assertEqual(load_config(path)["review_api_workers"], 5)
            original = file_hash
            with patch.object(run_module, "file_hash", side_effect=lambda p: original(p) +
                       ("changed" if Path(p).name == "classification_prompt_strict.txt" else "")):
                self.assertNotEqual(identity, engine_identity(load_config(path)))

    def test_current_rule_revision_preserves_archive_provenance_and_hides_lora_schema(self):
        source = read_json(ROOT / "review_prompt_source.json")
        self.assertEqual(file_hash(ROOT / "classification_prompt_strict.txt"), source["prompt_sha256"])
        self.assertEqual(file_hash(ROOT / "review_prompt.txt"), source["review_prompt_sha256"])
        for name in ("classification_prompt_strict.txt", "review_prompt.txt"):
            self.assertNotIn(b"\r", (ROOT / name).read_bytes())
        # The historical archive is not distributed or required on a new machine.
        self.assertNotIn("archive_path", source)
        self.assertEqual(source["archive_name"], "DeepSafe.zip")
        self.assertEqual(len(source["archive_prompt_sha256"]), 64)
        row = {**ROW, "initial_label": "secret-initial", "label": "secret-gold"}
        messages = review_messages(row, {"instructions": "unused-lora-schema-marker"})
        self.assertTrue(messages[0]["content"].startswith(
            (ROOT / "classification_prompt_strict.txt").read_text(encoding="utf-8-sig").strip()))
        for marker in ("secret-initial", "secret-gold", "unused-lora-schema-marker"):
            self.assertNotIn(marker, json.dumps(messages))


FAKE_VLLM = '''import json, os, time
from types import SimpleNamespace
class SamplingParams:
    def __init__(self, **kwargs): self.options=kwargs
class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['tokenize'] and kwargs['add_generation_prompt']
        return [1,2,3,4]
class LLM:
    def __init__(self, **kwargs):
        self.options=kwargs
        os.write(1, b'engine native stdout log\\n')
        print('engine Python stdout log', flush=True)
        assert kwargs['trust_remote_code'] is False
        assert kwargs['generation_config']=='vllm'
        if kwargs['model'].endswith('init_error'): raise RuntimeError('fake startup failure')
    def get_tokenizer(self): return Tokenizer()
    def generate(self, prompts, sampling, **kwargs):
        assert prompts==[{'prompt_token_ids':[1,2,3,4]}]
        assert sampling.options['temperature']==0
        if self.options['model'].endswith('slow'): time.sleep(5)
        with open(os.environ['FAKE_REVIEW_PAYLOAD'],encoding='utf-8') as f: text=f.read()
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text,token_ids=[5,6],finish_reason='stop')])]
'''


class VLLMProtocolTests(unittest.TestCase):
    def fixture(self,root,model_name="model"):
        cfg=fixture_config(root)
        (root/"vllm.py").write_text(FAKE_VLLM,encoding="utf-8")
        (root/model_name).mkdir()
        write_json(root/"review.json",medium_review())
        cfg.update(review_backend="vllm",review_local_model=str(root/model_name),
                   review_vllm_python=sys.executable,review_vllm_startup_timeout=15)
        env={"PYTHONPATH":str(root),"FAKE_REVIEW_PAYLOAD":str(root/"review.json")}
        return cfg,env

    def test_worker_ipc_logs_and_shared_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cfg,env=self.fixture(root)
            with patch.dict(os.environ,env):
                client=ReviewVLLMBackend(cfg,root/"worker.log")
                try:
                    text,usage=client.review(review_messages(ROW,SCHEMA))
                    result=decide({"label":"high"},parse_json(text),ROW,"experimental")
                    self.assertEqual(result["final_label"],"medium")
                    self.assertEqual(usage["prompt_tokens"],4)
                finally:
                    client.close();client.close()
                self.assertIsNotNone(client.process.poll())
                self.assertIn("native stdout",(root/"worker.log").read_text(encoding="utf-8"))

    def test_worker_overlong_input_refused_without_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,env=self.fixture(root);cfg["review_max_input_tokens"]=3
            with patch.dict(os.environ,env):
                client=ReviewVLLMBackend(cfg,root/"worker.log")
                try:
                    with self.assertRaisesRegex(BackendError,"no truncation"): client.review(review_messages(ROW,SCHEMA))
                finally: client.close()

    def test_worker_timeout_terminates_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,env=self.fixture(root,"slow");cfg["review_max_time_seconds"]=1
            with patch.dict(os.environ,env):
                client=ReviewVLLMBackend(cfg,root/"worker.log")
                try:
                    with self.assertRaisesRegex(BackendError,"timeout"): client.review(review_messages(ROW,SCHEMA))
                    self.assertFalse(client.healthy)
                    self.assertIsNotNone(client.process.poll())
                finally: client.close()

    def test_worker_startup_failure_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,env=self.fixture(root,"init_error")
            with patch.dict(os.environ,env):
                with self.assertRaisesRegex(BackendError,"fake startup failure"):
                    ReviewVLLMBackend(cfg,root/"worker.log")

    def test_local_model_with_cached_independent_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,env=self.fixture(root)
            cache=root/"cached_tokenizer";cache.mkdir()
            cfg["review_local_tokenizer"]="cached/tokenizer"
            (root/"huggingface_hub.py").write_text(
                "import os\ndef snapshot_download(repo, **kwargs):\n"
                "    assert repo == 'cached/tokenizer' and kwargs['local_files_only']\n"
                "    return os.environ['FAKE_TOKENIZER_CACHE']\n",encoding="utf-8")
            env["FAKE_TOKENIZER_CACHE"]=str(cache)
            with patch.dict(os.environ,env):
                client=ReviewVLLMBackend(cfg,root/"worker.log")
                try:
                    text,_=client.review(review_messages(ROW,SCHEMA))
                    self.assertEqual(parse_json(text)["label"],"medium")
                finally: client.close()


class IdentityAndMetricRegressionTests(unittest.TestCase):
    def local_model(self,path):
        path.mkdir()
        write_json(path/"config.json",{"model_type":"qwen3_5"})
        (path/"model.safetensors").write_bytes(b"weight-one")
        return path

    def test_base_and_review_weight_replacements_change_engine_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=fixture_config(root)
            cfg.update(base_model=str(self.local_model(root/"base")),review_backend="vllm",
                       review_local_model=str(self.local_model(root/"review")))
            original=engine_identity(cfg)
            (root/"base/model.safetensors").write_bytes(b"weight-two")
            new_base=engine_identity(cfg)
            self.assertNotEqual(original,new_base)
            with self.assertRaises(ValueError):
                Checkpoint(self.saved_checkpoint(root,original),new_base,{"one":digest(blind(ROW))})
            (root/"review/model.safetensors").write_bytes(b"weight-two")
            self.assertNotEqual(new_base,engine_identity(cfg))

    def saved_checkpoint(self,root,identity):
        path=root/"saved.jsonl"
        cp=Checkpoint(path,identity,{"one":digest(blind(ROW))})
        cp.record("one","initial",status="succeeded",attempt=1,output={"label":"high"})
        return path

    def test_model_config_tokenizer_and_indexed_shards_are_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.local_model(Path(tmp)/"model")
            (root/"tokenizer.json").write_text("one",encoding="utf-8")
            first=model_directory_identity(root)
            write_json(root/"config.json",{"model_type":"changed"})
            self.assertNotEqual(first,model_directory_identity(root))
            before=model_directory_identity(root)
            (root/"tokenizer.json").write_text("two",encoding="utf-8")
            self.assertNotEqual(before,model_directory_identity(root))
            (root/"shards").mkdir();(root/"shards/part.bin").write_bytes(b"one")
            write_json(root/"model.safetensors.index.json",{"weight_map":{"w":"shards/part.bin"}})
            before=model_directory_identity(root)
            self.assertIn("shards/part.bin",before)
            (root/"shards/part.bin").write_bytes(b"two")
            self.assertNotEqual(before,model_directory_identity(root))
            (root/"shards/part.bin").unlink()
            with self.assertRaisesRegex(ValueError,"Missing"): model_directory_identity(root)

    def test_changed_gate_rejects_old_policy_without_invalidating_model_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture_config(Path(tmp));engine=engine_identity(cfg);gate=gate_identity()
            policy={"schema_version":2,"engine_identity":engine,"gate_identity":gate,"validated":True,
                    "enabled_transitions":["high->medium"],"validation_protocol":"independent confirmation",
                    "confirmation_dataset_sha256":"actual-hash"}
            validate_policy(policy,engine)
            with patch.object(gate_module, "file_hash",return_value="changed-gate-source"):
                self.assertNotEqual(gate,gate_identity())
                with self.assertRaises(ValueError): validate_policy(policy,engine)
            self.assertEqual(engine,engine_identity(cfg))
            policy["schema_version"]=1
            with self.assertRaises(ValueError): validate_policy(policy,engine)

    def test_candidate_scores_exclude_pending_but_include_failed_reviews(self):
        rows=[{**ROW,"sample_id":str(i),"label":"medium"} for i in range(10)]
        outputs=[{"sample_id":str(i),"initial_status":"succeeded","initial_label":"high",
                  "review_status":"succeeded" if i==0 else "pending",
                  "candidate_label":"medium" if i==0 else None,
                  "experimental_label":"medium" if i==0 else "high","final_label":"high"} for i in range(10)]
        summary,_,_=report(rows,outputs)
        self.assertEqual(summary["scores"]["candidate_label"]["n"],1)
        self.assertEqual(summary["scores"]["candidate_label"]["acc"],1)
        self.assertEqual(summary["review_pending_samples"],9)
        self.assertEqual(summary["review_attempted_scores"]["initial_label"]["n"],1)
        self.assertEqual(summary["scores"]["initial_label"]["n"],10)
        outputs[1]["review_status"]="terminal_failed"
        summary,_,_=report(rows,outputs)
        self.assertEqual(summary["scores"]["candidate_label"]["n"],2)
        self.assertEqual(summary["scores"]["candidate_label"]["acc"],0.5)
        self.assertEqual(summary["review_evaluated_samples"],2)
        for row in outputs: row["review_status"]="pending"
        summary,_,_=report(rows,outputs)
        self.assertIsNone(summary["scores"]["candidate_label"]["acc"])


class CrossPlatformAndAPIFailureTests(unittest.TestCase):
    def saved_initials(self, root):
        rows = [{**ROW, "sample_id": str(i)} for i in range(3)]
        hashes = {r["sample_id"]: digest(blind(r)) for r in rows}
        cp = Checkpoint(root / "cp.jsonl", "id", hashes)
        for row in rows:
            cp.record(row["sample_id"], "initial", status="succeeded", attempt=0, output={"label": "high"})
        return rows, hashes, cp

    def test_service_errors_stop_batch_and_resume_only_unsent_samples(self):
        state = {"status": 401, "requests": []}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                state["requests"].append(body)
                self.send_response(state["status"])
                self.end_headers()
                if state["status"] != 200:
                    self.wfile.write(b'{"error":"service unavailable"}')
                else:
                    self.wfile.write(json.dumps({"choices": [{"message": {"content": json.dumps(stage_response(body["messages"], medium_review()))},
                        "finish_reason": "stop"}]}).encode())
        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for status in (401, 402, 403, 404, 429, 500, 503):
                    with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp); cfg = fixture_config(root)
                        cfg["review_api_base"] = f"http://127.0.0.1:{server.server_port}/v1"
                        rows, hashes, cp = self.saved_initials(root)
                        state.update(status=status, requests=[])
                        def forbidden(): self.fail("Saved initials must not be recomputed")
                        client = ReviewAPIBackend(cfg)
                        process_phased(rows, SCHEMA, forbidden, lambda: client, cp, cfg, lambda: False)
                        self.assertFalse(client.healthy)
                        self.assertEqual(len(state["requests"]), 1)
                        self.assertEqual(cp.get("0", "review")["status"], "terminal_failed")
                        self.assertIsNone(cp.get("1", "review"))
                        self.assertIsNone(cp.get("2", "review"))
                        state["status"] = 200
                        cp = Checkpoint(root / "cp.jsonl", "id", hashes)
                        process_phased(rows, SCHEMA, forbidden, lambda: ReviewAPIBackend(cfg), cp, cfg, lambda: False)
                        self.assertEqual(len(state["requests"]), 3)
                        self.assertIsNone(cp.get("0", "requirements"))
                        self.assertEqual(cp.get("0", "review")["attempt"], 1)
                        self.assertEqual(cp.get("0", "review")["status"], "terminal_failed")
                        self.assertTrue(all(cp.get(sid, "review")["status"] == "succeeded" for sid in ("1", "2")))
            finally:
                server.shutdown(); thread.join(timeout=3)

    def test_network_failure_stops_before_other_samples_are_attempted(self):
        for error in (URLError("connection refused"), TimeoutError("timeout")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); cfg = fixture_config(root)
                rows, _, cp = self.saved_initials(root)
                client = ReviewAPIBackend(cfg)
                with patch.object(backend_module, "urlopen", side_effect=error) as send:
                    process_phased(rows, SCHEMA, lambda: self.fail("initial reloaded"), lambda: client,
                                   cp, cfg, lambda: False)
                self.assertEqual(send.call_count, 1)
                self.assertFalse(client.healthy)
                self.assertIsNone(cp.get("1", "review"))

    def test_single_bad_request_does_not_stop_other_reviews(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                self.send_response(400 if len(requests) == 1 else 200); self.end_headers()
                if len(requests) == 1:
                    self.wfile.write(b'{"error":"sample rejected"}')
                else:
                    self.wfile.write(json.dumps({"choices": [{"message": {"content": json.dumps(stage_response(body["messages"], medium_review()))},
                        "finish_reason": "stop"}]}).encode())
        with tempfile.TemporaryDirectory() as tmp, ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            try:
                root = Path(tmp); cfg = fixture_config(root)
                cfg["review_api_base"] = f"http://127.0.0.1:{server.server_port}/v1"
                rows, _, cp = self.saved_initials(root)
                client = ReviewAPIBackend(cfg)
                process_phased(rows, SCHEMA, lambda: self.fail("initial reloaded"), lambda: client,
                               cp, cfg, lambda: False)
                self.assertTrue(client.healthy)
                self.assertEqual(len(requests), 3)
                self.assertEqual(cp.get("0", "review")["status"], "terminal_failed")
                self.assertEqual(cp.get("2", "review")["status"], "succeeded")
            finally:
                server.shutdown(); thread.join(timeout=3)

    def test_vllm_requires_linux_but_api_works_on_both_platforms(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp); cfg = fixture_config(root); path = root / "config.json"
            write_json(path, cfg)
            for platform in ("win32", "linux"):
                with patch.object(run_module.sys, "platform", platform):
                    self.assertEqual(load_config(path)["backend"], "transformers")
            write_json(path, {**cfg, "review_backend": "vllm", "review_local_model": "another-model"})
            with patch.object(run_module.sys, "platform", "win32"), self.assertRaisesRegex(ValueError, "requires Linux"):
                load_config(path)
            with patch.object(run_module.sys, "platform", "linux"):
                self.assertEqual(load_config(path)["review_backend"], "vllm")

    def test_stream_rejected_in_config_and_before_http_send(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp); cfg = fixture_config(root); path = root / "config.json"
            cfg["review_api_extra_body"] = {"stream": True}
            write_json(path, cfg)
            with self.assertRaisesRegex(ValueError, "Streaming"):
                load_config(path)
            with patch.object(backend_module, "urlopen") as send, self.assertRaisesRegex(ValueError, "Streaming"):
                ReviewAPIBackend(cfg).review(review_messages(ROW, SCHEMA))
            send.assert_not_called()
            cfg["review_api_extra_body"]["stream"] = False
            write_json(path, cfg)
            self.assertEqual(load_config(path)["review_api_extra_body"], {"stream": False})

    def test_inactive_backend_settings_and_packages_preserve_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg = fixture_config(root); original = engine_identity(cfg)
            changed = {**cfg, "review_local_model": "unused-review-model", "review_local_tokenizer": "unused",
                       "review_local_revision": "unused", "review_vllm_python": "unused-python",
                       "review_vllm_gpu_memory_utilization": 0.4, "api_model": "unused-initial",
                       "api_extra_body": {"unused": True}}
            self.assertEqual(original, engine_identity(changed))
            cp = Checkpoint(root / "cp.jsonl", original, {"one": digest(blind(ROW))})
            cp.record("one", "initial", status="succeeded", attempt=1, output={"label": "high"})
            resumed = Checkpoint(cp.path, engine_identity(changed), cp.input_hashes)
            self.assertEqual(resumed.get("one", "initial")["output"]["label"], "high")
            self.assertNotEqual(original, engine_identity({**cfg, "review_api_model": "changed-reviewer"}))
            def version(name):
                return "changed-unused-version" if name in ("vllm", "bitsandbytes") else real_version(name)
            real_version = __import__("importlib.metadata", fromlist=["version"]).version
            with patch.object(run_module.importlib.metadata, "version", side_effect=version):
                self.assertEqual(original, engine_identity(cfg))
            local = {**cfg, "review_backend": "vllm", "review_local_model": "another-model"}
            identity = engine_identity(local)
            self.assertEqual(identity, engine_identity({**local, "review_api_model": "unused",
                "review_api_base": "https://unused/v1", "review_api_extra_body": {"stream": True}}))
            self.assertNotEqual(identity, engine_identity({**local, "review_local_model": "changed-model"}))


class EnvironmentCheckTests(unittest.TestCase):
    def test_installation_check_allows_unconfigured_paths_without_claiming_inference_ready(self):
        check = check_env_module.main
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            write_json(path, {"quantization": "none", "adapter_path": "CHANGE_ME", "review_api_base": "CHANGE_ME"})
            output = io.StringIO()
            with patch.object(check_env_module, "load_config", side_effect=AssertionError("configuration must be deferred")), \
                 patch.object(check_env_module, "inspect_dependencies", return_value={"dependencies_ready": True}), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(check(["--config", str(path), "--dependencies-only"]), 0)
            result = json.loads(output.getvalue())
            self.assertFalse(result["configuration_validated"])
            self.assertFalse(result["ready_for_inference_attempt"])
            self.assertTrue(result["dependencies_ready"])

    def test_missing_dependencies_and_invalid_configuration_exit_nonzero(self):
        check = check_env_module.main
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp); cfg = fixture_config(root); path = root / "config.json"; write_json(path, cfg)
            with patch.object(check_env_module, "inspect_dependencies", return_value={"dependencies_ready": False}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check(["--config", str(path)]), 1)
            cfg["review_api_model"] = "CHANGE_ME"; write_json(path, cfg)
            result = subprocess.run([sys.executable, str(ROOT / "check_env.py"), "--config", str(path)],
                                    capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(json.loads(result.stdout)["ready_for_inference_attempt"])

    def test_dependency_probe_detects_missing_packages_and_unsupported_bf16(self):
        inspect_dependencies = check_env_module.inspect_dependencies
        torch = SimpleNamespace(version=SimpleNamespace(cuda="12.6"), cuda=SimpleNamespace(
            is_available=lambda: True, is_bf16_supported=lambda: True, get_device_name=lambda i: "mock GPU",
            get_device_properties=lambda i: SimpleNamespace(total_memory=8 * 1024**3)))
        modules = {"torch": torch, "transformers": SimpleNamespace(Qwen3_5ForConditionalGeneration=object),
                   "peft": SimpleNamespace(), "accelerate": SimpleNamespace()}
        cfg = {"quantization": "none", "device": "cuda", "dtype": "bfloat16"}
        with patch.dict(sys.modules, modules), patch.object(check_env_module.importlib.metadata, "version", return_value="mock"):
            self.assertTrue(inspect_dependencies(cfg)["dependencies_ready"])
            torch.cuda.is_bf16_supported = lambda: False
            self.assertFalse(inspect_dependencies(cfg)["dependencies_ready"])
            with patch.dict(sys.modules, {"transformers": None}):
                self.assertFalse(inspect_dependencies(cfg)["dependencies_ready"])


class ServerPackageTests(unittest.TestCase):
    def test_extracted_entrypoint_and_relative_model_data_paths_are_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            module = project / ROOT.name
            module.mkdir(parents=True)
            for name in package_module.FILES:
                (module / name).write_bytes((ROOT / name).read_bytes())
            adapter = project / "models/qwen35_4b_lora/best_adapter"
            adapter.mkdir(parents=True)
            for name in package_module.ADAPTER_FILES:
                (adapter / name).write_bytes(b"mock artifact; never loaded")
            for name in package_module.DATA_FILES:
                path = project / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            with patch.object(package_module, "ROOT", module), \
                 patch.object(package_module, "ADAPTER", adapter), \
                 contextlib.redirect_stdout(io.StringIO()):
                package_module.main([])
            destination = module / (module.name + "_server.zip")
            unpacked = Path(tmp) / "unpacked"
            with zipfile.ZipFile(destination) as archive:
                archive.extractall(unpacked)
                manifest = json.loads(archive.read(module.name + "/package_manifest.json"))
            for name, metadata in manifest.items():
                self.assertEqual(file_hash(unpacked / name), metadata["sha256"])
            extracted = unpacked / module.name
            cfg = read_json(extracted / "config.current.server.json")
            self.assertTrue((extracted / cfg["input"]).is_file())
            self.assertTrue((extracted / cfg["references"]).is_file())
            self.assertTrue((extracted / cfg["adapter_path"] / "adapter_config.json").is_file())
            result = subprocess.run([sys.executable, "-B", str(extracted / "run.py"), "--help"],
                                    capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))


class PortableConfigTests(unittest.TestCase):
    def test_json_model_directory_is_config_relative_even_with_cwd_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "config"
            directory.mkdir()
            cwd = root / "shell"
            cwd.mkdir()
            (cwd / "same_name").mkdir()
            cfg = fixture_config(directory)
            cfg["base_model"] = "same_name"
            write_json(Path(cfg["adapter_path"]) / "adapter_config.json",
                       {"base_model_name_or_path": "same_name"})
            path = directory / "config.json"
            write_json(path, cfg)
            previous = Path.cwd()
            try:
                os.chdir(cwd)
                with patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(load_config(path)["base_model"], "same_name")
                    model = directory / "same_name"
                    model.mkdir()
                    self.assertEqual(load_config(path)["base_model"], str(model.resolve()))
            finally:
                os.chdir(previous)

    def test_shell_model_directory_environment_is_cwd_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "config"
            directory.mkdir()
            model = root / "base"
            model.mkdir()
            path = directory / "config.json"
            write_json(path, fixture_config(directory))
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch.dict(os.environ, {"QWEN_BASE_PATH": "base"}, clear=True):
                    self.assertEqual(load_config(path)["base_model"], str(model.resolve()))
            finally:
                os.chdir(previous)

    def test_cli_model_directory_is_resolved_before_config_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base").mkdir()
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(run_module, "load_config", side_effect=RuntimeError("probe")) as load:
                    with self.assertRaisesRegex(RuntimeError, "probe"):
                        main(["--config", "config.json", "--base-model", "base"])
                    self.assertEqual(load.call_args.args[1]["base_model"], str((root / "base").resolve()))
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
