"""LoRA class logits and evidence review via Transformers, offline vLLM or HTTP."""
import json
import os
import subprocess
import sys
import threading
import queue
import gc
import signal
import time
import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if __package__:
    from .common import LABELS, ROOT, blind, read_json
    from .initial_protocol import label_token_ids, prepare_samples
    from .prompts import initial_messages
else:
    from common import LABELS, ROOT, blind, read_json
    from initial_protocol import label_token_ids, prepare_samples
    from prompts import initial_messages


class BackendError(RuntimeError):
    def __init__(self, message, retryable=False, service_unavailable=False):
        super().__init__(message)
        self.retryable = retryable
        self.service_unavailable = service_unavailable


class LocalBackend:
    def __init__(self, config):
        os.environ.setdefault("USE_HUB_KERNELS", "0")
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Local dependencies missing/incompatible. See setup_server.sh and README.md") from exc
        self.torch, self.config = torch, config
        if config["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA torch is unavailable. Run check_env.py first")
        if config["device"] == "cuda":
            # Match the supplied training runtime's old-driver CUDA 11.x SDPA fallback.
            try:
                driver=subprocess.run(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],
                    capture_output=True,text=True,timeout=10,check=True).stdout.strip().splitlines()[0]
                cuda_major=int(torch.version.cuda.split(".")[0])
                driver_major=int(driver.split(".")[0])
                minimum={11:450,12:525,13:580}.get(cuda_major)
                if minimum is None or driver_major < minimum:
                    raise RuntimeError(f"CUDA {torch.version.cuda} build is incompatible with driver {driver}; reuse the training CUDA torch environment")
                if cuda_major==11 and driver_major < 520:
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    if hasattr(torch.backends.cuda,"enable_cudnn_sdp"):
                        torch.backends.cuda.enable_cudnn_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)
            except (OSError,subprocess.SubprocessError):
                pass
        adapter = Path(config["adapter_path"])
        self.training = read_json(adapter / "training_config.json")
        self.export = read_json(adapter / "classification_schema.json")
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True)
        self.class_ids = label_token_ids(self.tokenizer)
        if self.class_ids != self.export["class_token_ids"] or self.export["enable_thinking"]:
            raise ValueError("Tokenizer/class-token IDs/thinking mode differ from training")
        dtype = getattr(torch, config["dtype"])
        if config["device"] == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Configured BF16 requires GPU BF16 support")
        kwargs = dict(dtype=dtype, attn_implementation=config["attention"],
                      trust_remote_code=False, cache_dir=config["cache_dir"],
                      local_files_only=config["local_files_only"], low_cpu_mem_usage=True)
        if not Path(config["base_model"]).is_dir():
            kwargs["revision"] = config["base_revision"]
        quant = config["quantization"]
        if quant == "4bit":
            if config["device"] != "cuda":
                raise ValueError("This 4-bit recipe requires CUDA; CPU uses quantization=none")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype)
        elif quant != "none":
            raise ValueError("quantization must be 4bit or none")
        kwargs["device_map"] = {"": 0} if config["device"] == "cuda" else {"": "cpu"}
        print("Loading pinned Qwen3.5 base and adapter; first use may download model weights.", flush=True)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(config["base_model"], **kwargs)
        text = getattr(base.config, "text_config", None)
        if (base.config.model_type != "qwen3_5" or text is None
                or text.num_hidden_layers != 32 or text.hidden_size != 2560):
            raise ValueError("Expected official Qwen3.5-4B architecture")
        self.model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False).eval()
        self.device = self.model.get_input_embeddings().weight.device
        self.model.config.use_cache = True
        self.model.config.text_config.use_cache = True

    def initial(self, row):
        # Strip labels before preparation as well as before prompt serialization.
        visible = {"sample_id": str(row["sample_id"]), **blind(row)}
        sample = prepare_samples([visible], self.tokenizer, self.training, self.export["question_schema"])[0]
        torch = self.torch
        selected = torch.tensor(self.class_ids, device=self.device)
        values = []
        with torch.inference_mode():
            for prompt in sample["windows"]:
                ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
                logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    use_cache=False, logits_to_keep=1).logits[0, -1].float()
                values.append(logits[selected].detach())
                del logits, ids
            scores = torch.stack(values).mean(0)
            if not torch.isfinite(scores).all():
                raise BackendError("Non-finite class logits")
            probabilities = scores.softmax(0).cpu().tolist()
            label = LABELS[int(scores.argmax().item())]
        return {"label": label, "probabilities": dict(zip(LABELS, probabilities)),
                "class_logits": scores.cpu().tolist(), "window_count": len(sample["windows"]),
                "window_spans": sample["window_spans"], "full_tokens": sample["full_tokens"]}, {
                    "initial_forward_passes": len(sample["windows"]),
                    "prompt_tokens": sum(len(w) for w in sample["windows"]), "completion_tokens": 0}

    def clear_after_error(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def close(self):
        self.model=None
        self.tokenizer=None
        gc.collect()
        self.clear_after_error()


class APIBackend:
    """Optional generation endpoint; initial API labels are not logit-protocol parity."""
    def __init__(self, config):
        self.config = config
        adapter = Path(config["adapter_path"])
        self.schema = read_json(adapter / "classification_schema.json")["question_schema"]
        self._http_local = threading.local()
        self._http_sessions = []
        self._http_lock = threading.Lock()

    def _pooled_post(self, url, body, headers, timeout):
        """One session per worker, no implicit retries or cross-thread Session use."""
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests is required for the pooled HTTP transport") from exc
        session = getattr(self._http_local, "session", None)
        if session is None:
            session = requests.Session()
            self._http_local.session = session
            with self._http_lock:
                self._http_sessions.append(session)
            if self.config.get("api_warmup_models", False):
                endpoint = url.rsplit("/chat/completions", 1)[0] + "/models"
                try:
                    with session.get(endpoint, headers=headers, timeout=20) as response:
                        if response.status_code >= 400:
                            raise BackendError("Connection warm-up HTTP " + str(response.status_code), service_unavailable=True)
                except requests.exceptions.RequestException as exc:
                    raise BackendError("Connection warm-up " + type(exc).__name__, service_unavailable=True) from exc
        try:
            with session.post(url, json=body, headers=headers, timeout=timeout) as response:
                if response.status_code >= 400:
                    detail = response.text
                    key = os.environ.get(self.config["api_key_env"], "")
                    if key:
                        detail = detail.replace(key, "[REDACTED]")
                    code = response.status_code
                    raise BackendError(f"HTTP {code}: {detail}", code == 429 or code >= 500,
                        service_unavailable=code in (401,402,403,404,407,408,429) or code >= 500)
                return response.json()
        except requests.exceptions.RequestException as exc:
            raise BackendError(str(exc), retryable=True, service_unavailable=True) from exc

    def _curl_post(self, url, body, headers, timeout):
        """Explicit system-TLS transport; credential and payload pass through stdin."""
        binary = shutil.which("curl.exe" if sys.platform == "win32" else "curl")
        if not binary:
            raise BackendError("curl transport requires a system curl executable", service_unavailable=True)
        def quote(value):
            return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r') + '"'
        lines = ["header = " + quote(k + ": " + v) for k, v in headers.items()]
        lines.append("data = " + quote(json.dumps(body, ensure_ascii=False)))
        key = os.environ.get(self.config["api_key_env"], "")
        try:
            response = subprocess.run([binary, "--config", "-", "--silent", "--show-error",
                "--max-time", str(timeout), "--url", url, "--write-out", "\nHTTP_STATUS:%{http_code}"],
                input="\n".join(lines), encoding="utf-8", capture_output=True, timeout=timeout + 10)
        except (OSError, subprocess.SubprocessError) as exc:
            raise BackendError("System HTTP transport failed: " + type(exc).__name__, service_unavailable=True) from exc
        if response.returncode:
            detail = response.stderr.replace(key, "[REDACTED]") if key else response.stderr
            raise BackendError("System HTTP transport: " + detail, retryable=True, service_unavailable=True)
        content, marker, status = response.stdout.rpartition("\nHTTP_STATUS:")
        if not marker or not status.isdigit():
            raise BackendError("System HTTP transport omitted response status", service_unavailable=True)
        code = int(status)
        if code >= 400:
            if key:
                content = content.replace(key, "[REDACTED]")
            raise BackendError(f"HTTP {code}: {content}", code == 429 or code >= 500,
                service_unavailable=code in (401,402,403,404,407,408,429) or code >= 500)
        return json.loads(content)

    def send(self, messages, model, max_tokens):
        c = self.config
        body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        extra = c["api_extra_body"]
        if set(extra) & set(body):
            raise ValueError("api_extra_body cannot override standard request fields")
        if extra.get("stream", False) is not False:
            raise ValueError("Streaming responses are unsupported; omit stream or set stream=false")
        body.update(extra)
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(c["api_key_env"], "")
        if key:
            headers["Authorization"] = "Bearer " + key
        req = Request(c["api_base"].rstrip("/") + "/chat/completions",
                      data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers=headers)
        try:
            if c.get("api_transport", "requests") == "curl":
                data = self._curl_post(req.full_url, body, headers, c["api_timeout"])
            elif c.get("api_transport", "requests") == "requests":
                data = self._pooled_post(req.full_url, body, headers, c["api_timeout"])
            else:
                with urlopen(req, timeout=c["api_timeout"]) as response:
                    data = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if key:
                detail = detail.replace(key, "[REDACTED]")
            unavailable = exc.code in (401, 402, 403, 404, 407, 408, 429) or exc.code >= 500
            raise BackendError(f"HTTP {exc.code}: {detail}", exc.code == 429 or exc.code >= 500,
                               service_unavailable=unavailable) from exc
        except (URLError, TimeoutError, ConnectionError) as exc:
            raise BackendError(str(exc), retryable=True, service_unavailable=True) from exc
        if not isinstance(data, dict) or data.get("error") or not isinstance(data.get("choices"), list) or not data["choices"]:
            detail = json.dumps(data.get("error", data) if isinstance(data, dict) else data, ensure_ascii=False)
            if key:
                detail = detail.replace(key, "[REDACTED]")
            raise BackendError("API did not return a valid completion: " + detail[:1000], service_unavailable=True)
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return choice["message"].get("content") or "", {**usage, "finish_reason": choice.get("finish_reason")}

    def initial(self, row):
        text, usage = self.send(initial_messages(row, self.schema), self.config["api_model"], 8)
        label = text.strip()
        if label not in LABELS or usage.get("finish_reason") != "stop":
            raise BackendError("API initial output is not one complete valid label: " + text)
        return {"label": label, "raw_text": text, "protocol": "api_label_generation"}, usage

    def review(self, messages):
        return self.send(messages, self.config.get("api_review_model", self.config["api_model"]),
                         self.config["review_max_new_tokens"])

    def clear_after_error(self):
        pass

    def close(self):
        for session in self._http_sessions:
            session.close()
        self._http_sessions.clear()


class ReviewAPIBackend(APIBackend):
    """Review-only endpoint settings are independent of initial classifier settings."""
    def __init__(self, config):
        settings=dict(config)
        for old,new in (("api_base","review_api_base"),("api_model","review_api_model"),
                        ("api_key_env","review_api_key_env"),("api_timeout","review_api_timeout"),
                        ("api_extra_body","review_api_extra_body"),("api_transport","review_api_transport")):
            settings[old]=config.get(new, "requests") if old == "api_transport" else config[new]
        settings.pop("api_review_model",None)
        settings["api_warmup_models"] = config.get("review_api_warmup_models", False)
        super().__init__(settings)
        self.healthy = True
        self.unavailable_reason = None

    def ensure_available(self):
        if not self.healthy:
            raise BackendError("Review API unavailable: " + str(self.unavailable_reason),
                               service_unavailable=True)

    def send(self, messages, model, max_tokens):
        self.ensure_available()
        try:
            return super().send(messages, model, max_tokens)
        except BackendError as exc:
            if exc.service_unavailable:
                # A sent sample consumes its attempt; later samples stay pending for a new invocation.
                self.healthy = False
                self.unavailable_reason = str(exc)
            raise


class ReviewVLLMBackend:
    """A persistent offline-vLLM subprocess: no HTTP service, independently selectable Python."""
    def __init__(self,config,log_path):
        self.config=config
        self.responses=queue.Queue()
        self.healthy=True
        Path(log_path).parent.mkdir(parents=True,exist_ok=True)
        self.log=Path(log_path).open("a",encoding="utf-8")
        python=config.get("review_vllm_python") or sys.executable
        env=dict(os.environ)
        env["PYTHONUNBUFFERED"]="1"
        env["PYTHONIOENCODING"]="utf-8"
        env["PYTHONUTF8"]="1"
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD","spawn")
        print("Loading offline vLLM review worker; logs: "+str(log_path),flush=True)
        try:
            self.process=subprocess.Popen([python,str(ROOT/"vllm_worker.py")],stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,stderr=self.log,text=True,encoding="utf-8",bufsize=1,env=env,
                start_new_session=os.name!="nt")
        except BaseException:
            self.log.close()
            raise
        self.reader=threading.Thread(target=self._read,daemon=True)
        self.reader.start()
        try:
            self._send({"command":"initialize","config":config})
            ready=self._receive(config["review_vllm_startup_timeout"])
            if ready.get("status") != "ready":
                raise BackendError("vLLM startup failed: "+str(ready.get("error")))
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    self.responses.put(json.loads(line))
                except ValueError:
                    self.responses.put({"status":"error","error":"Invalid worker protocol; see vllm_worker.log"})
        except (OSError,ValueError):
            pass
        finally:
            self.responses.put({"status":"closed","error":"vLLM worker exited; see vllm_worker.log"})

    def ensure_available(self):
        if not self.healthy or self.process.poll() is not None:
            self.healthy=False
            raise BackendError("vLLM worker unavailable")

    def _send(self,value):
        self.ensure_available()
        self.process.stdin.write(json.dumps(value,ensure_ascii=False)+"\n")
        self.process.stdin.flush()

    def _receive(self,timeout):
        try:
            result=self.responses.get(timeout=timeout)
        except queue.Empty as exc:
            self.close()
            raise BackendError("vLLM worker timeout; terminated owned process; initial label retained") from exc
        if result.get("status")=="closed":
            self.healthy=False
            raise BackendError(result["error"])
        return result

    def review(self,messages):
        try:
            self._send({"command":"review","messages":messages})
            result=self._receive(self.config["review_max_time_seconds"])
            if result.get("status")!="ok":
                raise BackendError(result.get("error","vLLM review failed"))
            return result["text"],result["usage"]
        except (BrokenPipeError,OSError) as exc:
            self.healthy=False
            raise BackendError("vLLM worker connection closed") from exc

    def clear_after_error(self):
        pass

    def close(self):
        self.healthy=False
        process=getattr(self,"process",None)
        def terminate_group(sig):
            try:
                if os.name!="nt": os.killpg(process.pid,sig)
                elif process.poll() is None:
                    process.kill() if sig==getattr(signal,"SIGKILL",9) else process.terminate()
            except ProcessLookupError:
                pass
        if process is not None:
            terminate_group(signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                terminate_group(getattr(signal,"SIGKILL",9))
                process.wait(timeout=10)
        if process is not None:
            if hasattr(self,"reader"): self.reader.join(timeout=1)
            if process.stdin: process.stdin.close()
            if process.stdout: process.stdout.close()
        if hasattr(self,"log") and not self.log.closed:
            self.log.close()
