import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Union
import json
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

try:
    from openai import OpenAI
except ImportError:                    
    OpenAI = None

from uni_eval.models.base import BaseModel
from uni_eval.registry import MODELS

logger = logging.getLogger(__name__)

def _parse_drop_params_env() -> Set[str]:
    """
    Comma-separated list of request params to omit for APIModel calls.
    Example:
      export API_MODEL_DROP_PARAMS="temperature,top_p"
    """
    raw = str(os.getenv("API_MODEL_DROP_PARAMS", "")).strip()
    if not raw:
        return set()
    parts = [p.strip() for p in raw.split(",")]
    return {p for p in parts if p}

def _should_drop_sampling_params_by_model_name(model_name: str) -> bool:
    """
    Some OpenAI-compatible gateways (esp. multi-provider relays) have provider-specific
    constraints on sampling params. We keep a conservative default:
    - If model name contains claude/gemini: omit temperature/top_p/top_k.
    - If model name contains gpt: keep temperature (so temperature=0 can take effect),
      but still omit top_p/top_k by default for better relay compatibility.
    """
    name = (model_name or "").lower()
    return any(k in name for k in ("claude"))

def _default_drop_params_for_model_name(model_name: str) -> Set[str]:
    """
    Provider-specific default drop policy.
    - claude/gemini: drop temperature/top_p/top_k (many relays reject these or have constraints).
    - gpt: allow temperature (temperature=0 is commonly used and supported), but drop top_p/top_k
      to avoid relay-specific incompatibilities like "temperature and top_p cannot both be specified".
    """
    name = (model_name or "").lower()
    if "claude" in name:
        return {"temperature", "top_p", "top_k"}
    if "gpt" in name:
        return {"top_p", "top_k"}
    return set()

@MODELS.register_module()
class APIModel(BaseModel):
    """
    OpenAI-compatible API model wrapper (Copycat Version).
    
    Logic:
    1. If http_proxy is in config -> set os.environ (mimic 'export http_proxy=...').
    2. Init OpenAI client with ONLY api_key and base_url.
    3. Trust the environment and the library to do the rest.
    """

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str = "EMPTY",
        http_proxy: Optional[str] = None,
        timeout: Optional[float] = None, 
        max_retries: int = 10,                                             
        concurrency: int = 10,
        mode: str = "chat",
        strip_reasoning: bool = False,
                                                          
        preflight: bool = False, 
        preflight_strict: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if OpenAI is None:
            raise ImportError("Please install openai package: pip install openai")

        if api_key == "ENV":
            api_key = os.getenv("OPENAI_API_KEY", "EMPTY")

        self.model_name = model_name
        self.api_base = api_base
        self.concurrency = concurrency
        self.mode = mode
        self.strip_reasoning = strip_reasoning
        self.default_gen_kwargs = kwargs

        self.debug_errors = str(os.getenv("API_MODEL_DEBUG", "")).strip().lower() in ("1", "true", "yes", "y")

        if http_proxy:
            logger.info(f"APIModel: Setting os.environ['http_proxy'] to {http_proxy}")
            os.environ["http_proxy"] = http_proxy
            os.environ["https_proxy"] = http_proxy
            os.environ["HTTP_PROXY"] = http_proxy
            os.environ["HTTPS_PROXY"] = http_proxy

        logger.info(f"APIModel init: model={model_name} base_url={api_base} max_retries={max_retries}")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=max_retries
        
        )

    def _call_api(self, prompt: Union[str, List[Dict]], **kwargs) -> str:
        gen_kwargs = {**self.default_gen_kwargs, **kwargs}
        drop_params = _parse_drop_params_env()

        temperature = gen_kwargs.get("temperature", 0.0)
        max_tokens = gen_kwargs.get("max_tokens", 1024)
        top_p = gen_kwargs.get("top_p", 0.95)

        drop_params = set(drop_params)
        drop_params.update(_default_drop_params_for_model_name(self.model_name))

        extra_params = {
            k: v
            for k, v in gen_kwargs.items()
            if k not in ["temperature", "max_tokens", "top_p"] and k not in drop_params
        }

        def _is_local_vllm(base: str) -> bool:
            try:
                u = urlparse(base)
                return u.hostname in ("127.0.0.1", "localhost")
            except Exception:
                return False

        def _sanitize_chat_messages(msgs: List[Dict]) -> List[Dict]:
            """
            Some OpenAI-compatible gateways reject empty assistant/user messages, e.g.:
              "message at position N with role 'assistant' must not be empty"
            This can happen in multi-turn evals if a prior model call failed and returned "".
            We defensively drop any messages whose content is empty/None after stripping.
            """
            out: List[Dict] = []
            for m in msgs or []:
                if not isinstance(m, dict):
                    continue
                role = m.get("role", None)
                content = m.get("content", None)
                                                                                       
                if isinstance(content, str):
                    if not content.strip():
                        continue
                elif content is None:
                    continue
                if role is None:
                    continue
                out.append(m)
            return out

        def _extract_text_from_chat_response(obj: Dict) -> str:
            """
            vLLM/OpenAI-compatible chat response extraction.
            Handles both:
            - message.content as string
            - message.content as list of {"type":"text","text":...} parts
            - message.content missing/None (fallbacks)
            """
            choices = obj.get("choices") or []
            if not choices:
                return ""
            msg = (choices[0] or {}).get("message") or {}
            content = msg.get("content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                        parts.append(p["text"])
                return "".join(parts)
                                                           
            for k in ("text", "reasoning", "reasoning_content"):
                v = msg.get(k, None)
                if isinstance(v, str) and v.strip():
                    return v
            return ""

        def _extract_text_from_completion_response(obj: Dict) -> str:
            choices = obj.get("choices") or []
            if not choices:
                return ""
            c0 = choices[0] or {}
            txt = c0.get("text", "")
            return txt if isinstance(txt, str) else ""

        def _call_via_http(prompt_obj: Union[str, List[Dict]], is_chat: bool) -> str:
            """
            HTTP fallback for localhost vLLM when openai-python strict parsing fails.
            """
            base = (self.api_base or "").rstrip("/") + "/"
            endpoint = "chat/completions" if is_chat else "completions"
            url = urljoin(base, endpoint)

            if is_chat:
                if isinstance(prompt_obj, str):
                    messages = [{"role": "user", "content": prompt_obj}]
                else:
                    messages = prompt_obj
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                }
            else:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt_obj,
                    "max_tokens": max_tokens,
                }

            if "temperature" not in drop_params:
                payload["temperature"] = temperature
            if "top_p" not in drop_params:
                payload["top_p"] = top_p
            if extra_params:
                payload["extra_body"] = extra_params

            req = Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=gen_kwargs.get("timeout", None) or 60) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(raw)
            if self.debug_errors:
                                                             
                logger.error(
                    "APIModel localhost HTTP fallback raw response (truncated): %s",
                    raw[:2000].replace("\n", "\\n"),
                )
            text_out = _extract_text_from_chat_response(obj) if is_chat else _extract_text_from_completion_response(obj)
            if self.debug_errors:
                logger.error(
                    "APIModel localhost HTTP fallback extracted text (truncated): %s",
                    (text_out or "")[:300].replace("\n", "\\n"),
                )
            return text_out

        try:
            if self.mode == "chat":
                if isinstance(prompt, str):
                    messages = [{"role": "user", "content": prompt}]
                else:
                    messages = prompt
                messages = _sanitize_chat_messages(messages)

                req_kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "extra_body": extra_params,
                }
                if "temperature" not in drop_params:
                    req_kwargs["temperature"] = temperature
                if "top_p" not in drop_params:
                    req_kwargs["top_p"] = top_p

                response = self.client.chat.completions.create(**req_kwargs)
                msg = response.choices[0].message

                text = (getattr(msg, "content", None) or "") if msg is not None else ""
                if not (text or "").strip():
                    rc = getattr(msg, "reasoning_content", None) if msg is not None else None
                    if isinstance(rc, str) and rc.strip():
                        text = rc

                if _is_local_vllm(self.api_base) and not (text or "").strip():
                    try:
                        text = _call_via_http(prompt, is_chat=True)
                    except Exception:
                        pass
            else:
                req_kwargs = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "extra_body": extra_params,
                }
                if "temperature" not in drop_params:
                    req_kwargs["temperature"] = temperature
                if "top_p" not in drop_params:
                    req_kwargs["top_p"] = top_p

                response = self.client.completions.create(**req_kwargs)
                text = response.choices[0].text or ""
                if _is_local_vllm(self.api_base) and not (text or "").strip():
                    try:
                        text = _call_via_http(prompt, is_chat=False)
                    except Exception:
                        pass
        except Exception as e:

            if _is_local_vllm(self.api_base):
                try:
                    text = _call_via_http(prompt, is_chat=(self.mode == "chat"))
                    if self.strip_reasoning and text:
                        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
                    return text
                except Exception:
                                                             
                    pass

            err_str = str(e)
            err_type = type(e).__name__
            status_code = getattr(e, "status_code", None)
            request_id = getattr(e, "request_id", None)
                                                                                           
            body = getattr(e, "body", None)
            response = getattr(e, "response", None)

            if "content_filter" in err_str or "content management policy" in err_str:
                logger.warning(f"API Triggered Content Filter: {err_str[:200]}...")
                return "[CONTENT_FILTER]"

            logger.error(
                "API call failed: type=%s status=%s request_id=%s model=%s base_url=%s err=%r",
                err_type,
                status_code,
                request_id,
                self.model_name,
                self.api_base,
                err_str,
            )
                                                                     
            if self.debug_errors:
                logger.exception("API call exception (debug)")
                if body is not None:
                    logger.error("API error body (debug): %r", body)
                if response is not None:
                                                                           
                    logger.error("API error response (debug): %r", response)
            return ""

        if self.strip_reasoning and text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        return text

    def generate(self, prompts: List[Union[str, List[Dict]]], **kwargs) -> List[str]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            return list(executor.map(lambda p: self._call_api(p, **kwargs), prompts))
