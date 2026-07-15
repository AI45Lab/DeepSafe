from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import httpx
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from .utils import strip_json_fence

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    @abstractmethod
    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class JudgeLLM(BaseLLM):
    """OpenAI-compatible API client used as the judge LLM.

    Aligned with the simple client style in `api_refer.py`: no httpx proxy
    handling, no vllm-specific `chat_template_kwargs`, no model-list discovery.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        *,
        temperature: float = 0.0,
        timeout: float = 300.0,
        max_tokens: int = 4096,
        use_env_proxy: bool = True,
        max_retries: int = 3,
        retry_wait_seconds: float = 5.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        http_client = None if use_env_proxy else httpx.Client(trust_env=False, timeout=timeout)
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            http_client=http_client,
        )
        self.last_error: str | None = None
        self._call = retry(
            wait=wait_fixed(retry_wait_seconds),
            stop=stop_after_attempt(max(1, max_retries)),
            reraise=True,
        )(self._call_impl)

    def _call_impl(self, system_prompt: str, user_prompt: str) -> str:
        try:
            self.last_error = None
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("LLM call failed: %s", self.last_error)
            raise

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        raw = self._call(system_prompt, user_prompt)
        text = strip_json_fence(raw).strip()
        if not text:
            raise RuntimeError(f"Model returned empty content. last_error={self.last_error}")
        return text


MLLM = JudgeLLM
