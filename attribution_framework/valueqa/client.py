"""OpenAI-compatible API and vLLM requests with bounded retries."""

import time
from collections import Counter

from .prompts import parse_json


class OpenAICompatibleRequester:
    def __init__(self, backend, api_base, model, api_key, max_tokens, timeout, thinking_mode="auto"):
        from openai import OpenAI
        if backend == "api" and not api_key:
            raise ValueError("API mode requires OPENAI_API_KEY")
        self.client = OpenAI(base_url=api_base, api_key=api_key or "local-vllm", max_retries=0)
        self.model, self.max_tokens, self.timeout = model, max_tokens, timeout
        self.thinking_mode = thinking_mode
        self.backend = backend

    def __call__(self, content):
        options = {"extra_body": {"thinking": {"type": "disabled"}}} if (
            self.backend == "api" and self.thinking_mode == "disabled") else {}
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": content}],
            temperature=0, max_tokens=self.max_tokens, timeout=self.timeout, **options)
        usage = response.usage
        return response.choices[0].message.content, {
            "prompt": getattr(usage, "prompt_tokens", 0) or 0,
            "completion": getattr(usage, "completion_tokens", 0) or 0}


def ask(requester, content, attempts, should_stop=None):
    calls, usage, error, terminal = 0, Counter(), None, False
    for index in range(attempts):
        if should_stop is not None and should_stop():
            break
        calls += 1
        try:
            result = requester(content)
            text, tokens = result if isinstance(result, tuple) else (result, {})
            usage.update(tokens)
            parsed = parse_json(text)
            if parsed is not None:
                return {"output": parsed, "calls": calls, "usage": dict(usage), "error": None,
                        "terminal": False}
            error = "non-JSON output"
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
            status = getattr(exc, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                # GLM's 1301 content filter rejects this particular input.
                # Authentication, billing, and configuration errors may be fixed later.
                terminal = status == 400 and "1301" in str(exc) and "contentFilter" in str(exc)
                break
        if index + 1 < attempts and not (should_stop is not None and should_stop()):
            time.sleep(min(2 ** index, 4))
    return {"output": None, "calls": calls, "usage": dict(usage), "error": error,
            "terminal": terminal}


