"""System/user message support for the new profile; legacy client stays unchanged."""

from .client import OpenAICompatibleRequester


class ReviewRequester(OpenAICompatibleRequester):
    def __call__(self, messages):
        if (not isinstance(messages, list) or not messages or
                any(not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in ("system", "user")
                    or not isinstance(message["content"], str) for message in messages)):
            raise ValueError("Expected explicit system/user prompt messages")
        options = {"extra_body": {"thinking": {"type": "disabled"}}} if (
            self.backend == "api" and self.thinking_mode == "disabled") else {}
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0,
            max_tokens=self.max_tokens, timeout=self.timeout, **options)
        usage = response.usage
        return response.choices[0].message.content, {
            "prompt": getattr(usage, "prompt_tokens", 0) or 0,
            "completion": getattr(usage, "completion_tokens", 0) or 0}
