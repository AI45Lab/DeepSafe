import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from uni_eval.models.base import BaseModel

logger = logging.getLogger(__name__)

Message = Dict[str, Any]
UserContent = Union[str, List[Dict[str, Any]]]

def _flatten_messages_to_text(messages: List[Message]) -> str:
    """
    Fallback prompt builder for models that don't accept OpenAI-style `messages`.
    This is intentionally simple and model-agnostic.
    """
    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "").strip() or "user"
        content = m.get("content")
        if isinstance(content, list):
                                                                   
            text_parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text_parts.append(str(c.get("text", "")))
            content_str = "\n".join([t for t in text_parts if t])
        else:
            content_str = "" if content is None else str(content)
        parts.append(f"{role.upper()}:\n{content_str}".rstrip())
    parts.append("ASSISTANT:\n")
    return "\n\n".join(parts)

class MultiTurnChatMixin:
    """
    A reusable executor for multi-turn conversations.

    - Prefer sending OpenAI-style `messages` to models (works with APIModel in chat mode).
    - If the model can't handle `messages`, fallback to flattening the conversation into text.
    """

    def _run_multi_turn_chat(
        self,
        model: BaseModel,
        user_turns: Sequence[UserContent],
        *,
        system_prompt: str = "",
        turn_template: str = "{prompt}",
        gen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Message], List[str]]:
        gen_kwargs = gen_kwargs or {}

        messages: List[Message] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        assistant_outputs: List[str] = []

        use_messages_api = True

        for turn in user_turns:
            if isinstance(turn, str):
                user_content: UserContent = turn_template.format(prompt=turn)
            else:

                user_content = turn

            if use_messages_api:
                to_send = messages + [{"role": "user", "content": user_content}]
                try:
                    resp = model.generate([to_send], **gen_kwargs)[0]
                except Exception as e:
                    logger.warning(
                        "Model doesn't accept `messages` input; falling back to text prompt. Error: %s",
                        e,
                    )
                    use_messages_api = False
                                            
                    flat_prompt = _flatten_messages_to_text(to_send)
                    resp = model.generate([flat_prompt], **gen_kwargs)[0]
            else:
                to_send = messages + [{"role": "user", "content": user_content}]
                flat_prompt = _flatten_messages_to_text(to_send)
                resp = model.generate([flat_prompt], **gen_kwargs)[0]

            resp = "" if resp is None else str(resp)

            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": resp})
            assistant_outputs.append(resp)

        return messages, assistant_outputs

