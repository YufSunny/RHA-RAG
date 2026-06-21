"""LLM configuration and ChatDeepSeekFixed for DeepSeek V4 thinking mode."""

import os
import json
from typing import Any
from config import (
    LLM_MODEL, LLM_BASE_URL, EMBED_MODEL, EMBED_BASE_URL, OCR_MODEL,
    LLM_THINKING, LLM_REASONING_EFFORT,
)


class ModelConfig:
    def __init__(self, name: str, api_key: str, model_name: str, base_url: str | None):
        self.name = name
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url


models = {
    "ocr":   ModelConfig("ocr",   "ZAI_API_KEY",  OCR_MODEL,   None),
    "embed": ModelConfig("embed", "QWEN_API_KEY", EMBED_MODEL, EMBED_BASE_URL),
    "llm":   ModelConfig("llm",   "OPENAI_API_KEY", LLM_MODEL,  LLM_BASE_URL),
}


def create_llms():
    """Create and return (response_model, grader_model).

    Returns (None, None) if OPENAI_API_KEY is not set.
    """
    from langchain_deepseek import ChatDeepSeek
    from langchain_core.language_models import LanguageModelInput

    if not os.environ.get(models["llm"].api_key):
        return None, None

    _is_deepseek = "deepseek" in (models["llm"].base_url or "").lower()
    _use_thinking = _is_deepseek and LLM_THINKING

    class _Patched(ChatDeepSeek):
        """Patches the request payload for DeepSeek thinking mode when enabled."""

        def _get_request_payload(
            self,
            input_: LanguageModelInput,
            *,
            stop: list[str] | None = None,
            **kwargs: Any,
        ) -> dict:
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)

            if _use_thinking:
                # Inject the thinking-mode toggle and effort level.
                payload["thinking"] = {"type": "enabled"}
                payload["reasoning_effort"] = LLM_REASONING_EFFORT
                # temperature / top_p / presence_penalty / frequency_penalty
                # are not supported in thinking mode — drop them.
                for _k in (
                    "temperature", "top_p", "presence_penalty", "frequency_penalty",
                ):
                    payload.pop(_k, None)

                # Preserve reasoning_content across tool-call round-trips.
                input_messages = self._convert_input(input_).to_messages() or []
                for idx, message in enumerate(payload["messages"]):
                    rc = input_messages[idx].additional_kwargs.get("reasoning_content")
                    if rc and message["role"] == "assistant":
                        message["reasoning_content"] = rc
                    if message["role"] == "tool" and isinstance(message["content"], list):
                        message["content"] = json.dumps(message["content"])
                    elif message["role"] == "assistant" and isinstance(
                        message["content"], list
                    ):
                        text_parts = [
                            b.get("text", "")
                            for b in message["content"]
                            if isinstance(b, dict) and b.get("type") == "text"
                        ]
                        message["content"] = "".join(text_parts) if text_parts else ""
                # Demote structured tool_choice (rejected by thinking mode).
                if isinstance(payload.get("tool_choice"), dict):
                    payload["tool_choice"] = "auto"

            return payload

    llm_kwargs: dict = dict(
        model=models["llm"].model_name,
        api_key=os.environ[models["llm"].api_key],
        api_base=models["llm"].base_url,
        max_retries=5,
        timeout=300,
    )
    if not _use_thinking:
        llm_kwargs["temperature"] = 0

    llm_cls = _Patched if _is_deepseek else ChatDeepSeek
    return llm_cls(**llm_kwargs), llm_cls(**llm_kwargs)


# For backward compatibility with old code
ChatDeepSeekFixed = None  # Will be set when create_llms() is called
