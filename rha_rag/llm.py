"""LLM configuration and ChatDeepSeekFixed for DeepSeek V4 thinking mode."""

import os
import json
from typing import Any


class ModelConfig:
    def __init__(self, name: str, api_key: str, model_name: str, base_url: str | None):
        self.name = name
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url


models = {
    "ocr": ModelConfig("ocr", "ZAI_API_KEY", "glm-ocr", None),
    "embed": ModelConfig(
        "embed",
        "QWEN_API_KEY",
        "text-embedding-v4",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "llm": ModelConfig(
        "llm",
        "OPENAI_API_KEY",
        "deepseek-v4-pro",
        "https://api.deepseek.com",
    ),
}


def create_llms():
    """Create and return (response_model, grader_model).

    Returns (None, None) if OPENAI_API_KEY is not set.
    """
    from langchain_deepseek import ChatDeepSeek
    from langchain_core.language_models import LanguageModelInput

    if not os.environ.get(models["llm"].api_key):
        return None, None

    class ChatDeepSeekFixed(ChatDeepSeek):
        """ChatDeepSeek with fixes for V4 thinking mode.

        Three patches:
        1. Preserve reasoning_content across tool-call round-trips
        2. Serialize list-type tool/assistant message content
        3. Demote structured-output tool_choice dict to "auto"
        """

        def _get_request_payload(
            self,
            input_: LanguageModelInput,
            *,
            stop: list[str] | None = None,
            **kwargs: Any,
        ) -> dict:
            payload = super(ChatDeepSeek, self)._get_request_payload(
                input_, stop=stop, **kwargs
            )
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
            if isinstance(payload.get("tool_choice"), dict):
                payload["tool_choice"] = "auto"
            return payload

    llm_kwargs = dict(
        model=models["llm"].model_name,
        temperature=0,
        api_key=os.environ[models["llm"].api_key],
        api_base=models["llm"].base_url,
    )

    return ChatDeepSeekFixed(**llm_kwargs), ChatDeepSeekFixed(**llm_kwargs)


# For backward compatibility with old code
ChatDeepSeekFixed = None  # Will be set when create_llms() is called
