"""Friendli-compatible remote LLM client."""

from __future__ import annotations

import importlib
import os
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

if __package__:
    from .llm_debug_logger import log_llm_call
    from .llm_request_helpers import (
        LLMMessage,
        append_reasoning_message,
        build_json_messages,
        build_response_format,
        extract_completion_text,
        extract_message_text,
        parse_json_output,
        parse_reasoning_response,
        request_json,
    )
else:
    from llm_debug_logger import log_llm_call
    from llm_request_helpers import (
        LLMMessage,
        append_reasoning_message,
        build_json_messages,
        build_response_format,
        extract_completion_text,
        extract_message_text,
        parse_json_output,
        parse_reasoning_response,
        request_json,
    )


class ChatTokenizer(Protocol):
    """Tokenizer operation required for assistant-prefill completions."""

    def apply_chat_template(
        self,
        conversation: list[LLMMessage],
        *,
        tokenize: bool,
        continue_final_message: bool,
    ) -> str: ...


@runtime_checkable
class TokenizerFactory(Protocol):
    def from_pretrained(
        self,
        pretrained_model_name_or_path: str,
        **kwargs: object,
    ) -> ChatTokenizer: ...


class AttributeGetter(Protocol):
    def __call__(self, instance: object, name: str, /) -> object: ...


_GET_ATTRIBUTE: AttributeGetter = getattr


def load_tokenizer(
    tokenizer_id: str,
    options: dict[str, object],
) -> ChatTokenizer:
    """Load transformers only when a client requests a tokenizer."""
    module = importlib.import_module("transformers")
    factory = _GET_ATTRIBUTE(module, "AutoTokenizer")
    if not isinstance(factory, TokenizerFactory):
        raise TypeError("transformers.AutoTokenizer is not a tokenizer factory")
    return factory.from_pretrained(tokenizer_id, **options)


class LLMClient:
    """Client for Friendli's chat and text-completion endpoints."""

    def __init__(
        self,
        url: str = "https://api.friendli.ai/serverless/v1",
        api_key: str | None = None,
        api_model: str = "deepseek-r1",
        hf_tokenizer_id: str | None = "deepseek-ai/deepseek-v3",
        debug_mode: bool = False,
    ) -> None:
        self.url: str = url
        token = api_key or os.getenv("FRIENDLI_TOKEN")
        self.headers: dict[str, str] = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self.api_model: str = api_model
        self.debug_mode: bool = debug_mode
        self.tokenizer: ChatTokenizer | None = (
            load_tokenizer(
                hf_tokenizer_id,
                {"token": os.environ.get("HF_TOKEN"), "legacy": False},
            )
            if hf_tokenizer_id is not None
            else None
        )

    def get_token_usage(self) -> dict[str, int]:
        """Get token usage statistics. Override in tracking subclasses."""
        return {
            "total_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def __apply_chat_template(
        self,
        messages: list[LLMMessage],
        prefill: bool = True,
    ) -> str:
        if self.tokenizer is None:
            raise RuntimeError("assistant prefill requires a tokenizer")
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=prefill,
        )

    def chat(
        self,
        messages: list[LLMMessage],
        kwargs: dict[str, object],
    ) -> tuple[str, str]:
        if self.debug_mode:
            log_llm_call(
                debug_mode=True,
                call_type="chat",
                model=self.api_model,
                endpoint=f"{self.url}/chat/completions",
                messages=messages,
                kwargs=kwargs,
            )

        if messages[-1]["role"] == "assistant":
            prompt = self.__apply_chat_template(messages, prefill=True)
            return self.completions(prompt=prompt, kwargs=kwargs), ""

        payload: dict[str, object] = {
            "model": self.api_model,
            "messages": messages,
            **kwargs,
        }
        response_object = request_json(
            f"{self.url}/chat/completions",
            self.headers,
            payload,
        )
        response = extract_message_text(response_object, "content")

        if self.debug_mode:
            log_llm_call(
                debug_mode=True,
                call_type="chat_response",
                model=self.api_model,
                endpoint=f"{self.url}/chat/completions",
                messages=messages,
                kwargs=kwargs,
                response=response,
            )

        return parse_reasoning_response(response)

    def generate(
        self,
        messages: list[LLMMessage],
        **kwargs: object,
    ) -> str:
        """Generate a response and return only its visible text."""
        response, _ = self.chat(messages, kwargs)
        return response

    def completions(
        self,
        prompt: str,
        kwargs: dict[str, object],
    ) -> str:
        payload: dict[str, object] = {
            "model": self.api_model,
            "prompt": prompt,
            **kwargs,
        }
        response = request_json(f"{self.url}/completions", self.headers, payload)
        return extract_completion_text(response)

    def json_output(
        self,
        prompt: str,
        system_prompt: str | None = None,
        schema: BaseModel | None = None,
        reasoning: bool = True,
    ) -> tuple[object, str]:
        messages = build_json_messages(prompt, system_prompt, schema)
        if reasoning:
            reasoning_messages = [
                *messages,
                {"role": "assistant", "content": "<think>\n"},
            ]
            reasoning_str, _ = self.chat(
                messages=reasoning_messages,
                kwargs={"stop": ["</think>"]},
            )
        else:
            reasoning_str = ""

        final_messages = append_reasoning_message(messages, reasoning_str)
        response_format = build_response_format(schema)
        raw_json, _ = self.chat(
            messages=final_messages,
            kwargs={"response_format": response_format},
        )
        parsed = parse_json_output(raw_json)

        if self.debug_mode:
            log_llm_call(
                debug_mode=True,
                call_type="json_output",
                model=self.api_model,
                endpoint=f"{self.url}/chat/completions",
                messages=final_messages,
                kwargs={"response_format": response_format},
                response=raw_json,
                reasoning=reasoning_str,
                parsed_output=parsed,
            )

        return parsed, reasoning_str
