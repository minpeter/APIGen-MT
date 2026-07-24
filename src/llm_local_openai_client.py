"""Local and OpenAI-compatible LLM client with token tracking."""

from __future__ import annotations

import os
from typing import override

from pydantic import BaseModel

if __package__:
    from .llm_local_retry import request_chat_with_retries
    from .llm_remote_client import ChatTokenizer, LLMClient, load_tokenizer
    from .llm_request_helpers import (
        LLMMessage,
        append_reasoning_message,
        build_json_messages,
        build_response_format,
        parse_json_output,
        parse_reasoning_response,
    )
    from .llm_token_accounting import TokenCounter, TokenUsage
else:
    from llm_local_retry import request_chat_with_retries
    from llm_remote_client import ChatTokenizer, LLMClient, load_tokenizer
    from llm_request_helpers import (
        LLMMessage,
        append_reasoning_message,
        build_json_messages,
        build_response_format,
        parse_json_output,
        parse_reasoning_response,
    )
    from llm_token_accounting import TokenCounter, TokenUsage


def _load_local_tokenizer(tokenizer_id: str) -> ChatTokenizer:
    """Load transformers only when an explicit tokenizer is requested."""
    options: dict[str, object] = {"legacy": False}
    token = os.environ.get("HF_TOKEN")
    if token:
        options["token"] = token
    return load_tokenizer(tokenizer_id, options)


class LocalOpenAILLMClient(LLMClient):
    """Client for local LLM servers and OpenAI-compatible API endpoints."""

    def __init__(
        self,
        url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        api_model: str = "local-model",
        hf_tokenizer_id: str | None = None,
        debug_mode: bool = False,
    ) -> None:
        super().__init__(
            url=url,
            api_key=api_key,
            api_model=api_model,
            hf_tokenizer_id=None,
            debug_mode=debug_mode,
        )
        self.url: str = url
        self.headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.api_model: str = api_model
        self.debug_mode: bool = debug_mode
        self.total_calls: int = 0
        self.token_usage: TokenUsage = TokenUsage()
        self.token_counter: TokenCounter = TokenCounter()
        self.tokenizer: ChatTokenizer | None = (
            _load_local_tokenizer(hf_tokenizer_id) if hf_tokenizer_id else None
        )

    @override
    def get_token_usage(self) -> dict[str, int]:
        """Get accumulated token usage statistics."""
        return {"total_calls": self.total_calls, **self.token_usage.to_dict()}

    def reset_token_usage(self) -> None:
        """Reset token usage counters."""
        self.total_calls = 0
        self.token_usage = TokenUsage()

    @override
    def chat(
        self,
        messages: list[LLMMessage],
        kwargs: dict[str, object],
    ) -> tuple[str, str]:
        prompt_tokens = self.token_counter.count_prompt_tokens(messages)

        if messages[-1]["role"] == "assistant" and self.tokenizer is not None:
            response = super().chat(messages, kwargs)
            completion_tokens = self.token_counter.count_completion_tokens(response[0])
            self.token_usage.add(
                prompt=prompt_tokens,
                completion=completion_tokens,
                total=prompt_tokens + completion_tokens,
            )
            self.total_calls += 1
            return response

        response_text = request_chat_with_retries(
            self.url,
            self.headers,
            messages,
            {"model": self.api_model, **kwargs},
        )
        completion_tokens = self.token_counter.count_completion_tokens(response_text)
        self.token_usage.add(
            prompt=prompt_tokens,
            completion=completion_tokens,
            total=prompt_tokens + completion_tokens,
        )
        self.total_calls += 1
        return parse_reasoning_response(response_text)

    @override
    def json_output(
        self,
        prompt: str,
        system_prompt: str | None = None,
        schema: BaseModel | None = None,
        reasoning: bool = True,
    ) -> tuple[object, str]:
        messages = build_json_messages(prompt, system_prompt, schema)
        if reasoning:
            reasoning_str, _ = self.chat(
                messages=[
                    *messages,
                    {"role": "assistant", "content": "<think>\n"},
                ],
                kwargs={"stop": ["</think>"]},
            )
        else:
            reasoning_str = ""

        raw_json, _ = self.chat(
            messages=append_reasoning_message(messages, reasoning_str),
            kwargs={"response_format": build_response_format(schema, strict=True)},
        )
        return parse_json_output(raw_json), reasoning_str
