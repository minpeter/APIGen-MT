"""Local token counting and accumulated usage accounting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tiktoken import Encoding

try:
    import tiktoken as _tiktoken
except ImportError:
    _tiktoken = None

TIKTOKEN_AVAILABLE = _tiktoken is not None


class TokenUsage:
    """Tracks token usage for LLM calls."""

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0

    def add(
        self,
        prompt: int = 0,
        completion: int = 0,
        total: int = 0,
    ) -> None:
        """Add token counts from a single LLM call."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class TokenCounter:
    """Count tokens locally with the ``cl100k_base`` encoding by default."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        """Initialize the counter with a tiktoken encoding."""
        if _tiktoken is None:
            raise ImportError(
                "tiktoken is required for local token counting. Install it with: "
                + "pip install tiktoken"
            )
        self.encoding: Encoding = _tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in a single text string."""
        if not text:
            return 0
        return len(self.encoding.encode(text, disallowed_special=()))

    def count_chat_tokens(
        self,
        messages: Sequence[Mapping[str, object]],
    ) -> int:
        """Count framing, role, and content tokens for a chat conversation."""
        if not messages:
            return 0

        total_tokens = 0
        tokens_per_message = 3
        for message in messages:
            total_tokens += tokens_per_message
            content = message.get("content", "")
            if isinstance(content, str) and content:
                total_tokens += self.count_tokens(content)

            role = message.get("role", "")
            if isinstance(role, str):
                total_tokens += self.count_tokens(role)

        total_tokens += 3
        return total_tokens

    def count_prompt_tokens(
        self,
        messages: Sequence[Mapping[str, object]],
    ) -> int:
        """Count tokens in messages sent to the API."""
        return self.count_chat_tokens(messages)

    def count_completion_tokens(self, response_text: str) -> int:
        """Count tokens in a completion response."""
        return self.count_tokens(response_text)
