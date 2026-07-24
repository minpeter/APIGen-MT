"""Compatibility facade for the LLM client implementations."""

from pydantic import BaseModel

if __package__:
    from .llm_local_openai_client import LocalOpenAILLMClient
    from .llm_remote_client import LLMClient
    from .llm_token_accounting import (
        TIKTOKEN_AVAILABLE,
        TokenCounter,
        TokenUsage,
    )
else:
    from llm_local_openai_client import LocalOpenAILLMClient
    from llm_remote_client import LLMClient
    from llm_token_accounting import (
        TIKTOKEN_AVAILABLE,
        TokenCounter,
        TokenUsage,
    )

__all__ = [
    "TIKTOKEN_AVAILABLE",
    "LLMClient",
    "LocalOpenAILLMClient",
    "TokenCounter",
    "TokenUsage",
]


if __name__ == "__main__":
    llm_client = LLMClient()

    class schema(BaseModel):
        name: str
        age: int

    response = llm_client.json_output(
        prompt="Extract the name and age from this text: John Doe, 30 years old.",
        reasoning=True,
    )
    print(response)
