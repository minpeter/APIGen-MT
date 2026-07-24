"""Runtime configuration for the OpenAI-compatible generation entry points."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_API_BASE_ENV = "OPENAI_API_BASE"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "minimax/minimax-m2.7"


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved settings used by an OpenAI-compatible API client."""

    api_key: str
    api_base: str
    model: str

    @classmethod
    def from_environment(
        cls,
        *,
        model: str = DEFAULT_MODEL,
        environ: Mapping[str, str] | None = None,
    ) -> RuntimeConfig:
        """Resolve credentials and defaults without making a network request."""
        environment = os.environ if environ is None else environ
        api_key = environment.get(OPENAI_API_KEY_ENV)
        if not api_key:
            raise ValueError(f"{OPENAI_API_KEY_ENV} is not set")
        return cls(
            api_key=api_key,
            api_base=environment.get(OPENAI_API_BASE_ENV, DEFAULT_API_BASE),
            model=model,
        )
