"""Detailed opt-in logging for LLM prompts, responses, and parsing."""

from __future__ import annotations

import json
from datetime import UTC, datetime


def _truncated(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n... (truncated, {len(value)} total chars)"


def log_llm_call(
    debug_mode: bool,
    call_type: str,
    model: str,
    endpoint: str,
    messages: list[dict[str, str]],
    kwargs: dict[str, object],
    response: str | None = None,
    reasoning: str | None = None,
    parsed_output: object | None = None,
) -> None:
    """Log one LLM call when debug mode is enabled."""
    if not debug_mode:
        return

    print("\n" + "=" * 80)
    print(f"📤 LLM API CALL - {call_type}()")
    print("=" * 80)
    print(f"Timestamp: {datetime.now(UTC).isoformat()}")
    print(f"Model: {model}")
    print(f"Endpoint: {endpoint}")

    print(f"\n📝 Messages ({len(messages)} total):")
    for index, message in enumerate(messages):
        role = message.get("role", "unknown")
        content = _truncated(message.get("content", ""), 500)
        print(f"\n[{index}] Role: {role}")
        print("    Content:")
        for line in content.split("\n"):
            print(f"      {line}")

    if kwargs:
        print("\n📋 kwargs:")
        print(json.dumps(kwargs, indent=2, ensure_ascii=False))

    if response is not None:
        print("\n" + "=" * 80)
        print("📥 LLM RAW RESPONSE")
        print("=" * 80)
        for line in _truncated(response, 1000).split("\n"):
            print(f"  {line}")

    if reasoning:
        print("\n" + "=" * 80)
        print("🤔 EXTRACTED REASONING")
        print("=" * 80)
        for line in reasoning.split("\n"):
            print(f"  {line}")

    if parsed_output is not None:
        print("\n" + "=" * 80)
        print("✅ PARSED OUTPUT")
        print("=" * 80)
        print(json.dumps(parsed_output, indent=2, ensure_ascii=False))

    print("\n" + "=" * 80 + "\n")
