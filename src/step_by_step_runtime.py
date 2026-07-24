"""Shared token accounting and tool-schema formatting helpers."""

import json
from typing import Protocol, override

from step_by_step_models import TokenUsageStats
from step_by_step_protocols import StepByStepMixinBase


def _string_field(item: dict[str, object], key: str, default: str) -> str:
    value = item.get(key, default)
    return value if isinstance(value, str) else default


class RuntimeMixin(StepByStepMixinBase, Protocol):
    _accumulated_prompt_tokens: int
    _accumulated_completion_tokens: int
    _accumulated_total_tokens: int
    _accumulated_llm_calls: int
    _initial_token_usage: dict[str, int] | None

    def _reset_token_tracking(self) -> None:
        """Reset token tracking for a new datapoint."""
        self._accumulated_prompt_tokens = 0
        self._accumulated_completion_tokens = 0
        self._accumulated_total_tokens = 0
        self._accumulated_llm_calls = 0
        self._initial_token_usage = None

    def _capture_initial_usage(self) -> None:
        """Capture initial token usage before starting a datapoint."""
        self._initial_token_usage = self.llm.get_token_usage()

    @override
    def _update_token_usage(self) -> None:
        """Update accumulated token usage from LLM client."""
        if self._initial_token_usage is None:
            return

        current_usage = self.llm.get_token_usage()
        initial = self._initial_token_usage
        self._accumulated_prompt_tokens = (
            current_usage["prompt_tokens"] - initial["prompt_tokens"]
        )
        self._accumulated_completion_tokens = (
            current_usage["completion_tokens"] - initial["completion_tokens"]
        )
        self._accumulated_total_tokens = (
            current_usage["total_tokens"] - initial["total_tokens"]
        )
        self._accumulated_llm_calls = (
            current_usage["total_calls"] - initial["total_calls"]
        )

    @override
    def _get_token_stats(self) -> TokenUsageStats:
        """Get current token usage stats."""
        return TokenUsageStats(
            prompt_tokens=self._accumulated_prompt_tokens,
            completion_tokens=self._accumulated_completion_tokens,
            total_tokens=self._accumulated_total_tokens,
            total_llm_calls=self._accumulated_llm_calls,
        )

    @override
    def _get_tool_schemas_str(self, tools_subset: list[str] | None = None) -> str:
        schemas = self.tool_manager.get_tools_json_schema()
        if tools_subset:
            schemas = [
                schema
                for schema in schemas
                if _string_field(schema, "name", "") in tools_subset
            ]
        return json.dumps(schemas, indent=2, ensure_ascii=False)

    def _get_tools_with_descriptions_str(
        self,
        category: str | None = None,
        compact: bool = False,
    ) -> str:
        """Format tool names and descriptions, grouped by category."""
        tools = self.tool_manager.get_tools_json_schema()
        if category:
            tools = [
                tool
                for tool in tools
                if _string_field(tool, "category", "Unknown") == category
            ]

        if compact:
            return "\n".join(
                f'{_string_field(tool, "name", "")}: {_string_field(tool, "description", "")[:80]}'
                for tool in tools
            )

        tools_by_category: dict[str, list[dict[str, object]]] = {}
        for tool in tools:
            tool_category = _string_field(tool, "category", "Unknown")
            tools_by_category.setdefault(tool_category, []).append(tool)

        result: list[str] = []
        for tool_category, category_tools in sorted(tools_by_category.items()):
            result.append(f"\n{tool_category}:")
            result.extend(
                f' - {_string_field(tool, "name", "")}: {_string_field(tool, "description", "No description available.")}'
                for tool in category_tools
            )
        return "\n".join(result)
