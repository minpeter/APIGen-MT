"""User-turn simulation and cross-turn placeholder resolution."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from src.multi_turn_models import DialogBlueprint, MultiTurnConversation
    from src.multi_turn_protocols import GeneratorMixinBase
    from src.step_by_step_models import QueryGenerationResult
else:
    from multi_turn_protocols import GeneratorMixinBase
    from step_by_step_models import QueryGenerationResult

from multi_turn_protocols import (
    is_object_dict,
    string_list,
    string_value,
    tool_call_view,
)


class UserSimulationMixin(GeneratorMixinBase):
    """Use blueprint queries as simulated user turns."""

    @override
    def _generate_turn_query(
        self,
        blueprint: DialogBlueprint,
        conversation: MultiTurnConversation,
        turn_index: int,
    ) -> QueryGenerationResult | None:
        """Use the blueprint's pre-written user query for this turn."""
        turn_spec = (
            blueprint.turns[turn_index]
            if turn_index < len(blueprint.turns)
            else {}
        )
        user_query = self._resolve_turn_placeholders(
            string_value(turn_spec, "user_query"),
            turn_index,
            conversation,
        )
        expected_tools = string_list(turn_spec, "expected_tools")

        max_tools = max(3, self.target_num_actions + 2)
        if not user_query or not 1 <= len(expected_tools) <= max_tools:
            print(
                f"  ✗ Turn {turn_index + 1}: Blueprint has invalid query "
                + f"({len(expected_tools)} tools, need 1-{max_tools})"
            )
            return None

        invalid = [
            tool
            for tool in expected_tools
            if not self.tool_manager.tool_exists(tool)
        ]
        if invalid:
            print(
                f"  ✗ Turn {turn_index + 1}: Invalid tools in blueprint: "
                + f"{invalid}"
            )
            return None

        print(f"  ✓ Using blueprint query for turn {turn_index + 1}")
        print(f"   Query: {user_query[:80]}...")
        print(f"   Tools: {expected_tools}")
        return QueryGenerationResult(
            query=user_query,
            intent="",
            expected_tools=expected_tools,
        )

    def _resolve_turn_placeholders(
        self,
        query: str,
        turn_index: int,
        conversation: MultiTurnConversation,
    ) -> str:
        """Resolve prior-turn tool-output placeholders in a query."""
        pattern = re.compile(r"\{\{TURN(\d+)\.(\w+)\.(\w+)\}\}")

        def replacer(match: re.Match[str]) -> str:
            referenced_turn = int(match.group(1))
            tool_name = match.group(2)
            output_key = match.group(3)

            if referenced_turn > turn_index:
                return match.group(0)

            referenced_turn_index = referenced_turn - 1
            if referenced_turn_index >= len(conversation.turns):
                return match.group(0)

            prior_turn = conversation.turns[referenced_turn_index]
            for step in prior_turn.steps:
                for raw_tool_call in step.tool_calls:
                    tool_call = tool_call_view(raw_tool_call)
                    output = tool_call.output
                    if tool_call.tool_name != tool_name or not is_object_dict(output):
                        continue
                    if output_key in output:
                        return str(output[output_key])
                    for key, value in output.items():
                        if not isinstance(key, str):
                            continue
                        if (
                            output_key.lower() in key.lower()
                            or key.lower() in output_key.lower()
                        ):
                            return str(value)
                    if len(output) == 1:
                        return str(next(iter(output.values())))
            return match.group(0)

        resolved = pattern.sub(replacer, query)
        if resolved != query:
            print(
                f"   Resolved placeholders: {query[:60]}... -> "
                + f"{resolved[:60]}..."
            )
        return resolved

    def _format_conversation_history(
        self,
        conversation: MultiTurnConversation,
    ) -> str:
        """Format completed turns as readable history for the LLM."""
        if not conversation.turns:
            return ""

        lines: list[str] = []
        for turn in conversation.turns:
            lines.append(f"--- Turn {turn.turn_number} ---")
            lines.append(f"User: {turn.user_query}")
            for step in turn.steps:
                for raw_tool_call in step.tool_calls:
                    tool_call = tool_call_view(raw_tool_call)
                    output = tool_call.output
                    output_preview = str(output)[:100] if output else ""
                    arguments = json.dumps(tool_call.arguments, default=str)[:200]
                    lines.append(
                        f"  → {tool_call.tool_name}({arguments}) -> "
                        + output_preview
                    )
            lines.append(f"Assistant: {turn.assistant_response}")

        return "\n".join(lines)

    @staticmethod
    def _assign_tools_to_turns(
        blueprint: DialogBlueprint,
        all_tool_names: list[str],
    ) -> dict[int, list[str]]:
        """Distribute the blueprint's declared tools across turns."""
        del all_tool_names
        return {
            turn_index: string_list(turn_spec, "expected_tools")
            for turn_index, turn_spec in enumerate(blueprint.turns)
        }
