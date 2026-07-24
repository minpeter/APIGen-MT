"""LLM-assisted blueprint capability validation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, override

from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from src.multi_turn_protocols import ApiState, BlueprintTurn, GeneratorMixinBase
else:
    from multi_turn_protocols import GeneratorMixinBase

from multi_turn_protocols import is_object_dict, string_list, string_value


class _CapabilityVerdict(BaseModel):
    is_valid: bool = False
    issues: list[str] = Field(default_factory=list)


class CapabilityValidationMixin(GeneratorMixinBase):
    """Check that selected tools can fulfill each blueprint query."""

    @override
    def _verify_blueprint_capabilities(
        self,
        turns: list[BlueprintTurn],
        focus_category: str | None = None,
        initial_api_state: ApiState | None = None,
    ) -> tuple[bool, list[str]]:
        """Verify that each turn's query intent matches tool capabilities."""
        del focus_category
        if not turns:
            return False, ["No turns provided"]

        tool_capabilities: list[tuple[str, str]] = []
        recorded_tools: set[str] = set()
        for turn in turns:
            for tool_name in string_list(turn, "expected_tools"):
                if tool_name in recorded_tools:
                    continue
                recorded_tools.add(tool_name)
                try:
                    schema = self.tool_manager.get_tool_schema(tool_name)
                    description = string_value(
                        schema, "description", "No description"
                    )[:300]
                    parameters = schema.get("parameters")
                    properties: dict[object, object] = {}
                    if is_object_dict(parameters):
                        raw_properties = parameters.get("properties")
                        if is_object_dict(raw_properties):
                            properties = raw_properties
                    parameter_info = ""
                    for parameter in properties.values():
                        if not is_object_dict(parameter):
                            continue
                        parameter_type = parameter.get("type", "unknown")
                        enum_values = parameter.get("enum")
                        if enum_values:
                            parameter_info += f" (enum: {enum_values})"
                        else:
                            parameter_info += f" ({parameter_type})"
                    tool_capabilities.append(
                        (tool_name, f"{description}{parameter_info}")
                    )
                except ValueError:
                    tool_capabilities.append(
                        (tool_name, "Tool description unavailable")
                    )

        tool_capability_text = "\n".join(
            f"- {name}: {description}"
            for name, description in tool_capabilities
        )

        state_summary = ""
        if initial_api_state:
            for class_key, state in initial_api_state.items():
                state_json = json.dumps(state, indent=2, default=str)
                state_summary += f"\n{class_key}:\n{state_json[:5000]}"

        prompt = f"""You are verifying that a dialog blueprint's user queries can be fulfilled by the selected tools.

Check each turn: does the user_query intent match what the selected tool can actually do?

=== TOOL CAPABILITIES ===
{tool_capability_text}

=== CURRENT API STATE ===
This shows what entities (files, IDs, etc.) exist. Queries should reference only these entities.
{state_summary if state_summary else "No specific state provided."}

=== BLUEPRINT TURNS ===
{json.dumps(turns, indent=2, default=str)}

=== VERIFICATION TASK ===
For each turn, verify:
1. Does the user_query ask for something the selected tool can actually do?
2. Does the query phrasing match tool capabilities? (e.g., "search all files" can't be done by a single-file grep)
3. CRITICAL: Check if query asks for something NOT in the tool's parameter enums. For example, if displayCarStatus only has options [fuel, battery, doors, climate, headlights, parkingBrake, brakePadle, engine], then asking for "cruise control speed" is INVALID.
4. Are entity names (files, IDs, etc.) consistent with the API state?
5. If multiple tools are listed, is that realistic for one turn?

IMPORTANT: Reject queries that ask for things outside the tool's enum values or capabilities, even if the tool description mentions a general category that might seem related.

Respond ONLY with valid JSON:
{{"is_valid": true/false, "issues": ["Turn N: issue description", ...]}}

If ALL turns are achievable with their selected tools, set is_valid to true with empty issues."""

        try:
            response = self._safe_llm_generate(
                [{"role": "user", "content": prompt}]
            )
            response_text = response.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                response_text = response_text[start:end]

            result = _CapabilityVerdict.model_validate_json(response_text)
            return result.is_valid, result.issues
        except (RuntimeError, TypeError, ValueError, ValidationError) as error:
            return False, [f"Capability check error: {str(error)[:100]}"]
