"""Tool selection, consistency checking, and immediate execution helpers."""

import json
from typing import Protocol, override

from step_by_step_models import (
    ObjectMap,
    StepSelectionResult,
    TrajectoryStep,
    ValidationResponse,
)
from step_by_step_protocols import (
    StepByStepMixinBase,
    is_object_list,
    is_object_map,
)

_AUTH_STATUS_FIELDS = {
    "comment": "comment_status",
    "create_ticket": "status",
    "edit_ticket": "status",
    "follow_user": "follow_status",
    "mention": "mention_status",
    "post_tweet": "tweet_status",
    "resolve_ticket": "status",
    "close_ticket": "status",
    "retweet": "retweet_status",
    "unfollow_user": "follow_status",
}


def _is_tool_specific_error(
    tool_name: str,
    key: str,
    value: object,
) -> bool:
    if key == "error" and tool_name in {"get_flight_cost", "si_unit_conversion"}:
        return bool(value)
    if tool_name in _AUTH_STATUS_FIELDS and key == _AUTH_STATUS_FIELDS[tool_name]:
        return isinstance(value, str) and (
            "not authenticated" in value.lower()
            or (
                tool_name in {"close_ticket", "edit_ticket", "resolve_ticket"}
                and "not found" in value.lower()
            )
        )
    if tool_name == "get_ticket" and key == "status":
        return isinstance(value, str) and "not found" in value.lower()
    if tool_name in {"authenticate_twitter", "message_login"}:
        return key in {"authentication_status", "login_status"} and value is False
    if tool_name in {"ticket_login", "verify_traveler_information"}:
        return key in {"success", "verification_status"} and value is False
    if tool_name == "authenticate_travel":
        return (key == "success" and value is False) or (
            key == "access_token" and value == ""
        )
    if tool_name == "get_flight_cost" and key == "travel_cost_list":
        return is_object_list(value) and not value
    if tool_name == "book_flight" and key in {
        "booking_status",
        "booking_confirmation",
    }:
        return isinstance(value, str) and any(
            word in value.lower() for word in ("fail", "error")
        )
    return False


class ToolStepsMixin(StepByStepMixinBase, Protocol):
    def _generate_next_step(
        self,
        query: str,
        trajectory: list[TrajectoryStep],
        execution_context: ObjectMap,
        expected_tools: list[str],
        step_num: int = 1,
    ) -> StepSelectionResult:
        del step_num
        trajectory_lines: list[str] = []
        tools_used: set[str] = set()
        for index, step in enumerate(trajectory, 1):
            trajectory_lines.append(f"\nStep {index}:")
            for tool_call in step.tool_calls:
                tools_used.add(tool_call.tool_name)
                line = (
                    f"\n - {tool_call.tool_name}"
                    f"({json.dumps(tool_call.arguments)})"
                )
                if tool_call.output:
                    line += f" -> {json.dumps(tool_call.output, default=str)[:200]}"
                trajectory_lines.append(line)

        tools_remaining = [
            tool for tool in expected_tools if tool not in tools_used
        ]
        if not tools_remaining:
            return StepSelectionResult(
                tool_name="__FINAL_RESPONSE__",
                arguments={},
                reasoning="All expected tools have been used.",
            )

        description_lines: list[str] = []
        for tool_name in tools_remaining:
            try:
                schema = self.tool_manager.get_tool_schema(tool_name)
            except ValueError:
                schema = {}
            description = schema.get(
                "description",
                "(tool for completing the task)",
            )
            description_text = (
                description
                if isinstance(description, str)
                else "(tool for completing the task)"
            )
            description_lines.append(
                f" - {tool_name}: {description_text[:150]}"
            )

        prompt = f"""You are selecting the next tool to call based on the conversation context.

=== USER QUERY ===
{query}

=== CURRENT TRAJECTORY ===
{''.join(trajectory_lines)}

=== EXPECTED TOOLS REMAINING ===
{chr(10).join(description_lines)}

=== EXECUTION CONTEXT (previous tool outputs) ===
{json.dumps(execution_context, indent=2, default=str)[:1000]}

=== YOUR TASK ===
Select the NEXT tool to call from the EXPECTED TOOLS REMAINING list above.

CRITICAL:
- You MUST select a tool name EXACTLY as shown in EXPECTED TOOLS REMAINING
- The tool must logically follow from the current trajectory and context
- Use values from Execution Context when available

Respond ONLY with valid JSON:
{{
    "tool_name": "exact_name_from_expected_tools_list",
    "arguments": {{"arg1": "value1"}},
    "reasoning": "brief explanation"
}}"""
        try:
            response_text = self._safe_llm_generate(
                [{"role": "user", "content": prompt}]
            ).strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            else:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]
            return StepSelectionResult.model_validate_json(response_text)
        except ValueError as exc:
            print(f"    JSON decode error in step generation: {exc}")
            return StepSelectionResult(
                tool_name="__ERROR__",
                arguments={},
                reasoning=f"JSON error: {exc}",
            )

    @override
    def _simulate_tool_execution(
        self,
        tool_name: str,
        arguments: object,
        execution_context: ObjectMap,
    ) -> object:
        if is_object_list(arguments):
            first: object = arguments[0] if arguments else None
            arguments = first if is_object_map(first) else {}
        if not is_object_map(arguments):
            raise TypeError("tool arguments must be an object")
        processed_args = self._process_placeholders(arguments, execution_context)
        if self._python_tools_available:
            if self.tool_manager.has_python_implementation(tool_name):
                return self.tool_manager.invoke_python_tool(
                    tool_name,
                    processed_args,
                )
            class_key = self.tool_manager.api_name_to_class_key.get(
                tool_name,
                "NOT IN MAP",
            )
            raise NotImplementedError(
                f"No Python implementation for '{tool_name}' (api_name_to_class_key={class_key})."
            )
        return self.tool_manager.invoke_tool(tool_name, processed_args)

    @override
    def _verify_tool_query_consistency(
        self,
        tool_name: str,
        arguments: ObjectMap,
        query: str,
        trajectory: list[TrajectoryStep],
        execution_context: ObjectMap,
    ) -> tuple[bool, str]:
        """Verify that a tool invocation fits the policy-visible context."""
        trajectory_summary = "".join(
            f"Step {index}: {tool_call.tool_name}({tool_call.arguments}) -> {str(tool_call.output)[:200] if tool_call.output else 'None'}\n"
            for index, step in enumerate(trajectory, 1)
            for tool_call in step.tool_calls
        )
        prompt = f"""You are verifying that a tool invocation is consistent with the user query and conversation context.

=== USER QUERY ===
{query}

=== SELECTED TOOL ===
{tool_name}

=== FULL TOOL DEFINITION ===
{json.dumps(self.tool_manager.get_tool_schema(tool_name), indent=2, default=str)}

=== GENERATED ARGUMENTS ===
{json.dumps(arguments, indent=2)}

=== PREVIOUS TRAJECTORY ===
{trajectory_summary or "None"}

=== EXECUTION CONTEXT ===
{json.dumps(execution_context, indent=2, default=str)[:1500]}

Verify intent, types, ranges, dependencies, sufficiency, and provenance of opaque values.
Do not suggest replacement values. Respond ONLY with valid JSON:
{{"is_valid": true/false, "issues": ["issue1"]}}"""
        try:
            response_text = self._safe_llm_generate(
                [{"role": "user", "content": prompt}],
                llm=self.judge,
            ).strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            else:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]
            result = ValidationResponse.model_validate_json(response_text)
            return result.is_valid, "; ".join(result.issues)
        except (RuntimeError, ValueError) as exc:
            print(f"    Warning: Consistency verification failed: {exc}")
            return False, "Consistency verifier unavailable"

    @staticmethod
    @override
    def _detect_tool_error(
        tool_name: str,
        output: ObjectMap,
    ) -> tuple[bool, str]:
        """Detect generic and tool-specific error values."""
        for key in ("error", "error_message", "error_code"):
            if key in output:
                return True, str(output[key])
        for key, value in output.items():
            if _is_tool_specific_error(tool_name, key, value):
                return True, f"{key}: {value}"
            if isinstance(value, str) and value.startswith("Error:"):
                return True, f"{key}: {value}"
        return False, ""
