"""Policy-context-aware argument generation for one tool call."""

import json
from typing import Protocol, override

from step_by_step_models import ObjectMap, TrajectoryStep
from step_by_step_protocols import (
    StepByStepMixinBase,
    parse_object_map,
)


class ArgumentGenerationMixin(StepByStepMixinBase, Protocol):
    @override
    def _generate_tool_arguments(
        self,
        tool_name: str,
        query: str,
        trajectory: list[TrajectoryStep],
        execution_context: ObjectMap,
        feedback: str | None = None,
    ) -> tuple[ObjectMap | None, str | None]:
        """Generate schema-conforming arguments from policy-visible context."""
        tool_schema = self.tool_manager.get_tool_schema(tool_name)
        if not tool_schema:
            return None, f"Tool '{tool_name}' not found"

        visible_history: list[ObjectMap] = []
        for step in trajectory:
            for tool_call in step.tool_calls:
                visible_history.append(
                    {
                        "step_number": step.step_number,
                        "tool_name": tool_call.tool_name,
                        "arguments": tool_call.arguments,
                        "output": tool_call.output,
                    }
                )

        prompt = f"""Generate arguments for '{tool_name}' based on user query and previous steps.

=== USER QUERY ===
{query}

=== PREVIOUS POLICY-VISIBLE TOOL CALLS AND OUTPUTS ===
{json.dumps(visible_history, indent=2, ensure_ascii=False, default=str) if visible_history else "None"}

=== EXECUTION CONTEXT ===
{json.dumps(execution_context, indent=2, ensure_ascii=False, default=str)}
=== FULL TOOL DEFINITION ===
{json.dumps(tool_schema, indent=2, ensure_ascii=False, default=str)}

=== EXPECTED OUTPUT ===
Type: {tool_schema.get("output_type", "unknown")}
Description: {tool_schema.get("output_description", "")}
"""
        if feedback:
            prompt += """
=== RETRY NOTICE ===
A previous candidate was rejected. Recompute the arguments only from the user
query, prior saved tool outputs, and the full tool definition above. No value
from a judge message, failed internal attempt, or tool error is available.
"""
        prompt += """
=== TASK ===
Generate args matching schema and fulfilling query:
- Use only values explicitly present in the USER QUERY, previous tool outputs,
  EXECUTION CONTEXT, or defaults declared in the TOOL SCHEMA.
- Deterministic calculations and format normalization from visible values are
  allowed only when they do not require an external lookup.
- General/model knowledge is not an argument source. Do not convert a visible
  human-readable label into an opaque ID, code, token, symbol, handle,
  coordinate, path, or credential unless that exact value is visible.
- The simulator's private API state is not available to the solving assistant.
  Never invent, guess, or copy a value from hidden state.
- Values mentioned only by an internal judge, rejected attempt, or failed tool
  call are unavailable because those diagnostics are not saved in the trace.
- If any required argument is unavailable from the visible sources above, return
  {"__missing_required_argument__": ["argument_name"]} instead of guessing.
- Storage tools use direct arguments, never a 'calls' batch wrapper.

Respond JSON: {"arg1": "value1", ...}
"""

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

            arguments = parse_object_map(response_text)
            if "__missing_required_argument__" in arguments:
                missing = arguments.get("__missing_required_argument__")
                return None, f"Required argument is not policy-visible: {missing}"
            return arguments, None
        except ValueError as exc:
            return None, f"JSON parsing error: {exc}"
