"""Assistant-response assembly for completed multi-turn trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from src.multi_turn_models import Turn
    from src.multi_turn_protocols import ExecutionContext, GeneratorMixinBase
    from src.step_by_step_models import QueryGenerationResult, TrajectoryStep
else:
    from multi_turn_protocols import GeneratorMixinBase

from multi_turn_protocols import (
    get_public_facade,
    is_object_dict,
    is_object_list,
    tool_call_view,
)


class AssistantResponsesMixin(GeneratorMixinBase):
    """Convert an executed trajectory into a persisted assistant turn."""

    @override
    def _complete_generated_turn(
        self,
        turn_number: int,
        query_result: QueryGenerationResult,
        trajectory: list[TrajectoryStep],
        turn_execution_context: ExecutionContext,
        execution_context: ExecutionContext,
        tools_used: set[str],
        categories_used: set[str],
    ) -> Turn:
        for key, value in turn_execution_context.items():
            execution_context[key] = value

        turn_output_aggregate: dict[str, object] = {}
        for step in trajectory:
            for raw_tool_call in step.tool_calls:
                tool_call = tool_call_view(raw_tool_call)
                output = tool_call.output
                if output and is_object_dict(output):
                    turn_output_aggregate[tool_call.tool_name] = output

        raw_turn_outputs = execution_context.get("turn_outputs")
        if is_object_list(raw_turn_outputs):
            turn_outputs = raw_turn_outputs
        else:
            turn_outputs = []
            execution_context["turn_outputs"] = turn_outputs
        turn_outputs.append(turn_output_aggregate)

        assistant_response = self._generate_final_response(
            query_result.query,
            trajectory,
            execution_context,
        )
        self._update_token_usage()

        for step in trajectory:
            for raw_tool_call in step.tool_calls:
                tool_call = tool_call_view(raw_tool_call)
                tools_used.add(tool_call.tool_name)
                category = self.tool_manager.get_tool_category(tool_call.tool_name)
                if category:
                    categories_used.add(category)

        return get_public_facade().Turn(
            turn_number=turn_number,
            user_query=query_result.query,
            query_intent=query_result.intent,
            steps=trajectory,
            assistant_response=assistant_response,
            expected_tools=query_result.expected_tools,
            execution_context=dict(execution_context),
        )
