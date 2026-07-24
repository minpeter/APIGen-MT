"""Final response generation and datapoint assembly."""

import json
from typing import Protocol, override

from step_by_step_models import (
    ConversationTrajectory,
    ObjectMap,
    QueryGenerationResult,
    StateSnapshot,
    StepByStepDatapoint,
    TrajectoryStep,
)
from step_by_step_protocols import StepByStepMixinBase, parse_object_map
from tool_manager import filter_api_state


class FinalizationMixin(StepByStepMixinBase, Protocol):
    @override
    def _stage3_finalize(
        self,
        query_result: QueryGenerationResult,
        trajectory: list[TrajectoryStep],
        execution_context: ObjectMap,
        focus_category: str | None,
        initial_api_state: StateSnapshot | None = None,
    ) -> StepByStepDatapoint | None:
        """Assemble and verify the final datapoint without retrying."""
        print("\nGenerating final response...")
        final_response = self._generate_final_response(
            query_result.query,
            trajectory,
            execution_context,
        )
        print(f" Final response: {final_response}")

        tools_used: list[str] = []
        categories_used: set[str] = set()
        for step in trajectory:
            for tool_call in step.tool_calls:
                if tool_call.tool_name not in tools_used:
                    tools_used.append(tool_call.tool_name)
                category = self.tool_manager.get_tool_category(tool_call.tool_name)
                if category:
                    categories_used.add(category)

        filtered_initial_state = (
            filter_api_state(initial_api_state, tools_used)
            if initial_api_state
            else None
        )
        filtered_trajectory: list[TrajectoryStep] = []
        for step in trajectory:
            filtered_trajectory.append(
                TrajectoryStep(
                    step_number=step.step_number,
                    tool_calls=step.tool_calls,
                    reasoning=step.reasoning,
                    pre_state=(
                        filter_api_state(step.pre_state, tools_used)
                        if step.pre_state
                        else None
                    ),
                    post_state=(
                        filter_api_state(step.post_state, tools_used)
                        if step.post_state
                        else None
                    ),
                    state_verification=step.state_verification,
                )
            )

        intermediate_states: list[ObjectMap] = []
        for step in filtered_trajectory:
            verification = step.state_verification
            if step.post_state is not None and verification is not None:
                intermediate_states.append(
                    {
                        "step_number": step.step_number,
                        "post_state": step.post_state,
                        "state_verification": {
                            "is_valid": verification.is_valid,
                            "reasoning": verification.reasoning,
                            "issues": verification.issues,
                            "state_changes_summary": (
                                verification.state_changes_summary
                            ),
                        },
                    }
                )

        conversation = ConversationTrajectory(
            query=query_result.query,
            steps=filtered_trajectory,
            final_response=final_response,
            tools_used=tools_used,
            categories_used=list(categories_used),
            initial_api_state=filtered_initial_state,
        )

        print("\nRunning verification...")
        verification_result = self.run_full_verification(
            query=query_result.query,
            trajectory=trajectory,
            execution_context=execution_context,
            initial_api_state=initial_api_state,
            expected_tools=query_result.expected_tools,
        )
        if not verification_result.overall_verification_passed:
            print("  Verification: FAILED")
            print(f"  Details: {verification_result.verification_summary}")
            for output_validation in verification_result.output_validations:
                if not output_validation.get("output_type_matches", True):
                    print(
                        f"    - {output_validation.get('tool_name')}: {output_validation.get('issues')}"
                    )
            print("\n✗ Datapoint failed verification - discarding")
            return None

        print(" Verification: PASSED")
        self._update_token_usage()
        verification_data = parse_object_map(
            verification_result.model_dump_json()
        )
        return StepByStepDatapoint(
            trajectory=conversation,
            generation_metadata={
                "num_actions": len(trajectory),
                "focus_category": focus_category,
                "query_intent": query_result.intent,
                "expected_tools": query_result.expected_tools,
            },
            verification_result=verification_data,
            token_usage=self._get_token_stats(),
            initial_api_state=filtered_initial_state,
            intermediate_api_states=intermediate_states,
        )

    @override
    def _generate_final_response(
        self,
        query: str,
        trajectory: list[TrajectoryStep],
        execution_context: ObjectMap,
    ) -> str:
        """Generate a natural final response based on the conversation."""
        del execution_context
        actions_summary: list[ObjectMap] = []
        for step in trajectory:
            for tool_call in step.tool_calls:
                actions_summary.append(
                    {
                        "tool": tool_call.tool_name,
                        "arguments": tool_call.arguments,
                        "output_summary": (
                            str(tool_call.output)[:100]
                            if tool_call.output
                            else None
                        ),
                    }
                )

        prompt = f"""Based on the following conversation, generate a natural final response.

User Query: {query}

Actions taken:
{json.dumps(actions_summary, indent=2)}

Generate a concise, natural response that summarizes what was accomplished."""
        try:
            return self._safe_llm_generate(
                [{"role": "user", "content": prompt}]
            ).strip()
        except RuntimeError as exc:
            print(f"    Error generating final response: {exc}")
            return "I have completed your request."
