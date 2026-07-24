"""Checkpoint reconstruction and resumed multi-turn generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import TypeAdapter, ValidationError

if TYPE_CHECKING:
    from src.multi_turn_models import MultiTurnDatapoint
    from src.multi_turn_protocols import (
        ApiState,
        Checkpoint,
        ExecutionContext,
        GeneratorMixinBase,
    )
    from src.step_by_step_models import TrajectoryStep
else:
    from multi_turn_protocols import GeneratorMixinBase

from multi_turn_protocols import (
    get_public_facade,
    string_list,
    string_value,
    tool_call_view,
)

_API_STATE_ADAPTER = TypeAdapter(dict[str, dict[str, object]])
_CONTEXT_ADAPTER = TypeAdapter(dict[str, object])
_TURNS_ADAPTER = TypeAdapter(list[dict[str, object]])


class CheckpointRestorationMixin(GeneratorMixinBase):
    """Restore persisted state and generate the remaining turns."""

    def continue_from_checkpoint(
        self,
        checkpoint: Checkpoint,
        focus_category: str | None = None,
        query_retries: int = 3,
        tool_retries: int = 3,
    ) -> MultiTurnDatapoint | None:
        """Continue generating a multi-turn datapoint from a checkpoint."""
        del query_retries
        facade = get_public_facade()
        raw_blueprint = checkpoint.get("blueprint")
        if not isinstance(raw_blueprint, dict):
            print("✗ Checkpoint missing blueprint")
            return None

        try:
            blueprint_data = _CONTEXT_ADAPTER.validate_python(raw_blueprint)
            blueprint = facade.DialogBlueprint(
                overall_task=string_value(blueprint_data, "overall_task"),
                num_turns=(
                    raw_num_turns
                    if isinstance(
                        raw_num_turns := blueprint_data.get("num_turns"),
                        int,
                    )
                    else self.num_turns
                ),
                turns=_TURNS_ADAPTER.validate_python(
                    blueprint_data.get("turns", [])
                ),
            )
        except (TypeError, ValueError, ValidationError) as error:
            print(f"✗ Failed to reconstruct blueprint: {error}")
            return None

        try:
            partial_data = _CONTEXT_ADAPTER.validate_python(
                checkpoint.get("partial_conversation", {})
            )
            conversation = facade.MultiTurnConversation.model_validate(
                {
                    "overall_task": string_value(
                        partial_data,
                        "overall_task",
                        blueprint.overall_task,
                    ),
                    "turns": partial_data.get("turns", []),
                    "tools_used": string_list(partial_data, "tools_used"),
                    "categories_used": string_list(
                        partial_data,
                        "categories_used",
                    ),
                }
            )
        except (TypeError, ValueError, ValidationError) as error:
            print(f"✗ Failed to reconstruct conversation: {error}")
            return None

        raw_completed_turns = checkpoint.get("completed_turns")
        completed_turns = (
            raw_completed_turns
            if isinstance(raw_completed_turns, int)
            else len(conversation.turns)
        )
        execution_context: ExecutionContext = _CONTEXT_ADAPTER.validate_python(
            checkpoint.get("execution_context", {})
        )
        raw_initial_state = checkpoint.get("initial_api_state")
        initial_api_state: ApiState | None = (
            _API_STATE_ADAPTER.validate_python(raw_initial_state)
            if raw_initial_state is not None
            else None
        )

        print(f"\n{'=' * 70}")
        print("RESUMING FROM CHECKPOINT")
        print("=" * 70)
        print(f" Overall task: {blueprint.overall_task}")
        print(f" Completed turns: {completed_turns}/{blueprint.num_turns}")
        print(f" Remaining turns: {blueprint.num_turns - completed_turns}")

        if self._python_tools_available and initial_api_state:
            print("\n Restoring API state from checkpoint...")
            self.tool_manager.restore_api_state(initial_api_state)

            if completed_turns > 0:
                print(f" Replaying {completed_turns} turns to restore state...")
                for turn_index in range(completed_turns):
                    if turn_index >= len(conversation.turns):
                        break
                    for step in conversation.turns[turn_index].steps:
                        for raw_tool_call in step.tool_calls:
                            tool_call = tool_call_view(raw_tool_call)
                            if self.tool_manager.has_python_implementation(
                                tool_call.tool_name
                            ):
                                _ = self.tool_manager.invoke_python_tool(
                                    tool_call.tool_name,
                                    tool_call.arguments,
                                )
                print(f" Replayed {completed_turns} turns to restore state")

        self._update_token_usage()
        tools_used = set(conversation.tools_used)
        categories_used = set(conversation.categories_used)

        for turn_index in range(completed_turns, blueprint.num_turns):
            print(f"\n{'=' * 70}")
            print(f"TURN {turn_index + 1}/{blueprint.num_turns} (resumed)")
            print("=" * 70)

            query_result = self._generate_turn_query(
                blueprint=blueprint,
                conversation=conversation,
                turn_index=turn_index,
            )
            if query_result is None:
                print(
                    f"✗ Turn {turn_index + 1} failed: Could not generate query"
                )
                return None
            self._update_token_usage()

            trajectory: list[TrajectoryStep] | None = None
            turn_context: ExecutionContext = {}
            for attempt in range(tool_retries):
                raw_trajectory, turn_context = self._stage2_generate_tools(
                    query_result,
                    tool_retries - attempt,
                    initial_execution_context=execution_context,
                )
                if raw_trajectory is None:
                    print(
                        f"✗ Turn {turn_index + 1}: Could not generate tool calls"
                    )
                    return None

                errors = self._validate_tool_arguments(raw_trajectory)
                cross_errors = self._validate_cross_turn_consistency(
                    raw_trajectory,
                    execution_context,
                )
                if not errors and not cross_errors:
                    trajectory = raw_trajectory
                    break

                print(
                    f"  ⚠ Turn {turn_index + 1} validation failed "
                    + f"(attempt {attempt + 1}/{tool_retries}):"
                )
                for error in errors:
                    print(f"    arg: {error}")
                for error in cross_errors:
                    print(f"    cross: {error}")
                if attempt < tool_retries - 1:
                    print(f"  Retrying turn {turn_index + 1}...")

            if trajectory is None:
                print(
                    f"✗ Turn {turn_index + 1}: Too many validation failures, "
                    + "rejecting datapoint"
                )
                return None
            self._update_token_usage()

            turn = self._complete_generated_turn(
                turn_index + 1,
                query_result,
                trajectory,
                turn_context,
                execution_context,
                tools_used,
                categories_used,
            )
            conversation.turns.append(turn)
            print(f"\n✓ Turn {turn_index + 1} complete (resumed)")
            print(f"   Query: {query_result.query[:80]}...")

        conversation.tools_used = sorted(tools_used)
        conversation.categories_used = sorted(categories_used)
        conversation.initial_api_state = initial_api_state

        all_steps = [
            step
            for turn in conversation.turns
            for step in turn.steps
        ]
        verification = self.run_full_verification(
            "\n".join(turn.user_query for turn in conversation.turns),
            all_steps,
            execution_context,
            initial_api_state,
            [
                tool
                for turn in blueprint.turns
                for tool in string_list(turn, "expected_tools")
            ],
        )
        if not verification.overall_verification_passed:
            print("✗ Resumed multi-turn final verification failed")
            return None

        datapoint = facade.MultiTurnDatapoint(
            conversation=conversation,
            generation_metadata={
                "num_turns": self.num_turns,
                "focus_category": focus_category,
                "overall_task": blueprint.overall_task,
                "resumed_from_turn": completed_turns,
                "blueprint_queries": [
                    string_value(turn, "user_query")
                    for turn in blueprint.turns
                ],
                "turn_expected_tools": [
                    string_list(turn, "expected_tools")
                    for turn in blueprint.turns
                ],
            },
            verification_result=verification.model_dump(),
            token_usage=self._get_token_stats(),
            initial_api_state=conversation.initial_api_state,
        )

        print("\n" + "=" * 70)
        print("✓ RESUMED MULTI-TURN DATAPOINT GENERATION COMPLETE")
        print("=" * 70)
        print(f" Turns: {len(conversation.turns)}")
        print(f" Tools used: {conversation.tools_used}")
        print(
            " Total tool calls: "
            + str(sum(len(turn.steps) for turn in conversation.turns))
        )

        return datapoint
