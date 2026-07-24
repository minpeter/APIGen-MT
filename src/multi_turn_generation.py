"""Fresh multi-turn datapoint generation orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.multi_turn_models import MultiTurnDatapoint
    from src.multi_turn_protocols import (
        ApiState,
        CheckpointCallback,
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
    verification_result_view,
)


class TurnGenerationMixin(GeneratorMixinBase):
    """Run Stage 0 and each tool-backed turn for a fresh datapoint."""

    def generate_multi_turn_datapoint(
        self,
        focus_category: str | None = None,
        query_retries: int = 3,
        tool_retries: int = 3,
        checkpoint_callback: CheckpointCallback | None = None,
    ) -> MultiTurnDatapoint | None:
        """Generate a multi-turn datapoint, checkpointing after each turn."""
        self._reset_token_tracking()
        self._capture_initial_usage()

        initial_api_state: ApiState | None = None
        if self._python_tools_available:
            self.tool_manager.initialize_api_state(force_new=True)
            initial_api_state = self.tool_manager.get_api_state()
            print(
                f" Captured initial API state ({len(initial_api_state)} class keys)"
            )

        print("\n" + "=" * 70)
        print("STAGE 0: Generate Dialog Blueprint")
        print("=" * 70)
        blueprint = self._stage0_generate_blueprint(
            focus_category,
            initial_api_state,
            max(1, query_retries),
        )
        if blueprint is None:
            print("✗ Stage 0 failed: Could not generate dialog blueprint")
            return None
        self._update_token_usage()
        print(f" Overall task: {blueprint.overall_task}")
        for index, turn_spec in enumerate(blueprint.turns, 1):
            user_query = string_value(turn_spec, "user_query")
            print(f"   Turn {index}: {user_query[:80]}...")

        facade = get_public_facade()
        conversation = facade.MultiTurnConversation(
            overall_task=blueprint.overall_task
        )

        if self._python_tools_available:
            print("\n" + "-" * 70)
            print("STAGE 0.5: Ensure User Identity Coherence")
            print("-" * 70)
            identity_adjusted = self._ensure_user_identity_coherence(
                blueprint.overall_task
            )
            if identity_adjusted:
                initial_api_state = self.tool_manager.get_api_state()
                print(" ✓ Identity coherence adjusted")
            else:
                print(" No identity adjustment needed")

        execution_context: ExecutionContext = {}
        tools_used: set[str] = set()
        categories_used: set[str] = set()

        for turn_index in range(blueprint.num_turns):
            print(f"\n{'=' * 70}")
            print(f"TURN {turn_index + 1}/{blueprint.num_turns}")
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

            print(f"\n✓ Turn {turn_index + 1} complete")
            print(f"   Query: {query_result.query[:80]}...")
            print(f"   Steps: {len(trajectory)}")

            if checkpoint_callback:
                checkpoint_callback(
                    {
                        "blueprint": {
                            "overall_task": blueprint.overall_task,
                            "num_turns": blueprint.num_turns,
                            "turns": blueprint.turns,
                        },
                        "partial_conversation": conversation.model_dump(),
                        "execution_context": dict(execution_context),
                        "completed_turns": turn_index + 1,
                        "initial_api_state": initial_api_state,
                        "focus_category": focus_category,
                    }
                )
                print(f"   Checkpoint saved after turn {turn_index + 1}")

        conversation.tools_used = sorted(tools_used)
        conversation.categories_used = sorted(categories_used)
        conversation.initial_api_state = (
            facade.filter_api_state(initial_api_state, list(tools_used))
            if initial_api_state
            else None
        )

        all_steps = [
            step
            for turn in conversation.turns
            for step in turn.steps
        ]
        verification = verification_result_view(
            self.run_full_verification(
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
        )
        if not verification.overall_verification_passed:
            print("✗ Multi-turn final verification failed")
            return None

        datapoint = facade.MultiTurnDatapoint(
            conversation=conversation,
            generation_metadata={
                "num_turns": self.num_turns,
                "focus_category": focus_category,
                "overall_task": blueprint.overall_task,
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
        print("✓ MULTI-TURN DATAPOINT GENERATION COMPLETE")
        print("=" * 70)
        print(f" Turns: {len(conversation.turns)}")
        print(f" Tools used: {conversation.tools_used}")
        print(
            " Total tool calls: "
            + str(sum(len(turn.steps) for turn in conversation.turns))
        )

        return datapoint
