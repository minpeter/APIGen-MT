"""Argument generation and stage-two tool-call orchestration."""

import json
from typing import Protocol, override

from step_by_step_models import (
    ObjectMap,
    QueryGenerationResult,
    ToolCallWithOutput,
    TrajectoryStep,
)
from step_by_step_protocols import (
    StepByStepMixinBase,
    is_object_list,
    is_object_map,
)


class ToolGenerationMixin(StepByStepMixinBase, Protocol):
    @override
    def _stage2_generate_tools(
        self,
        query_result: QueryGenerationResult,
        max_retries_per_tool: int,
        initial_execution_context: ObjectMap | None = None,
    ) -> tuple[list[TrajectoryStep] | None, ObjectMap | None]:
        """
        Stage 2: Generate tool invocations tool-by-tool.
        Uses expected_tools from Stage 1 directly - no LLM selection needed.
        - Each tool has its own retry count for argument generation
        - Feedback is wiped on successful tool completion
        - Captures pre/post API state snapshots around each tool call
        - Runs LLM-as-judge state verification after each call
        - If any tool fails after max retries, entire stage fails
        - Returns (trajectory, execution_context) or None
        """
        trajectory: list[TrajectoryStep] = []
        execution_context: ObjectMap = (
            initial_execution_context.copy()
            if initial_execution_context
            else {}
        )

        for step_num, tool_name in enumerate(query_result.expected_tools, 1):
            total_steps = len(query_result.expected_tools)
            print(f"\n[Step {step_num}/{total_steps}] Processing tool: {tool_name}")

            tool_feedback = ""
            step_success = False

            for attempt in range(max_retries_per_tool):
                print(f" [Attempt {attempt + 1}/{max_retries_per_tool}]")

                # ── Capture PRE state snapshot ──
                pre_state = self.tool_manager.get_api_state() if self._python_tools_available else None

                # Generate arguments for this tool (with feedback from previous failures)
                print(f"  Generating arguments for {tool_name}...")
                arguments, error = self._generate_tool_arguments(
                    tool_name=tool_name,
                    query=query_result.query,
                    trajectory=trajectory,
                    execution_context=execution_context,
                    feedback=tool_feedback if tool_feedback else None,
                )

                if error:
                    print(f" ✗ {error}")
                    if error.startswith("Required argument is not policy-visible"):
                        print(" ✗ Rejecting datapoint: the generated query does not expose all required arguments")
                        return None, None
                    if attempt < max_retries_per_tool - 1:
                        continue
                    break

                if arguments is None:
                    print(" ✗ Argument generation returned no arguments")
                    break
                print(f" Arguments: {json.dumps(arguments)}")

                # ── LLM-as-judge consistency verification ──
                print("  Verifying tool-query consistency...")
                is_consistent, consistency_feedback = self._verify_tool_query_consistency(
                    tool_name=tool_name,
                    arguments=arguments,
                    query=query_result.query,
                    trajectory=trajectory,
                    execution_context=execution_context,
                )
                if not is_consistent:
                    print(f"  ✗ Consistency check failed: {consistency_feedback}")
                    if attempt < max_retries_per_tool - 1:
                        tool_feedback = "Previous arguments failed consistency verification."
                        print("  Retrying with feedback...")
                        continue
                    # Last attempt failed - this is a HARD ERROR, do not proceed
                    print("  ✗ Max retries exceeded for consistency check - aborting tool")
                    break
                else:
                    print("  ✓ Consistency check passed")

                # Simulate tool execution
                print(f" Simulating {tool_name}...")
                output: object = self._simulate_tool_execution(
                    tool_name=tool_name,
                    arguments=arguments,
                    execution_context=execution_context
                )

                rendered_output = (
                    json.dumps(output, indent=2, ensure_ascii=False)
                    if is_object_map(output) or is_object_list(output)
                    else str(output)
                )
                print(f" Output: {rendered_output}")

                # Check for tool errors
                if is_object_map(output):
                    has_error, error_detail = self._detect_tool_error(tool_name, output)
                    if has_error:
                        error_type = output.get('error_type', 'execution_error')
                        print(f" ✗ Tool returned error: {error_detail}")
                        if error_type == 'validation_failure' and attempt < max_retries_per_tool - 1:
                            tool_feedback = "The previous internal call failed validation."
                            print(" Retrying due to validation failure...")
                            continue
                        elif attempt < max_retries_per_tool - 1:
                            tool_feedback = (
                                "The previous internal call failed. Re-read only the "
                                "saved policy-visible context and tool definition."
                            )
                            print(" Retrying with feedback...")
                            continue
                        break

                # Validate output against declared type/description immediately
                tool_schema = self.tool_manager.get_tool_schema(tool_name)
                if tool_schema and self.validate_outputs:
                    expected_type_value = tool_schema.get("output_type", "unknown")
                    expected_description_value = tool_schema.get(
                        "output_description",
                        "",
                    )
                    expected_type = (
                        expected_type_value
                        if isinstance(expected_type_value, str)
                        else "unknown"
                    )
                    expected_description = (
                        expected_description_value
                        if isinstance(expected_description_value, str)
                        else ""
                    )
                    validation = self.verify_output_consistency(
                        tool_name,
                        step_num,
                        output,
                        expected_type,
                        expected_description,
                    )
                    validation_issues = validation.get("issues", [])
                    issues = (
                        [item for item in validation_issues if isinstance(item, str)]
                        if is_object_list(validation_issues)
                        else ["Type mismatch"]
                    )
                    if validation.get("output_type_matches") is not True or issues:
                        issues_str = "; ".join(issues or ["Type mismatch"])
                        print(f" ✗ Output validation failed: {issues_str}")
                        if attempt < max_retries_per_tool - 1:
                            tool_feedback = "The previous internal output failed schema validation."
                            print(" Retrying with new arguments...")
                            continue
                        print(" Max retries exceeded, proceeding with potentially invalid output")

                # ── Capture POST state snapshot ──
                post_state = self.tool_manager.get_api_state() if self._python_tools_available else None

                # ── LLM-as-judge state verification ──
                state_verification = None
                if pre_state is not None and post_state is not None:
                    print(f" Verifying state transition for {tool_name}...")
                    state_verification = self.verify_state_transition(
                        tool_name=tool_name,
                        tool_arguments=arguments,
                        tool_output=output,
                        pre_state=pre_state,
                        post_state=post_state,
                    )
                    if state_verification.is_valid:
                        print(f" ✓ State verification passed: {state_verification.state_changes_summary}")
                    else:
                        issues_joined = '; '.join(state_verification.issues)
                        print(f" ✗ State verification FAILED: {issues_joined}")
                        if attempt < max_retries_per_tool - 1:
                            tool_feedback = (
                                "The previous internal attempt failed state verification. "
                                "Recompute only from saved policy-visible sources."
                            )
                            print(" Retrying due to state verification failure...")
                            # Restore PRE-tool snapshot (do not draw a new config)
                            self._replay_state(trajectory, baseline_state=pre_state)
                            continue
                        print(" Max retries exceeded, proceeding despite state verification failure")

                # SUCCESS: Tool completed - add to trajectory
                print(" ✓ Tool execution successful")

                # Update execution context
                if is_object_map(output):
                    for k, v in output.items():
                        execution_context[f"{tool_name}_{k}"] = v
                    # Store access_token directly for convenience (critical for auth-gated tools)
                    if 'access_token' in output:
                        execution_context['access_token'] = output['access_token']
                execution_context[f"{tool_name}_output"] = output

                # Add to trajectory (with state snapshots + verification)
                tool_call = ToolCallWithOutput(
                    tool_name=tool_name,
                    arguments=arguments,
                    output=output
                )
                trajectory_step = TrajectoryStep(
                    step_number=step_num,
                    tool_calls=[tool_call],
                    reasoning=f"Generated arguments for {tool_name} based on query context",
                    pre_state=pre_state,
                    post_state=post_state,
                    state_verification=state_verification,
                )
                trajectory.append(trajectory_step)
                step_success = True
                break

            if not step_success:
                print(f"\n✗ Tool {tool_name} failed after {max_retries_per_tool} attempts")
                return None, None

        # All tools completed successfully
        return trajectory, execution_context
