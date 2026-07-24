from typing import Protocol, override

from step_by_step_models import (
    ObjectMap,
    StateSnapshot,
    TrajectoryStep,
    VerificationResult,
)
from step_by_step_protocols import (
    StepByStepMixinBase,
    is_object_map,
    placeholder_keys,
)
from trajectory_replay import verify_trajectory_replay


def _relevance_tokens(text: str) -> set[str]:
    normalized = "".join(
        character if character.isalnum() else " "
        for character in text.lower()
    )
    tokens = normalized.split()
    return {
        token[:-1] if len(token) > 3 and token.endswith("s") else token
        for token in tokens
    }


def _string_field(mapping: ObjectMap, key: str, default: str) -> str:
    value = mapping.get(key, default)
    return value if isinstance(value, str) else default


class VerificationMixin(StepByStepMixinBase, Protocol):
    def verify_tool_relevance(
        self,
        query: str,
        tool_name: str,
        _step: TrajectoryStep,
    ) -> ObjectMap:
        tool_schema = self.tool_manager.get_tool_schema(tool_name)
        if not tool_schema:
            return {
                "tool_name": tool_name,
                "is_relevant": False,
                "relevance_score": 0.0,
                "reasoning": "Tool not found in tool pool",
            }

        keywords = _relevance_tokens(
            _string_field(tool_schema, "description", "")
        )
        query_words = _relevance_tokens(query)
        overlap = len(keywords & query_words)
        relevance_score = min(1.0, overlap / max(1, len(keywords)))
        name_overlap = len(
            _relevance_tokens(tool_name.replace("_", " ")) & query_words
        )
        is_relevant = relevance_score > 0.1 or name_overlap > 0
        reasoning = (
            f"Tool '{tool_name}': score={relevance_score:.2f}, "
            f"name_match={name_overlap}"
        )
        reasoning += (
            ". Tool appears relevant."
            if is_relevant
            else ". Tool may not be directly relevant."
        )
        return {
            "tool_name": tool_name,
            "is_relevant": is_relevant,
            "relevance_score": relevance_score,
            "reasoning": reasoning,
        }

    def verify_invocation_order(
        self,
        _query: str,
        trajectory: list[TrajectoryStep],
        expected_tools: list[str] | None = None,
    ) -> ObjectMap:
        step_numbers = [step.step_number for step in trajectory]
        actual_tools = [
            call.tool_name
            for step in trajectory
            for call in step.tool_calls
        ]
        issues: list[str] = []
        if expected_tools is None and step_numbers != sorted(step_numbers):
            issues.append("Trajectory step numbers are out of order")
        if expected_tools is not None and actual_tools != expected_tools:
            issues.append(
                f"Expected tool sequence {expected_tools}, received {actual_tools}"
            )
        details = (
            "; ".join(issues)
            if issues
            else "No steps to verify"
            if not trajectory
            else "Order appears logical."
        )
        return {
            "order_is_correct": not issues,
            "order_verification_details": details,
            "issues": issues,
        }

    def verify_placeholder_resolution(
        self,
        trajectory: list[TrajectoryStep],
        execution_context: ObjectMap,
    ) -> ObjectMap:
        total_placeholders = 0
        resolved_count = 0
        details: list[ObjectMap] = []
        for step in trajectory:
            for tool_call in step.tool_calls:
                for argument_name, argument_value in tool_call.arguments.items():
                    if not isinstance(argument_value, str):
                        continue
                    for placeholder in placeholder_keys(argument_value):
                        total_placeholders += 1
                        current: object = execution_context
                        for key in placeholder.split("."):
                            if not is_object_map(current) or key not in current:
                                break
                            current = current[key]
                        else:
                            resolved_count += 1
                            details.append(
                                {
                                    "step": step.step_number,
                                    "tool": tool_call.tool_name,
                                    "argument": argument_name,
                                    "placeholder": f"{{{{{placeholder}}}}}",
                                    "resolved": True,
                                    "resolved_value": str(current)[:100],
                                }
                            )
                            continue
                        details.append(
                            {
                                "step": step.step_number,
                                "tool": tool_call.tool_name,
                                "argument": argument_name,
                                "placeholder": f"{{{{{placeholder}}}}}",
                                "resolved": False,
                                "resolved_value": None,
                            }
                        )

        return {
            "all_resolved": total_placeholders == resolved_count,
            "total_placeholders": total_placeholders,
            "resolved_count": resolved_count,
            "details": details,
        }

    @override
    def run_full_verification(
        self,
        query: str,
        trajectory: list[TrajectoryStep],
        execution_context: ObjectMap,
        initial_api_state: StateSnapshot | None = None,
        expected_tools: list[str] | None = None,
    ) -> VerificationResult:
        print("\n  Running Verification...")
        relevance_checks = [
            self.verify_tool_relevance(query, tool_call.tool_name, step)
            for step in trajectory
            for tool_call in step.tool_calls
        ]
        all_relevant = all(
            check.get("is_relevant") is True for check in relevance_checks
        )

        order_result = self.verify_invocation_order(
            query,
            trajectory,
            expected_tools,
        )
        order_is_correct = order_result.get("order_is_correct") is True
        order_details = _string_field(
            order_result,
            "order_verification_details",
            "",
        )

        output_validations: list[ObjectMap] = []
        for step in trajectory:
            for tool_call in step.tool_calls:
                schema = self.tool_manager.get_tool_schema(tool_call.tool_name)
                output_validations.append(
                    self.verify_output_consistency(
                        tool_call.tool_name,
                        step.step_number,
                        tool_call.output,
                        _string_field(schema, "output_type", "unknown"),
                        _string_field(schema, "output_description", ""),
                    )
                )
        all_outputs_valid = all(
            validation.get("output_type_matches") is True
            for validation in output_validations
        )

        placeholder_result = self.verify_placeholder_resolution(
            trajectory,
            execution_context,
        )
        all_placeholders_resolved = (
            placeholder_result.get("all_resolved") is True
        )
        deterministic_replay = verify_trajectory_replay(
            self.tool_manager,
            trajectory,
            initial_api_state,
        )
        state_verdicts_valid = all(
            step.state_verification is None or step.state_verification.is_valid
            for step in trajectory
        )
        overall_passed = (
            all_relevant
            and order_is_correct
            and all_outputs_valid
            and all_placeholders_resolved
            and deterministic_replay.is_valid
            and state_verdicts_valid
        )

        issues: list[str] = []
        if not all_relevant:
            issues.append("Some tools are not relevant to the query")
        if not order_is_correct:
            issues.append("Tool invocation order may be incorrect")
        if not all_outputs_valid:
            issues.append("Some tool outputs don't match their declarations")
        if not all_placeholders_resolved:
            total = placeholder_result.get("total_placeholders", 0)
            resolved = placeholder_result.get("resolved_count", 0)
            if isinstance(total, int) and isinstance(resolved, int):
                issues.append(f"{total - resolved} placeholders were not resolved")
        if not deterministic_replay.is_valid:
            issues.append(
                f"Deterministic replay found {len(deterministic_replay.issues)} issue(s)"
            )
        if not state_verdicts_valid:
            issues.append("A recorded state transition verdict is invalid")

        summary = (
            "Verification PASSED"
            if overall_passed
            else "Verification FAILED - " + "; ".join(issues)
        )
        return VerificationResult(
            query=query,
            tool_relevance_checks=relevance_checks,
            order_is_correct=order_is_correct,
            order_verification_details=order_details,
            output_validations=output_validations,
            placeholder_resolution=placeholder_result,
            deterministic_replay=deterministic_replay,
            overall_verification_passed=overall_passed,
            verification_summary=summary,
        )
