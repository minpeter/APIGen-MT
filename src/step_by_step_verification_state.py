"""Existing LLM-as-judge state-transition verification."""

import json
from typing import Protocol, override

from step_by_step_models import (
    ObjectMap,
    StateSnapshot,
    StateVerificationResult,
)
from step_by_step_protocols import StepByStepMixinBase, is_object_map

_MISSING = "<MISSING>"


def _state_changes(
    pre_state: StateSnapshot,
    post_state: StateSnapshot,
) -> dict[str, ObjectMap]:
    changed_classes: dict[str, ObjectMap] = {}
    for class_key in set(pre_state) | set(post_state):
        before = pre_state.get(class_key, {})
        after = post_state.get(class_key, {})
        if before == after:
            continue
        differences: ObjectMap = {}
        for key in set(before) | set(after):
            before_value = before.get(key, _MISSING)
            after_value = after.get(key, _MISSING)
            if before_value != after_value:
                differences[key] = {
                    "before": before_value,
                    "after": after_value,
                }
        if differences:
            changed_classes[class_key] = differences
    return changed_classes


def _change_summary(changed_classes: dict[str, ObjectMap]) -> ObjectMap:
    summary: ObjectMap = {}
    for class_key, differences in changed_classes.items():
        changes: list[str] = []
        for key, values in differences.items():
            if not is_object_map(values):
                continue
            before = values.get("before", _MISSING)
            after = values.get("after", _MISSING)
            if before == _MISSING:
                changes.append(f"{key}: added")
            elif after == _MISSING:
                changes.append(f"{key}: removed")
            else:
                changes.append(f"{key}: modified")
        summary[class_key] = changes
    return summary


class StateVerificationMixin(StepByStepMixinBase, Protocol):
    @override
    def verify_state_transition(
        self,
        tool_name: str,
        tool_arguments: ObjectMap,
        tool_output: object,
        pre_state: StateSnapshot,
        post_state: StateSnapshot,
    ) -> StateVerificationResult:
        """Ask the judge whether a recorded state transition is coherent."""
        changed_classes = _state_changes(pre_state, post_state)
        if not changed_classes:
            return StateVerificationResult(
                is_valid=True,
                reasoning="No state changes detected (read-only or no-op call).",
                issues=[],
                state_changes_summary="No state changes.",
            )

        tool_class_key = self.tool_manager.api_name_to_class_key.get(
            tool_name,
            "unknown",
        )
        output_text = (
            tool_output
            if isinstance(tool_output, str)
            else json.dumps(tool_output, default=str, ensure_ascii=False)
        )
        if len(output_text) > 1000:
            output_text = output_text[:1000] + "... (truncated)"
        arguments_text = json.dumps(
            tool_arguments,
            default=str,
            ensure_ascii=False,
        )
        if len(arguments_text) > 1000:
            arguments_text = arguments_text[:1000] + "... (truncated)"

        prompt = f"""You are an expert API state auditor. Verify that the state transition produced by a tool call is logically correct and consistent with the tool's output.

=== TOOL CALL ===
Tool: {tool_name}
Class: {tool_class_key}
Arguments: {arguments_text}
Output: {output_text}

=== STATE CHANGE SUMMARY ===
{json.dumps(_change_summary(changed_classes), indent=2, default=str, ensure_ascii=False)}

For each changed class, the list shows what fields were added, removed, or modified.

=== YOUR TASK ===
1. Check whether the state changes match the tool semantics.
2. Verify authentication/login state updates.
3. Verify data mutations are reflected in state.
4. Check for contradictory or nonsensical changes.

If validity cannot be determined from the summary, assume it is valid. Mark it invalid only for clear contradictions.

Respond ONLY with valid JSON:
{{
  "is_valid": true/false,
  "reasoning": "brief explanation of your verdict",
  "issues": ["list of issues found, empty if valid"],
  "state_changes_summary": "human-readable summary of what changed"
}}"""
        try:
            response_text = self._safe_llm_generate(
                [{"role": "user", "content": prompt}],
                llm=self.judge,
            ).strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                response_text = response_text[start:end]
            return StateVerificationResult.model_validate_json(response_text)
        except (RuntimeError, ValueError) as exc:
            print(f" Warning: State verification LLM call failed: {exc}")
            return StateVerificationResult(
                is_valid=False,
                reasoning=f"LLM judge call failed: {exc}",
                issues=[f"Judge call failed: {exc}"],
                state_changes_summary="Could not verify (LLM error).",
            )
