"""Deterministic replay verification for saved step-by-step trajectories."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, TypeAdapter

from step_by_step_models import (
    DeterministicReplayResult,
    ObjectMap,
    ReplayIssue,
    StateSnapshot,
    TrajectoryStep,
)
from step_by_step_protocols import (
    is_object_list,
    is_object_map,
    is_object_tuple,
)

_PLACEHOLDER = re.compile(r"\{\{([^{}]+)\}\}")
_OBJECT_ADAPTER = TypeAdapter(object)


@runtime_checkable
class ReplayManager(Protocol):
    def get_api_state(self) -> StateSnapshot: ...

    def restore_api_state(self, state: StateSnapshot) -> None: ...

    def has_python_implementation(self, tool_name: str) -> bool: ...

    def is_replay_safe(self, tool_name: str) -> bool: ...

    def invoke_python_tool(
        self,
        tool_name: str,
        params: ObjectMap,
    ) -> object: ...


def _normalized(value: object) -> object:
    if isinstance(value, BaseModel):
        return _normalized(_OBJECT_ADAPTER.validate_json(value.model_dump_json()))
    if isinstance(value, str):
        try:
            return _normalized(_OBJECT_ADAPTER.validate_json(value))
        except ValueError:
            return value
    if is_object_map(value):
        return {
            key: _normalized(item)
            for key, item in sorted(value.items())
        }
    if is_object_list(value):
        return [_normalized(item) for item in value]
    if is_object_tuple(value):
        return [_normalized(item) for item in value]
    return value


def _state_subset(
    state: StateSnapshot,
    expected: StateSnapshot | None,
) -> StateSnapshot | None:
    if expected is None:
        return None
    return {
        key: copy.deepcopy(state.get(key, {}))
        for key in expected
    }


def _digest(state: StateSnapshot) -> str:
    payload = json.dumps(
        _normalized(state),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _context_value(path: str, context: ObjectMap) -> object:
    parts = path.split(".")
    value: object = context[parts[0]]
    for part in parts[1:]:
        if is_object_map(value):
            value = value[part]
        elif is_object_list(value) and part.isdigit():
            value = value[int(part)]
        else:
            raise KeyError(path)
    return value


def _resolve_arguments(value: object, context: ObjectMap) -> object:
    if is_object_map(value):
        return {
            key: _resolve_arguments(item, context)
            for key, item in value.items()
        }
    if is_object_list(value):
        return [_resolve_arguments(item, context) for item in value]
    if not isinstance(value, str):
        return value

    full_match = _PLACEHOLDER.fullmatch(value)
    if full_match:
        return copy.deepcopy(_context_value(full_match.group(1), context))
    return _PLACEHOLDER.sub(
        lambda match: str(_context_value(match.group(1), context)),
        value,
    )


def verify_trajectory_replay(
    manager: object,
    trajectory: Sequence[TrajectoryStep],
    initial_state: StateSnapshot | None = None,
) -> DeterministicReplayResult:
    """Replay every available Python call and compare outputs and snapshots."""
    if not isinstance(manager, ReplayManager):
        return DeterministicReplayResult(status="unavailable", is_valid=False)

    replay_state = initial_state
    if replay_state is None:
        replay_state = next(
            (step.pre_state for step in trajectory if step.pre_state is not None),
            None,
        )
    if replay_state is None:
        return DeterministicReplayResult(status="unavailable", is_valid=False)

    unavailable_tools = list(
        dict.fromkeys(
            call.tool_name
            for step in trajectory
            for call in step.tool_calls
            if (
                not manager.has_python_implementation(call.tool_name)
                or not manager.is_replay_safe(call.tool_name)
            )
        )
    )
    if unavailable_tools:
        return DeterministicReplayResult(
            status="unavailable",
            is_valid=False,
            unavailable_tools=unavailable_tools,
        )

    original_state = copy.deepcopy(manager.get_api_state())
    issues: list[ReplayIssue] = []
    checked_calls = 0
    final_state_digest: str | None = None
    output_context: ObjectMap = {}

    try:
        manager.restore_api_state(copy.deepcopy(replay_state))
        for step in trajectory:
            replayable_calls = step.tool_calls
            if not replayable_calls:
                continue

            actual_pre_state = manager.get_api_state()
            if step.pre_state is not None:
                actual_subset = _state_subset(actual_pre_state, step.pre_state)
                if _normalized(actual_subset) != _normalized(step.pre_state):
                    issues.append(
                        ReplayIssue(
                            step_number=step.step_number,
                            tool_name=replayable_calls[0].tool_name,
                            check="pre_state",
                            expected=step.pre_state,
                            actual=actual_subset,
                        )
                    )

            execution_failed = False
            for call in replayable_calls:
                try:
                    resolved = _resolve_arguments(
                        call.arguments,
                        output_context,
                    )
                    if not is_object_map(resolved):
                        raise TypeError("resolved arguments are not an object")
                    actual_output = manager.invoke_python_tool(
                        call.tool_name,
                        resolved,
                    )
                except (
                    AttributeError,
                    IndexError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    issues.append(
                        ReplayIssue(
                            step_number=step.step_number,
                            tool_name=call.tool_name,
                            check="execution",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    execution_failed = True
                    break

                checked_calls += 1
                output_context[f"{call.tool_name}_output"] = copy.deepcopy(
                    actual_output
                )
                if _normalized(actual_output) != _normalized(call.output):
                    issues.append(
                        ReplayIssue(
                            step_number=step.step_number,
                            tool_name=call.tool_name,
                            check="output",
                            expected=call.output,
                            actual=actual_output,
                        )
                    )

            if execution_failed:
                break

            actual_post_state = manager.get_api_state()
            if step.post_state is not None:
                actual_subset = _state_subset(actual_post_state, step.post_state)
                if _normalized(actual_subset) != _normalized(step.post_state):
                    issues.append(
                        ReplayIssue(
                            step_number=step.step_number,
                            tool_name=replayable_calls[-1].tool_name,
                            check="post_state",
                            expected=step.post_state,
                            actual=actual_subset,
                        )
                    )
        final_state_digest = _digest(manager.get_api_state())
    finally:
        manager.restore_api_state(original_state)

    status = (
        "failed"
        if issues
        else "unavailable"
        if unavailable_tools
        else "verified"
    )
    return DeterministicReplayResult(
        status=status,
        is_valid=not issues and not unavailable_tools,
        checked_calls=checked_calls,
        unavailable_tools=unavailable_tools,
        issues=issues,
        final_state_digest=final_state_digest,
    )
