"""API state snapshot, restore, cache, and filtering operations."""

from __future__ import annotations

from collections.abc import Callable

if __package__:
    from .tool_manager_types import (
        ApiState,
        JsonModule,
        ToolInstances,
        get_attribute,
        get_attributes,
        is_string_object_dict,
        set_attribute,
    )
else:
    from tool_manager_types import (
        ApiState,
        JsonModule,
        ToolInstances,
        get_attribute,
        get_attributes,
        is_string_object_dict,
        set_attribute,
    )

_MISSING = object()


def filter_api_state(
    full_state: ApiState,
    tool_names: list[str],
    relevant_class_keys: Callable[[list[str]], set[str]],
) -> ApiState:
    """Return state for relevant tool domains, or the original full state."""
    relevant = relevant_class_keys(tool_names)
    if not relevant:
        return full_state
    return {key: value for key, value in full_state.items() if key in relevant}


def get_api_state(
    python_tool_instances: ToolInstances, json_module: JsonModule
) -> ApiState:
    """Create a JSON-serializable deep snapshot of all live instances."""
    state: ApiState = {}
    for class_key, instance in python_tool_instances.items():
        raw = get_attributes(instance)
        try:
            serialized = json_module.dumps(raw, default=str)
            decoded = json_module.loads(serialized)
        except (TypeError, ValueError):
            fallback = {key: str(value) for key, value in raw.items()}
            decoded = json_module.loads(json_module.dumps(fallback))
        if not is_string_object_dict(decoded):
            raise TypeError(f"Serialized state for {class_key} is not an object")
        state[class_key] = decoded
    return state


def restore_api_state(
    python_tool_instances: ToolInstances,
    state: ApiState,
) -> None:
    """Apply a snapshot to matching live tool instances."""
    if not state:
        return

    for class_key, instance_state in state.items():
        if class_key not in python_tool_instances:
            continue
        instance = python_tool_instances[class_key]
        for key, value in instance_state.items():
            try:
                setattr(instance, key, value)
            except (AttributeError, TypeError) as error:
                print(f"  Warning: Could not restore {class_key}.{key}: {error}")


def clear_cached_config(manager: object) -> None:
    """Clear an existing cached initial config without creating the attribute."""
    if get_attribute(manager, "_cached_initial_config", _MISSING) is not _MISSING:
        set_attribute(manager, "_cached_initial_config", None)
