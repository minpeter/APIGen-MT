"""Dynamic Python tool loading and direct invocation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from types import ModuleType
from typing import Never

if __package__:
    from .tool_manager_types import (
        ConfigMap,
        DeepCopier,
        InspectModule,
        JsonModule,
        ToolCallable,
        ToolInstances,
        ToolManagerProtocol,
        ToolParams,
        get_attribute,
        get_callable_attribute,
        is_string_object_dict,
        iterable_value,
    )
else:
    from tool_manager_types import (
        ConfigMap,
        DeepCopier,
        InspectModule,
        JsonModule,
        ToolCallable,
        ToolInstances,
        ToolManagerProtocol,
        ToolParams,
        get_attribute,
        get_callable_attribute,
        is_string_object_dict,
        iterable_value,
    )

_DYNAMIC_TOOL_ERRORS: tuple[type[Exception], ...] = (Exception,)


def create_python_tool_instances(
    canonical_configs: ConfigMap,
    tool_class_keys: Iterable[str],
    class_key_to_initial_config_key: Mapping[str, str],
    class_key_to_class_name: Mapping[str, str],
    import_module: Callable[[str], ModuleType],
    deepcopy: DeepCopier,
) -> ToolInstances:
    """Dynamically import and instantiate every configured Python tool class."""
    instances: ToolInstances = {}
    for class_key in tool_class_keys:
        config_key = class_key_to_initial_config_key[class_key]
        config = deepcopy(canonical_configs.get(config_key, {}))
        try:
            module = import_module(f"tools.{class_key}")
            tool_class = get_attribute(
                module, class_key_to_class_name[class_key]
            )
            if not callable(tool_class):
                raise TypeError(
                    f"{class_key_to_class_name[class_key]} is not callable"
                )
            instances[class_key] = tool_class(initial_config=config)
        except _DYNAMIC_TOOL_ERRORS as error:
            print(f"Warning: Could not instantiate {class_key}: {error}")
    return instances


def has_python_implementation(
    manager: ToolManagerProtocol, tool_name: str
) -> bool:
    """Return whether a mapped, instantiated Python implementation exists."""
    class_key = manager.api_name_to_class_key.get(tool_name)
    return class_key is not None and class_key in manager.python_tool_instances


def invoke_python_tool(
    manager: ToolManagerProtocol, tool_name: str, params: ToolParams
) -> object:
    """Invoke a mapped method, converting invocation errors to error outputs."""
    class_key = manager.api_name_to_class_key.get(tool_name)
    if class_key is None or class_key not in manager.python_tool_instances:
        raise ValueError(f"No Python implementation for tool '{tool_name}'")

    instance = manager.python_tool_instances[class_key]
    method = get_attribute(instance, tool_name)
    if not callable(method):
        _raise_missing_method(tool_name, class_key)

    try:
        coerced = get_callable_attribute(manager, "_coerce_params")(
            method, params
        )
        if not is_string_object_dict(coerced):
            raise TypeError("_coerce_params must return a string-keyed mapping")
        return method(**coerced)
    except _DYNAMIC_TOOL_ERRORS as error:
        return {"error": str(error)}


def _raise_missing_method(tool_name: str, class_key: str) -> Never:
    """Raise the legacy missing-method error outside the type-narrowing branch."""
    raise ValueError(f"Method '{tool_name}' not found on {class_key}")


def coerce_params(
    method: ToolCallable,
    params: ToolParams,
    inspect_module: InspectModule,
    json_module: JsonModule,
) -> ToolParams:
    """Coerce LLM-generated values to the callable's annotated primitive types."""
    signature = inspect_module.signature(method)
    coerced: ToolParams = {}
    for key, value in params.items():
        if key not in signature.parameters:
            coerced[key] = value
            continue

        annotation = get_attribute(signature.parameters[key], "annotation")
        if annotation is inspect_module.Parameter.empty:
            coerced[key] = value
            continue

        try:
            if annotation is int and not isinstance(value, int):
                coerced[key] = _to_int(value)
            elif annotation is float and not isinstance(value, float):
                coerced[key] = _to_float(value)
            elif annotation is bool and not isinstance(value, bool):
                coerced[key] = (
                    value.lower() in ("true", "1", "yes")
                    if isinstance(value, str)
                    else bool(value)
                )
            elif annotation is list and not isinstance(value, list):
                coerced[key] = _to_list(value, json_module)
            elif annotation is dict and not isinstance(value, dict):
                coerced[key] = (
                    json_module.loads(value)
                    if isinstance(value, str)
                    else value
                )
            else:
                coerced[key] = value
        except (ValueError, TypeError, json_module.JSONDecodeError):
            coerced[key] = value
    return coerced


def _to_int(value: object) -> int:
    """Apply the integer conversions supported by JSON tool arguments."""
    if isinstance(value, (str, bytes, bytearray, float)):
        return int(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to int")


def _to_float(value: object) -> float:
    """Apply the float conversions supported by JSON tool arguments."""
    if isinstance(value, (str, bytes, bytearray, int)):
        return float(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to float")


def _to_list(value: object, json_module: JsonModule) -> object:
    """Apply the legacy list conversion with a typed iterable boundary."""
    if isinstance(value, str):
        return json_module.loads(value)
    if not value:
        return []
    iterable = iterable_value(value)
    if iterable is None:
        raise TypeError(f"Cannot convert {type(value).__name__} to list")
    return list(iterable)
