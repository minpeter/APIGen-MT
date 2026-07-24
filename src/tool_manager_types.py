"""Shared structural types for the split tool-manager implementation."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType
from typing import Protocol, TypeGuard

type ToolValue = object
type ToolParams = dict[str, ToolValue]
type ToolSchema = dict[str, ToolValue]
type ToolImplementation = dict[str, ToolValue]
type ToolCallable = Callable[..., ToolValue]
type ToolInstances = dict[str, object]
type Config = dict[str, object]
type ConfigMap = dict[str, Config]
type ApiState = dict[str, dict[str, object]]


class JsonLoader(Protocol):
    """The subset of ``json.loads`` used by the manager."""

    def __call__(self, value: str, /) -> object: ...


class JsonDumper(Protocol):
    """The subset of ``json.dumps`` used by the manager."""

    def __call__(
        self,
        value: object,
        /,
        *,
        indent: int | None = None,
        default: Callable[[object], object] | None = None,
    ) -> str: ...


class JsonModule(Protocol):
    """Structural view of the standard ``json`` module."""

    @property
    def loads(self) -> JsonLoader: ...

    @property
    def dumps(self) -> JsonDumper: ...

    @property
    def JSONDecodeError(self) -> type[json.JSONDecodeError]: ...


class InspectModule(Protocol):
    """Structural view of the standard ``inspect`` module."""

    @property
    def signature(self) -> Callable[[ToolCallable], inspect.Signature]: ...

    @property
    def Parameter(self) -> type[inspect.Parameter]: ...


class JsonLLM(Protocol):
    """LLM behavior required for simulation and semantic validation."""

    def json_output(
        self, prompt: str, *, reasoning: bool = True
    ) -> tuple[object, str]: ...


class SchemaBuilder(Protocol):
    """Callable that converts a Python function into a tool schema."""

    def __call__(self, function: ToolCallable) -> ToolSchema: ...


class ConfigFactory(Protocol):
    """Callable exported by the legacy configuration pool."""

    def __call__(self, seed: int | None = None) -> ConfigMap: ...


class DeepCopier(Protocol):
    """Generic deep-copy callable."""

    def __call__[Value](self, value: Value, /) -> Value: ...


class ToolManagerProtocol(Protocol):
    """Mutable and behavioral surface shared by the manager modules."""

    llm: JsonLLM | None
    use_config_pool: bool
    tool_schemas: list[ToolSchema]
    tool_implementations: dict[str, ToolImplementation]
    python_tool_instances: ToolInstances
    api_name_to_class_key: dict[str, str]

    def load_python_tool_implementations(
        self, invocation_examples_path: str
    ) -> None: ...

    def reset_python_tool_instances(self) -> None: ...

    def get_tools_by_category(self, category: str) -> list[ToolSchema]: ...

    def has_python_implementation(self, tool_name: str) -> bool: ...

    def is_replay_safe(self, tool_name: str) -> bool: ...

    def invoke_python_tool(self, tool_name: str, params: ToolParams) -> object: ...

    def get_tools_json_schema(self) -> list[ToolSchema]: ...

    def invoke_tool(self, tool_name: str, params: ToolParams) -> object: ...


class LoadingModule(Protocol):
    """Loading helper used by the lifecycle module."""

    def read_invocation_examples(
        self, path_obj: Path, json_module: JsonModule
    ) -> list[ToolSchema]: ...


class VirtualExecutor(Protocol):
    """Callable shape retained by the facade's private simulation seam."""

    def __call__(
        self,
        tool_name: str,
        params: ToolParams,
        schema: ToolSchema,
    ) -> object: ...


class AttributeGetter(Protocol):
    """Typed boundary for Python's dynamically typed ``getattr``."""

    def __call__(
        self, instance: object, name: str, default: object, /
    ) -> object: ...


class AttributeDictionaryGetter(Protocol):
    """Typed boundary for Python's dynamically typed ``vars``."""

    def __call__(self, instance: object, /) -> dict[str, object]: ...


class AttributeSetter(Protocol):
    """Typed boundary for Python's ``setattr``."""

    def __call__(
        self, instance: object, name: str, value: object, /
    ) -> None: ...


_get_attribute: AttributeGetter = getattr
_get_attributes: AttributeDictionaryGetter = vars
_set_attribute: AttributeSetter = setattr


def get_attribute(
    instance: object, name: str, default: object = None
) -> object:
    """Read a dynamic attribute without allowing ``Any`` to escape."""
    return _get_attribute(instance, name, default)


def get_attributes(instance: object) -> dict[str, object]:
    """Read an instance dictionary without allowing ``Any`` to escape."""
    return _get_attributes(instance)


def set_attribute(instance: object, name: str, value: object) -> None:
    """Set a dynamic attribute through the typed runtime boundary."""
    _set_attribute(instance, name, value)


def get_callable_attribute(instance: object, name: str) -> ToolCallable:
    """Return a required dynamic callback."""
    callback = get_attribute(instance, name)
    if not callable(callback):
        raise TypeError(f"{type(instance).__name__}.{name} must be callable")
    return callback


def is_object_list(value: object) -> TypeGuard[list[object]]:
    """Narrow a runtime list to a list whose values stay opaque."""
    return isinstance(value, list)


def is_string_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    """Narrow a dictionary from a boundary that promises string keys."""
    return isinstance(value, dict)


def is_config_map(value: object) -> TypeGuard[ConfigMap]:
    """Narrow a class-name to configuration mapping."""
    return is_string_object_dict(value) and all(
        is_string_object_dict(config) for config in value.values()
    )


def is_json_llm(value: object) -> TypeGuard[JsonLLM]:
    """Narrow an LLM instance at the command-line boundary."""
    return callable(get_attribute(value, "json_output"))


def is_schema_builder(value: object) -> TypeGuard[SchemaBuilder]:
    """Narrow the third-party schema generator at its dynamic boundary."""
    return callable(value)


def is_config_factory(value: object) -> TypeGuard[ConfigFactory]:
    """Narrow the dynamically imported configuration factory."""
    return callable(value)


def require_config_map(value: object, *, source: str) -> ConfigMap:
    """Return a validated configuration map or fail at the boundary."""
    if not is_config_map(value):
        raise TypeError(f"{source} must be a mapping of configuration mappings")
    return value


def require_config_list(value: object, *, source: str) -> list[Config]:
    """Return a validated list of configurations."""
    if not is_object_list(value):
        raise TypeError(f"{source} must be a list of configurations")
    configs: list[Config] = []
    for config in value:
        if not is_string_object_dict(config):
            raise TypeError(f"{source} contains a non-mapping configuration")
        configs.append(config)
    return configs


def string_field(
    mapping: dict[str, object], key: str, default: str = ""
) -> str:
    """Read a string field, using the supplied default for other values."""
    value = mapping.get(key, default)
    return value if isinstance(value, str) else default


def mapping_field(
    mapping: dict[str, object], key: str
) -> dict[str, object]:
    """Read a string-keyed mapping field, defaulting to an empty mapping."""
    value = mapping.get(key)
    return value if is_string_object_dict(value) else {}


def import_function(importer: Callable[[str], ModuleType], path: str) -> ModuleType:
    """Keep dynamic imports explicit and typed."""
    return importer(path)


def iterable_value(value: object) -> Iterable[object] | None:
    """Narrow an opaque value to the iterable accepted by legacy coercion."""
    return value if isinstance(value, Iterable) else None
