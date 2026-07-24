"""Core ToolManager initialization, implementation loading, and config lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

if __package__:
    from .tool_manager_types import (
        ConfigMap,
        JsonLLM,
        JsonModule,
        LoadingModule,
        ToolCallable,
        ToolInstances,
        ToolManagerProtocol,
        ToolSchema,
        get_attribute,
        get_callable_attribute,
        is_string_object_dict,
        require_config_map,
        set_attribute,
        string_field,
    )
else:
    from tool_manager_types import (
        ConfigMap,
        JsonLLM,
        JsonModule,
        LoadingModule,
        ToolCallable,
        ToolInstances,
        ToolManagerProtocol,
        ToolSchema,
        get_attribute,
        get_callable_attribute,
        is_string_object_dict,
        require_config_map,
        set_attribute,
        string_field,
    )

_MISSING = object()


def initialize_manager(
    manager: ToolManagerProtocol,
    llm: JsonLLM | None,
    tool_pool_path: str | None,
    tools: list[ToolCallable] | None,
    invocation_examples_path: str | None,
    use_config_pool: bool,
) -> None:
    """Initialize mutable manager state and load the requested tool sources."""
    manager.llm = llm
    manager.use_config_pool = use_config_pool
    manager.tool_schemas = []
    manager.tool_implementations = {}
    manager.python_tool_instances = {}
    manager.api_name_to_class_key = {}
    set_attribute(manager, "_canonical_configs", {})
    if tool_pool_path:
        _ = get_callable_attribute(manager, "_load_tools_from_file")(
            tool_pool_path
        )
    if tools:
        _ = get_callable_attribute(manager, "_load_tools_from_functions")(tools)
    if not manager.tool_schemas:
        _ = get_callable_attribute(manager, "_load_default_tools")()
    if invocation_examples_path:
        manager.load_python_tool_implementations(invocation_examples_path)


def load_python_tool_implementations(
    manager: ToolManagerProtocol,
    invocation_examples_path: str,
    path_type: type[Path],
    json_module: JsonModule,
    loading_module: LoadingModule,
    canonical_config_builder: Callable[[list[ToolSchema]], ConfigMap],
    instance_factory: Callable[[ConfigMap], ToolInstances],
) -> None:
    """Load invocation configs, BFCL API mappings, and dynamic instances."""
    path_obj = path_type(invocation_examples_path)
    if not path_obj.exists():
        print(f"Warning: Invocation examples file not found: {invocation_examples_path}")
        return
    examples = loading_module.read_invocation_examples(path_obj, json_module)
    canonical_configs = canonical_config_builder(examples)
    set_attribute(manager, "_canonical_configs", canonical_configs)
    manager.api_name_to_class_key = {}
    for tool_name, impl_info in manager.tool_implementations.items():
        if string_field(impl_info, "type") != "bfcl":
            continue
        data = impl_info.get("data")
        if not is_string_object_dict(data):
            continue
        class_key = string_field(data, "tool_name")
        if class_key:
            manager.api_name_to_class_key[tool_name] = class_key
    manager.python_tool_instances = instance_factory(canonical_configs)
    loaded = len(manager.python_tool_instances)
    mapped = len(manager.api_name_to_class_key)
    print(f"Loaded {loaded} Python tool classes with {mapped} api_name mappings")


def reset_python_tool_instances(
    manager: ToolManagerProtocol,
    random_config_factory: Callable[[], ConfigMap],
    full_initial_configs: ConfigMap,
    instance_factory: Callable[[ConfigMap], ToolInstances],
) -> None:
    """Recreate instances from the cached, pooled, or full initial config."""
    cached = get_attribute(manager, "_cached_initial_config", _MISSING)
    if cached is not _MISSING and cached is not None:
        config = require_config_map(cached, source="_cached_initial_config")
    elif manager.use_config_pool:
        config = random_config_factory()
    else:
        config = full_initial_configs
    manager.python_tool_instances = instance_factory(config)


def initialize_api_state(
    manager: ToolManagerProtocol,
    force_new: bool,
    random_config_factory: Callable[[], ConfigMap],
    full_initial_configs: ConfigMap,
) -> None:
    """Cache a realistic initial config and reset live tool instances."""
    cached = get_attribute(manager, "_cached_initial_config", _MISSING)
    if force_new or cached is _MISSING or cached is None:
        config = (
            random_config_factory()
            if manager.use_config_pool
            else full_initial_configs
        )
        set_attribute(manager, "_cached_initial_config", config)
    manager.reset_python_tool_instances()
