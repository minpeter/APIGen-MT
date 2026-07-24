"""Public facade for loading, invoking, simulating, and snapshotting tools."""

from __future__ import annotations

import copy
import datetime
import importlib
import inspect
import json
import os
from pathlib import Path

if __package__:
    from . import tool_manager_dependencies as _dependencies
else:
    import tool_manager_dependencies as _dependencies

LLMClient = _dependencies.LLMClient
_bootstrap = _dependencies.bootstrap
_catalog = _dependencies.catalog
_cli = _dependencies.cli
_core_lifecycle = _dependencies.core_lifecycle
_core_lookup = _dependencies.core_lookup
_loading = _dependencies.loading
_output_validation = _dependencies.output_validation
_python = _dependencies.python_tools
_simulation = _dependencies.simulation
_state = _dependencies.state
_types = _dependencies.types

type ApiState = _types.ApiState
type ConfigMap = _types.ConfigMap
type JsonLLM = _types.JsonLLM
type ToolCallable = _types.ToolCallable
type ToolInstances = _types.ToolInstances
type ToolParams = _types.ToolParams
type ToolSchema = _types.ToolSchema

MESSAGE_CONFIGS = _bootstrap.MESSAGE_CONFIGS
generate_random_config = _bootstrap.generate_random_config
get_function_schema = _bootstrap.get_function_schema
CLASS_KEY_TO_CLASS_NAME = _bootstrap.CLASS_KEY_TO_CLASS_NAME
CLASS_KEY_TO_INITIAL_CONFIG_KEY = _bootstrap.CLASS_KEY_TO_INITIAL_CONFIG_KEY
TOOL_CLASS_KEYS = _bootstrap.TOOL_CLASS_KEYS
TOOL_NAME_TO_CLASS_KEY = _bootstrap.TOOL_NAME_TO_CLASS_KEY
FULL_INITIAL_CONFIGS = _bootstrap.FULL_INITIAL_CONFIGS


def get_relevant_class_keys(tool_names: list[str]) -> set[str]:
    """Return the set of class keys whose APIs are used by the tool names."""
    return _catalog.get_relevant_class_keys(tool_names, TOOL_NAME_TO_CLASS_KEY)


def filter_api_state(full_state: ApiState, tool_names: list[str]) -> ApiState:
    """Return relevant class-key entries, or the full state when none match."""
    return _state.filter_api_state(full_state, tool_names, get_relevant_class_keys)


def get_canonical_initial_configs(examples: list[ToolSchema]) -> ConfigMap:
    """Extract the most representative (largest) initial config per class."""
    return _catalog.get_canonical_initial_configs(examples)


def create_python_tool_instances(
    canonical_configs: ConfigMap,
) -> ToolInstances:
    """Dynamically instantiate all configured Python tool classes."""
    return _python.create_python_tool_instances(
        canonical_configs,
        TOOL_CLASS_KEYS,
        CLASS_KEY_TO_INITIAL_CONFIG_KEY,
        CLASS_KEY_TO_CLASS_NAME,
        importlib.import_module,
        copy.deepcopy,
    )


def build_api_name_to_class_key_map(
    tools_data: list[ToolSchema],
) -> dict[str, str]:
    """Build an API-name to class-key mapping from BFCL definitions."""
    return _catalog.build_api_name_to_class_key_map(tools_data)


DEFAULT_TOOLS: list[ToolCallable] = []


class ToolManager:
    """Manage tools loaded from BFCL definitions or Python callables."""

    llm: JsonLLM | None
    use_config_pool: bool
    tool_schemas: list[ToolSchema]
    tool_implementations: dict[str, dict[str, object]]
    python_tool_instances: ToolInstances
    api_name_to_class_key: dict[str, str]
    _canonical_configs: ConfigMap

    def __init__(
        self,
        llm: JsonLLM | None,
        tool_pool_path: str | None = None,
        tools: list[ToolCallable] | None = None,
        invocation_examples_path: str | None = None,
        use_config_pool: bool = True,
    ) -> None:
        self.llm = llm
        self.use_config_pool = use_config_pool
        self.tool_schemas = []
        self.tool_implementations = {}
        self.python_tool_instances = {}
        self.api_name_to_class_key = {}
        self._canonical_configs = {}
        _core_lifecycle.initialize_manager(
            self, llm, tool_pool_path, tools, invocation_examples_path, use_config_pool
        )

    def load_python_tool_implementations(self, invocation_examples_path: str) -> None:
        """Load invocation configs, API mappings, and dynamic tool instances."""
        _core_lifecycle.load_python_tool_implementations(
            self,
            invocation_examples_path,
            Path,
            json,
            _loading,
            get_canonical_initial_configs,
            create_python_tool_instances,
        )

    def reset_python_tool_instances(self) -> None:
        """Reset every Python tool instance from cached, pooled, or full config."""
        _core_lifecycle.reset_python_tool_instances(
            self,
            generate_random_config,
            FULL_INITIAL_CONFIGS,
            create_python_tool_instances,
        )

    def initialize_api_state(self, force_new: bool = False) -> None:
        """Initialize realistic API state, optionally choosing a fresh config."""
        _core_lifecycle.initialize_api_state(
            self, force_new, generate_random_config, FULL_INITIAL_CONFIGS
        )

    def clear_cached_config(self) -> None:
        """Clear the cached initial config when the cache attribute exists."""
        _state.clear_cached_config(self)

    def get_api_state(self) -> ApiState:
        """Snapshot all live Python tool state as JSON-serializable values."""
        return _state.get_api_state(self.python_tool_instances, json)

    def restore_api_state(self, state: ApiState) -> None:
        """Restore matching live Python tool instances from a snapshot."""
        _state.restore_api_state(self.python_tool_instances, state)

    def has_python_implementation(self, tool_name: str) -> bool:
        """Check whether a tool has an instantiated Python implementation."""
        return _python.has_python_implementation(self, tool_name)

    def is_replay_safe(self, tool_name: str) -> bool:
        return (
            self.api_name_to_class_key.get(tool_name)
            != "gorilla_file_system"
        )

    def invoke_python_tool(self, tool_name: str, params: ToolParams) -> object:
        """Invoke a Python tool implementation directly."""
        return _python.invoke_python_tool(self, tool_name, params)

    @staticmethod
    def _coerce_params(
        method: ToolCallable, params: ToolParams
    ) -> ToolParams:
        """Coerce parameter values to the callable's annotated primitive types."""
        return _python.coerce_params(method, params, inspect, json)

    def _load_tools_from_file(self, path: str) -> None:
        """Load tool definitions from JSON or JSONL."""
        _loading.load_tools_from_file(self, path, Path, json)

    def _add_tool_from_bfcl_definition(self, tool_data: ToolSchema) -> None:
        """Normalize and add one BFCL-style tool definition."""
        _loading.add_bfcl_definition(self, tool_data)

    def _load_tools_from_functions(self, tools: list[ToolCallable]) -> None:
        """Load schemas and implementations from Python functions."""
        _loading.load_tools_from_functions(self, tools, get_function_schema)

    def _load_default_tools(self) -> None:
        """Keep an empty tool set when no explicit source is configured."""
        _loading.load_tools_from_functions(self, DEFAULT_TOOLS, get_function_schema)

    def get_categories(self) -> list[str]:
        """Get sorted unique categories across all tools."""
        return _core_lookup.get_categories(self)

    def get_tools_by_category(self, category: str) -> list[ToolSchema]:
        """Get tool schemas belonging to a category."""
        return _core_lookup.get_tools_by_category(self, category)

    def get_tool_category(self, tool_name: str) -> str | None:
        """Get a tool's category, or ``None`` when it is not loaded."""
        return _core_lookup.get_tool_category(self, tool_name)

    def get_tools_json_schema(self) -> list[ToolSchema]:
        """Get all tool schemas in JSON format."""
        return _core_lookup.get_tools_json_schema(self)

    def get_tools_with_descriptions(
        self, category: str | None = None
    ) -> list[ToolSchema]:
        """Get all schemas, optionally restricted to a category."""
        return _core_lookup.get_tools_with_descriptions(self, category)

    def get_tool_schema(self, tool_name: str) -> ToolSchema:
        """Get a schema by name or raise with the available names."""
        return _core_lookup.get_tool_schema(self, tool_name)

    def tool_exists(self, tool_name: str) -> bool:
        """Return whether a schema with this tool name is loaded."""
        return _core_lookup.tool_exists(self, tool_name)

    def invoke_tool(self, tool_name: str, params: ToolParams) -> object:
        """Invoke Python code when available, otherwise simulate the tool."""
        return _core_lookup.invoke_tool(
            self, tool_name, params, self.__virtual_tool_executor
        )

    def __virtual_tool_executor(
        self, tool_name: str, params: ToolParams, schema: ToolSchema
    ) -> object:
        """Simulate tool execution through the configured LLM."""
        return _simulation.virtual_tool_executor(self, tool_name, params, schema)

    def _validate_tool_output(
        self,
        tool_name: str,
        output: object,
        expected_type: str,
        output_description: str,
    ) -> object:
        """Validate and coerce a simulated output."""
        return _output_validation.validate_tool_output(
            self, tool_name, output, expected_type, output_description, json
        )

    def _build_output_guidance(
        self, output_type: str, output_description: str
    ) -> str:
        """Build enhanced output guidance with examples and field requirements."""
        return _output_validation.build_output_guidance(
            output_type, output_description
        )

    def _get_output_format_instructions(
        self, output_type: str, output_description: str = ""
    ) -> str:
        """Get output formatting instructions for a declared type."""
        return _output_validation.get_output_format_instructions(
            output_type, output_description
        )

    def _build_simulation_prompt(
        self,
        tool_name: str,
        params: ToolParams,
        schema: ToolSchema,
        output_guidance: str,
    ) -> str:
        """Build the enhanced simulation prompt with examples."""
        local_now = datetime.datetime.now(datetime.UTC).astimezone().replace(tzinfo=None)
        return _simulation.build_simulation_prompt(
            self,
            tool_name,
            params,
            schema,
            output_guidance,
            local_now,
            json,
        )

    def _get_few_shot_examples(self, tool_name: str, schema: ToolSchema) -> str:
        """Generate few-shot examples based on the output type."""
        return _simulation.get_few_shot_examples(tool_name, schema)

    def _is_output_valid(
        self, output: object, expected_type: str, expected_description: str
    ) -> bool:
        """Check whether an output matches its declared basic type."""
        return _output_validation.is_output_valid(
            output, expected_type, expected_description
        )

    def _get_default_output(self, output_type: str) -> object:
        """Return the fallback output for a declared type."""
        return _output_validation.get_default_output(output_type)


if __name__ == "__main__":
    _cli.main(ToolManager, LLMClient, os, __file__)
