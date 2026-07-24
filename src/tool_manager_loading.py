"""Tool-definition loading and BFCL schema normalization."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

if __package__:
    from .tool_manager_types import (
        JsonModule,
        SchemaBuilder,
        ToolCallable,
        ToolImplementation,
        ToolManagerProtocol,
        ToolSchema,
        get_callable_attribute,
        is_object_list,
        is_string_object_dict,
        mapping_field,
        string_field,
    )
else:
    from tool_manager_types import (
        JsonModule,
        SchemaBuilder,
        ToolCallable,
        ToolImplementation,
        ToolManagerProtocol,
        ToolSchema,
        get_callable_attribute,
        is_object_list,
        is_string_object_dict,
        mapping_field,
        string_field,
    )

_TYPE_MAPPING = {
    "STRING": "string",
    "NUMBER": "number",
    "INTEGER": "integer",
    "FLOAT": "number",
    "BOOLEAN": "boolean",
    "ARRAY": "array",
    "OBJECT": "object",
    "DATE": "string",
    "TUPLE": "array",
}


def read_json_or_jsonl(
    path: str, path_type: type[Path], json_module: JsonModule
) -> list[object]:
    """Read a JSON array/object or newline-delimited JSON definitions file."""
    path_obj = path_type(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Tool pool file not found: {path}")

    content = path_obj.read_text(encoding="utf-8").strip()
    try:
        parsed = json_module.loads(content)
        return parsed if is_object_list(parsed) else [parsed]
    except json_module.JSONDecodeError:
        tools_data: list[object] = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                tools_data.append(json_module.loads(line))
            except json_module.JSONDecodeError as error:
                print(f"Warning: Skipping invalid JSON line: {line[:50]}... Error: {error}")
        return tools_data


def read_invocation_examples(
    path_obj: Path, json_module: JsonModule
) -> list[ToolSchema]:
    """Read valid JSON objects from an invocation-examples JSONL file."""
    examples: list[ToolSchema] = []
    with path_obj.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                example = json_module.loads(line)
            except json_module.JSONDecodeError:
                continue
            if is_string_object_dict(example):
                examples.append(example)
    return examples


def normalize_bfcl_definition(
    tool_data: ToolSchema,
) -> tuple[str, ToolSchema, ToolImplementation] | None:
    """Normalize one BFCL definition to the public schema and implementation record."""
    tool_name = string_field(tool_data, "api_name") or string_field(
        tool_data, "name"
    )
    if not tool_name:
        print(f"Warning: Skipping tool without name: {tool_data}")
        return None

    parameters = mapping_field(tool_data, "parameters")
    properties = mapping_field(parameters, "properties")
    normalized_properties: dict[str, object] = {}
    for param_name, param_info in properties.items():
        if is_string_object_dict(param_info):
            param_type = string_field(param_info, "type", "string").upper()
            normalized_param: dict[str, object] = {
                "type": _TYPE_MAPPING.get(param_type, "string"),
                "description": string_field(param_info, "description"),
            }
            if "default" in param_info:
                normalized_param["default"] = param_info["default"]
        else:
            normalized_param = {"type": "string", "description": ""}
        normalized_properties[param_name] = normalized_param

    param_schema: ToolSchema = {
        "type": "object",
        "properties": normalized_properties,
        "required": parameters.get("required", []),
    }
    schema: ToolSchema = {
        "name": tool_name,
        "description": string_field(tool_data, "api_description")
        or string_field(tool_data, "tool_description"),
        "parameters": param_schema,
        "output_type": string_field(tool_data, "output_type", "unknown"),
        "output_description": string_field(tool_data, "output_description"),
        "output_schema": tool_data.get("output_schema", {}),
        "category": string_field(tool_data, "category", "Unknown"),
    }
    implementation: ToolImplementation = {"type": "bfcl", "data": tool_data}
    return tool_name, schema, implementation


def add_bfcl_definition(
    manager: ToolManagerProtocol, tool_data: ToolSchema
) -> None:
    """Normalize and add one BFCL definition to a manager."""
    normalized = normalize_bfcl_definition(tool_data)
    if normalized is None:
        return
    tool_name, schema, implementation = normalized
    manager.tool_schemas.append(schema)
    manager.tool_implementations[tool_name] = implementation


def load_tools_from_file(
    manager: ToolManagerProtocol,
    path: str,
    path_type: type[Path],
    json_module: JsonModule,
) -> None:
    """Load definitions while retaining the manager's per-definition seam."""
    for tool_data in read_json_or_jsonl(path, path_type, json_module):
        if not is_string_object_dict(tool_data):
            raise TypeError("A tool definition must be a JSON object")
        _ = get_callable_attribute(
            manager, "_add_tool_from_bfcl_definition"
        )(tool_data)


def load_tools_from_functions(
    manager: ToolManagerProtocol,
    tools: Iterable[ToolCallable],
    schema_builder: SchemaBuilder,
) -> None:
    """Add Python callables and their generated schemas to a manager."""
    for tool_func in tools:
        manager.tool_schemas.append(schema_builder(tool_func))
        manager.tool_implementations[tool_func.__name__] = {
            "type": "python",
            "func": tool_func,
        }
