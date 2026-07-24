"""Core ToolManager schema lookup and invocation dispatch."""

from __future__ import annotations

if __package__:
    from .tool_manager_types import (
        ToolManagerProtocol,
        ToolParams,
        ToolSchema,
        VirtualExecutor,
        string_field,
    )
else:
    from tool_manager_types import (
        ToolManagerProtocol,
        ToolParams,
        ToolSchema,
        VirtualExecutor,
        string_field,
    )

_INVOCATION_ERRORS: tuple[type[Exception], ...] = (Exception,)


def get_categories(manager: ToolManagerProtocol) -> list[str]:
    """Return sorted unique categories across all loaded schemas."""
    return sorted(
        {string_field(tool, "category", "Unknown") for tool in manager.tool_schemas}
    )


def get_tools_by_category(
    manager: ToolManagerProtocol, category: str
) -> list[ToolSchema]:
    """Return schemas belonging to a category."""
    return [
        tool
        for tool in manager.tool_schemas
        if string_field(tool, "category") == category
    ]


def get_tool_category(
    manager: ToolManagerProtocol, tool_name: str
) -> str | None:
    """Return a tool category, or ``None`` when the tool is not loaded."""
    for tool in manager.tool_schemas:
        if string_field(tool, "name") == tool_name:
            category = tool.get("category")
            return category if isinstance(category, str) else None
    return None


def get_tools_json_schema(manager: ToolManagerProtocol) -> list[ToolSchema]:
    """Return the manager's live schema list."""
    return manager.tool_schemas


def get_tools_with_descriptions(
    manager: ToolManagerProtocol, category: str | None
) -> list[ToolSchema]:
    """Return all schemas, optionally restricted through the manager seam."""
    return manager.get_tools_by_category(category) if category else manager.tool_schemas


def get_tool_schema(
    manager: ToolManagerProtocol, tool_name: str
) -> ToolSchema:
    """Return a schema by name or raise with the available names."""
    tool_schema = _find_tool_schema(manager, tool_name)
    if tool_schema is not None:
        return tool_schema
    available_tools = [string_field(tool, "name") for tool in manager.tool_schemas]
    raise ValueError(
        f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}"
    )


def tool_exists(manager: ToolManagerProtocol, tool_name: str) -> bool:
    """Return whether a schema with this tool name is loaded."""
    return _find_tool_schema(manager, tool_name) is not None


def invoke_tool(
    manager: ToolManagerProtocol,
    tool_name: str,
    params: ToolParams,
    virtual_executor: VirtualExecutor,
) -> object:
    """Dispatch to a dynamic implementation, callable, or virtual executor."""
    tool_schema = _find_tool_schema(manager, tool_name)
    if tool_schema is None:
        available_tools = [
            string_field(tool, "name") for tool in manager.tool_schemas
        ]
        available = ", ".join(available_tools)
        raise ValueError(
            f"Tool '{tool_name}' not found. Available tools: {available}"
        )
    if manager.has_python_implementation(tool_name):
        return manager.invoke_python_tool(tool_name, params)
    impl_info = manager.tool_implementations.get(tool_name)
    if impl_info and string_field(impl_info, "type") == "python":
        function = impl_info.get("func")
        if callable(function):
            try:
                return function(**params)
            except _INVOCATION_ERRORS as error:
                return {"error": str(error)}
    return virtual_executor(tool_name, params, schema=tool_schema)


def _find_tool_schema(
    manager: ToolManagerProtocol, tool_name: str
) -> ToolSchema | None:
    """Find a loaded schema without changing the public missing-tool behavior."""
    return next(
        (
            tool
            for tool in manager.tool_schemas
            if string_field(tool, "name") == tool_name
        ),
        None,
    )
