"""Manual command-line exercise for ``tool_manager.py``."""

from __future__ import annotations

import os
from collections.abc import Callable
from types import ModuleType

if __package__:
    from .tool_manager_types import (
        ToolManagerProtocol,
        ToolParams,
        is_json_llm,
        string_field,
    )
else:
    from tool_manager_types import (
        ToolManagerProtocol,
        ToolParams,
        is_json_llm,
        string_field,
    )

_DEMO_ERRORS: tuple[type[Exception], ...] = (Exception,)


def main(
    tool_manager_class: Callable[..., ToolManagerProtocol],
    llm_class: Callable[..., object],
    os_module: ModuleType,
    file_path: str,
) -> None:
    """Run the legacy ToolManager demonstration."""
    _ = os_module
    project_root = os.path.join(os.path.dirname(file_path), "..", "..")
    tool_pool_path = os.path.join(
        project_root, "magnet_tool_extraction", "bfcl_v3_tools_with_outputs.jsonl"
    )
    invocation_examples_path = os.path.join(
        project_root, "magnet_tool_extraction", "bfcl_v3_invocation_examples.jsonl"
    )
    print("Initializing ToolManager with tool pool + Python implementations...")
    llm = llm_class()
    if not is_json_llm(llm):
        raise TypeError("llm_class must create an object with json_output")
    tool_manager = tool_manager_class(
        llm=llm,
        tool_pool_path=tool_pool_path,
        invocation_examples_path=invocation_examples_path,
    )
    tools = tool_manager.get_tools_json_schema()
    print(f"Available tools ({len(tools)}):")
    for tool in tools[:5]:
        name = string_field(tool, "name")
        description = string_field(tool, "description")
        print(f" - {name}: {description[:50]}...")
    if len(tools) > 5:
        print(f" ... and {len(tools) - 5} more")

    python_tools = [
        name
        for tool in tools
        if (name := string_field(tool, "name"))
        and tool_manager.has_python_implementation(name)
    ]
    print(f"\nTools with Python implementations: {len(python_tools)}/{len(tools)}")

    test_tools: list[tuple[str, ToolParams]] = [
        ("add", {"a": 10, "b": 20}),
        ("create_ticket", {"title": "Test ticket", "priority": 3}),
        (
            "get_flight_cost",
            {
                "travel_from": "SFO",
                "travel_to": "JFK",
                "travel_date": "2024-06-15",
                "travel_class": "economy",
            },
        ),
    ]
    for test_tool, test_params in test_tools:
        print(f"\nTesting Python tool: {test_tool}")
        try:
            result = tool_manager.invoke_tool(test_tool, test_params)
            print(f"Result: {result}")
        except _DEMO_ERRORS as error:
            print(f"Error: {error}")
