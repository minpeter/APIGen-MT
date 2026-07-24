"""
Canonical tool definition dataclass matching the Magnet paper's format.

The template used in the paper (Appendix A / Section 3.6) is:
{
    "category": "",
    "tool_name": "",
    "tool_description": "",
    "api_name": "",
    "api_description": "",
    "parameters": {
        "type": "dict",
        "properties": {},
        "required": [],
        "optional": []
    }
}
"""

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ToolParameters:
    """Parameters schema for a single API/function."""
    properties: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    optional: list[str] = field(default_factory=list)
    type: str = "dict"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "properties": self.properties,
            "required": self.required,
            "optional": self.optional,
        }


@dataclass
class ToolDefinition:
    """
    A single API/function definition in the Magnet canonical format.

    Each row represents one API endpoint (function), with the parent tool's
    metadata denormalised into 'tool_name' / 'tool_description'.
    """
    category: str
    tool_name: str
    tool_description: str
    api_name: str
    api_description: str
    parameters: ToolParameters = field(default_factory=ToolParameters)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "tool_name": self.tool_name,
            "tool_description": self.tool_description,
            "api_name": self.api_name,
            "api_description": self.api_description,
            "parameters": self.parameters.to_dict(),
        }
