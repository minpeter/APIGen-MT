"""
Simple function schema generator for Python functions.
This module provides a basic implementation for generating JSON schemas from Python functions.
"""

import inspect
from typing import Any, Dict, Callable


def get_function_schema(func: Callable) -> Dict[str, Any]:
    """
    Generate a JSON schema from a Python function's signature and docstring.

    Args:
        func: The Python function to generate a schema for.

    Returns:
        A dictionary containing the function schema with name, description, and parameters.
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    # Build parameters schema
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        param_type = "string"  # Default type
        if param.annotation != inspect.Parameter.empty:
            type_name = str(param.annotation)
            if "int" in type_name:
                param_type = "integer"
            elif "float" in type_name:
                param_type = "number"
            elif "bool" in type_name:
                param_type = "boolean"
            elif "list" in type_name.lower():
                param_type = "array"
            elif "dict" in type_name.lower():
                param_type = "object"

        properties[param_name] = {
            "type": param_type,
            "description": ""
        }

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "name": func.__name__,
        "description": doc.split("\n")[0] if doc else "",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }