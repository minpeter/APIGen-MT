#!/usr/bin/env python3
"""
Extract output schemas from Python tool source code by parsing return statements.

Parses each Python tool class and extracts the dict keys returned by each method,
then adds output_schema to the BFCL tool definitions.

Usage:
    python add_output_schemas.py [--input bfcl_v3_tools_with_outputs.jsonl] [--output bfcl_v3_tools_with_schema.jsonl]
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def extract_return_keys_from_python_file(file_path: Path, class_name: str) -> Dict[str, List[str]]:
    """Parse a Python file and extract dict keys from return statements for a given class.

    Returns dict mapping method_name -> list of output dict keys.
    """
    if not file_path.exists():
        return {}

    with open(file_path) as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    schemas: Dict[str, List[str]] = {}

    # Find the class
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_name = item.name
                    return_keys: Set[str] = set()

                    for stmt in ast.walk(item):
                        # Look for dict literal returns: return { ... }
                        if isinstance(stmt, ast.Return) and stmt.value:
                            keys = _extract_dict_keys_from_expr(stmt.value)
                            return_keys.update(keys)

                    if return_keys:
                        schemas[method_name] = sorted(return_keys)

    return schemas


def _extract_dict_keys_from_expr(expr: ast.AST) -> List[str]:
    """Extract dict key names from an AST expression node."""
    keys = []

    if isinstance(expr, ast.Dict):
        for key in expr.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.append(key.value)
            elif isinstance(key, ast.Str):  # Python 3.7 compatibility
                keys.append(key.s)

    elif isinstance(expr, ast.Call):
        # Look for dict() constructor or other dict calls
        if isinstance(expr.func, ast.Name) and expr.func.id == 'dict':
            # dict() with keyword args: dict(a=1, b=2) -> keys are the keyword names
            for kw in expr.keywords:
                if kw.arg:
                    keys.append(kw.arg)

    elif isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        # dict1 | dict2 (Python 3.9+ dict merge)
        keys.extend(_extract_dict_keys_from_expr(expr.left))
        keys.extend(_extract_dict_keys_from_expr(expr.right))

    return keys


def extract_all_schemas_from_python_tools() -> Dict[str, Dict[str, List[str]]]:
    """Extract output schemas from all Python tool classes.

    Returns: dict mapping class_key -> { method_name: [output_keys] }
    """
    root = get_project_root()
    tools_dir = root / "tools"

    class_to_file = {
        'gorilla_file_system': 'gorilla_file_system.py',
        'math_api': 'math_api.py',
        'message_api': 'message_api.py',
        'posting_api': 'posting_api.py',
        'ticket_api': 'ticket_api.py',
        'trading_bot': 'trading_bot.py',
        'travel_booking': 'travel_booking.py',
        'vehicle_control': 'vehicle_control.py',
    }

    class_to_python_class = {
        'gorilla_file_system': 'GorillaFileSystem',
        'math_api': 'MathAPI',
        'message_api': 'MessageAPI',
        'posting_api': 'PostingAPI',
        'ticket_api': 'TicketAPI',
        'trading_bot': 'TradingBot',
        'travel_booking': 'TravelBooking',
        'vehicle_control': 'VehicleControlAPI',
    }

    all_schemas: Dict[str, Dict[str, List[str]]] = {}

    for class_key, filename in class_to_file.items():
        file_path = tools_dir / filename
        python_class = class_to_python_class[class_key]
        schemas = extract_return_keys_from_python_file(file_path, python_class)
        all_schemas[class_key] = schemas

    return all_schemas


def build_schema_json(schema: List[str]) -> Dict[str, Any]:
    """Build a JSON schema dict from a list of field names."""
    # Fields that are commonly success/error/status indicators
    status_fields = {'success', 'error', 'message', 'status', 'status_code', 'error_message', 'error_code'}

    properties = {}
    for field in schema:
        if field in status_fields:
            prop = {"type": "string", "description": f"Status field: {field}"}
        elif 'id' in field.lower() or field == 'booking_id' or field == 'transaction_id':
            prop = {"type": "string", "description": f"Identifier: {field}"}
        elif 'cost' in field.lower() or 'price' in field.lower() or 'balance' in field.lower() or 'amount' in field.lower():
            prop = {"type": "number", "description": f"Amount: {field}"}
        elif field.endswith('_status') or field in {'status', 'verification_status', 'booking_status'}:
            prop = {"type": "boolean", "description": f"Status: {field}"}
        elif field in {'expires_in', 'timestamp', 'timeout'}:
            prop = {"type": "integer", "description": f"Duration: {field}"}
        elif 'list' in field.lower() or 'array' in field.lower():
            prop = {"type": "array", "description": f"List: {field}"}
        else:
            prop = {"type": "string", "description": f"Output field: {field}"}

        properties[field] = prop

    return {"type": "object", "properties": properties}


def main():
    parser = argparse.ArgumentParser(description="Add output_schema to BFCL tool definitions")
    parser.add_argument(
        "--input",
        default="bfcl_v3_tools_with_outputs.jsonl",
        help="Input file (default: bfcl_v3_tools_with_outputs.jsonl)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file (default: <input>.with_schema.jsonl)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    input_path = script_dir / args.input
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    output_path = script_dir / (args.output or (args.input + ".with_schema.jsonl"))

    # Load tool definitions
    print(f"Loading tools from {input_path}...")
    tool_defs = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                tool_defs.append(json.loads(line))
    print(f"Loaded {len(tool_defs)} tool definitions")

    # Extract schemas from Python source
    print("Extracting output schemas from Python source...")
    all_schemas = extract_all_schemas_from_python_tools()

    # Flatten into api_name -> schema
    api_name_to_schema: Dict[str, Dict[str, Any]] = {}
    for class_key, method_schemas in all_schemas.items():
        for method_name, fields in method_schemas.items():
            api_name_to_schema[method_name] = build_schema_json(fields)

    # Count how many tools have schemas
    matched = sum(1 for t in tool_defs if t.get('api_name', '') in api_name_to_schema)
    print(f"Found schemas for {matched}/{len(tool_defs)} tools")

    # Show sample
    for api_name, schema in list(api_name_to_schema.items())[:5]:
        props = list(schema.get('properties', {}).keys())
        print(f"  {api_name}: {props}")

    # Merge schemas into tool definitions and write output
    for tool in tool_defs:
        api_name = tool.get('api_name', '')
        if api_name in api_name_to_schema:
            tool['output_schema'] = api_name_to_schema[api_name]

    with open(output_path, 'w') as f:
        for tool in tool_defs:
            f.write(json.dumps(tool) + '\n')

    print(f"\nWrote {len(tool_defs)} tools to {output_path}")


if __name__ == "__main__":
    main()