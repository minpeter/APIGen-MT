#!/usr/bin/env python3
"""
Simple example showing how to use the Magnet tool extraction scripts
to load and work with tool definitions programmatically.
"""

import json
from pathlib import Path
import sys

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from parse_bfcl import parse_bfcl_func_doc, discover_bfcl_classes
from tool_definition import ToolDefinition


def example_basic_usage():
    """Basic example: Load and filter tool definitions."""
    
    # Path to BFCL data (adjust as needed)
    bfcl_data_dir = Path("~/data/magnet_mt/data/BFCL_v3").expanduser()
    func_doc_dir = bfcl_data_dir / "multi_turn_func_doc"
    
    # Parse all function documentation
    print("Loading tool definitions...")
    definitions = parse_bfcl_func_doc(func_doc_dir, require_parameters=True)
    
    print(f"\nLoaded {len(definitions)} tool definitions")
    
    # Filter by category
    storage_tools = [d for d in definitions if d.category == "Storage"]
    print(f"Storage tools: {len(storage_tools)}")
    
    # Filter by tool name
    gorilla_tools = [d for d in definitions if d.tool_name == "gorilla_file_system"]
    print(f"Gorilla file system functions: {len(gorilla_tools)}")
    
    # Show first tool
    if gorilla_tools:
        print("\nFirst Gorilla function:")
        print(json.dumps(gorilla_tools[0].to_dict(), indent=2))
    
    return definitions


def example_search_tools():
    """Example: Search for tools by name or description."""
    
    bfcl_data_dir = Path("~/data/magnet_mt/data/BFCL_v3").expanduser()
    func_doc_dir = bfcl_data_dir / "multi_turn_func_doc"
    
    definitions = parse_bfcl_func_doc(func_doc_dir, require_parameters=True)
    
    # Search for tools with "file" in the description
    file_tools = [
        d for d in definitions 
        if "file" in d.api_description.lower()
    ]
    
    print(f"\nFound {len(file_tools)} tools mentioning 'file':")
    for tool in file_tools[:5]:
        print(f"  - {tool.api_name}: {tool.api_description[:50]}...")
    
    return file_tools


def example_parameter_analysis():
    """Example: Analyze parameter complexity of tools."""
    
    bfcl_data_dir = Path("~/data/magnet_mt/data/BFCL_v3").expanduser()
    func_doc_dir = bfcl_data_dir / "multi_turn_func_doc"
    
    definitions = parse_bfcl_func_doc(func_doc_dir, require_parameters=True)
    
    # Find tools with optional parameters
    tools_with_optional = [
        d for d in definitions 
        if d.parameters.optional
    ]
    
    print(f"\nTools with optional parameters: {len(tools_with_optional)}")
    
    # Find tools with many required parameters
    complex_tools = [
        d for d in definitions 
        if len(d.parameters.required) >= 2
    ]
    
    print(f"Tools with 2+ required parameters: {len(complex_tools)}")
    
    # Show examples
    if complex_tools:
        tool = complex_tools[0]
        print(f"\nExample: {tool.api_name}")
        print(f"  Required params: {', '.join(tool.parameters.required)}")
        if tool.parameters.optional:
            print(f"  Optional params: {', '.join(tool.parameters.optional)}")
    
    return complex_tools


def example_export_jsonl():
    """Example: Export tool definitions to JSONL format."""
    
    bfcl_data_dir = Path("~/data/magnet_mt/data/BFCL_v3").expanduser()
    func_doc_dir = bfcl_data_dir / "multi_turn_func_doc"
    
    definitions = parse_bfcl_func_doc(func_doc_dir, require_parameters=True)
    
    output_path = Path("exported_tools.jsonl")
    
    with output_path.open("w", encoding="utf-8") as f:
        for defn in definitions:
            f.write(json.dumps(defn.to_dict(), ensure_ascii=False) + "\n")
    
    print(f"\nExported {len(definitions)} tool definitions to {output_path}")
    
    return output_path


def example_group_by_category():
    """Example: Group tool definitions by category."""
    
    bfcl_data_dir = Path("~/data/magnet_mt/data/BFCL_v3").expanduser()
    func_doc_dir = bfcl_data_dir / "multi_turn_func_doc"
    
    definitions = parse_bfcl_func_doc(func_doc_dir, require_parameters=True)
    
    # Group by category
    from collections import defaultdict
    by_category = defaultdict(list)
    
    for defn in definitions:
        by_category[defn.category].append(defn)
    
    print("\nTools by category:")
    for category in sorted(by_category.keys()):
        tools = by_category[category]
        print(f"\n{category} ({len(tools)} tools):")
        for tool in tools[:3]:  # Show first 3
            print(f"  - {tool.api_name}")
        if len(tools) > 3:
            print(f"  ... and {len(tools) - 3} more")
    
    return by_category


def main():
    """Run all examples."""
    
    print("=" * 80)
    print("MAGNET TOOL EXTRACTION - USAGE EXAMPLES")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)
    example_basic_usage()
    
    print("\n" + "=" * 80)
    print("Example 2: Search Tools")
    print("=" * 80)
    example_search_tools()
    
    print("\n" + "=" * 80)
    print("Example 3: Parameter Analysis")
    print("=" * 80)
    example_parameter_analysis()
    
    print("\n" + "=" * 80)
    print("Example 4: Group by Category")
    print("=" * 80)
    example_group_by_category()
    
    print("\n" + "=" * 80)
    print("Example 5: Export to JSONL")
    print("=" * 80)
    example_export_jsonl()
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()