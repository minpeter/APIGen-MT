#!/usr/bin/env python3
"""
Extract ALL tool definitions from BFCL_v3 dataset (not just multi-turn).
This extracts tools from:
1. multi_turn_func_doc directory (129 tools)
2. Embedded tool definitions in test files (simple, multiple, live, exec)
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Set

# Import the parser from our existing code
import sys
sys.path.insert(0, str(Path(__file__).parent))
from parse_bfcl import parse_bfcl_func_doc


def extract_from_multi_turn_func_doc(output_path: str = None):
    """Extract tools from multi_turn_func_doc directory."""
    
    func_doc_dir = Path("/home/ishalyminov/data/magnet_mt/data/BFCL_v3/multi_turn_func_doc")
    
    # Parse all function documentation
    definitions = parse_bfcl_func_doc(func_doc_dir, require_parameters=False)
    
    result = []
    for defn in definitions:
        result.append({
            'source': 'multi_turn_func_doc',
            'category': defn.category,
            'tool_name': defn.tool_name,
            'api_name': defn.api_name,
            'api_description': defn.api_description,
            'parameters': defn.parameters.to_dict() if defn.parameters else {},
            'raw_definition': defn.to_dict()
        })
    
    print(f"✅ Extracted {len(result)} tools from multi_turn_func_doc")
    return result


def extract_from_test_files(output_path: str = None):
    """Extract tool definitions embedded in test files."""
    
    data_dir = Path("/home/ishalyminov/data/magnet_mt/data/BFCL_v3")
    
    # Test files with embedded tool definitions
    test_files = [
        "BFCL_v3_simple.json",
        "BFCL_v3_multiple.json",
        "BFCL_v3_parallel.json",
        "BFCL_v3_parallel_multiple.json",
        "BFCL_v3_live_simple.json",
        "BFCL_v3_live_multiple.json",
        "BFCL_v3_live_parallel.json",
        "BFCL_v3_live_parallel_multiple.json",
        "BFCL_v3_live_relevance.json",
        "BFCL_v3_live_irrelevance.json",
        "BFCL_v3_exec_simple.json",
        "BFCL_v3_exec_multiple.json",
        "BFCL_v3_exec_parallel.json",
        "BFCL_v3_exec_parallel_multiple.json",
    ]
    
    # Track unique tools by name
    tools_by_name = {}
    tool_occurrences = defaultdict(int)
    
    for test_file in test_files:
        path = data_dir / test_file
        if not path.exists():
            print(f"⚠️  Skipping {test_file} - not found")
            continue
        
        print(f"📄 Processing {test_file}...")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Extract from 'function' field
                    functions = data.get('function', [])
                    if not functions:
                        functions = data.get('functions', [])
                    if not functions:
                        functions = data.get('tools', [])
                    
                    for func_def in functions:
                        func_name = func_def.get('name', '')
                        if not func_name:
                            continue
                        
                        tool_occurrences[func_name] += 1
                        
                        # Store first occurrence (most complete definition)
                        if func_name not in tools_by_name:
                            tools_by_name[func_name] = {
                                'source': test_file,
                                'api_name': func_name,
                                'api_description': func_def.get('description', ''),
                                'parameters': func_def.get('parameters', {}),
                                'raw_definition': func_def
                            }
                
                except json.JSONDecodeError as e:
                    print(f"  ⚠️  Error parsing line {line_no}: {e}")
                    continue
    
    result = list(tools_by_name.values())
    print(f"✅ Extracted {len(result)} unique tools from test files")
    return result


def merge_tool_definitions(tools_list1, tools_list2):
    """Merge tool definitions from multiple sources."""
    
    all_tools = {}
    
    # Add from first list
    for tool in tools_list1:
        api_name = tool['api_name']
        if api_name not in all_tools:
            all_tools[api_name] = tool
    
    # Add from second list
    for tool in tools_list2:
        api_name = tool['api_name']
        if api_name not in all_tools:
            all_tools[api_name] = tool
    
    return list(all_tools.values())


def analyze_tools(tools):
    """Analyze the extracted tools."""
    
    print("\n" + "=" * 80)
    print("TOOL ANALYSIS")
    print("=" * 80)
    
    print(f"\nTotal unique tools: {len(tools)}")
    
    # By source
    by_source = defaultdict(int)
    for tool in tools:
        by_source[tool['source']] += 1
    
    print(f"\nBy source:")
    for source, count in sorted(by_source.items()):
        print(f"  {source:40} {count:4} tools")
    
    # By category (for multi_turn tools)
    by_category = defaultdict(int)
    for tool in tools:
        if 'category' in tool and tool['category']:
            by_category[tool['category']] += 1
    
    if by_category:
        print(f"\nBy category (multi_turn only):")
        for cat, count in sorted(by_category.items()):
            print(f"  {cat:30} {count:4} tools")
    
    # Parameter analysis
    with_params = sum(1 for t in tools if t.get('parameters'))
    with_required = sum(1 for t in tools if t.get('parameters', {}).get('required'))
    with_optional = sum(1 for t in tools if t.get('parameters', {}).get('properties'))
    
    print(f"\nParameter statistics:")
    print(f"  With parameters field: {with_params}")
    print(f"  With required params: {with_required}")
    print(f"  With properties: {with_optional}")
    
    # Sample tools
    print(f"\nSample tools (first 10):")
    for i, tool in enumerate(tools[:10], 1):
        print(f"  {i}. {tool['api_name']}")
    
    return by_source, by_category


def main():
    """Main extraction function."""
    
    print("=" * 80)
    print("BFCL_v3 COMPLETE TOOL EXTRACTION")
    print("=" * 80)
    
    # Step 1: Extract from multi_turn_func_doc
    print("\n📦 Step 1: Extracting from multi_turn_func_doc...")
    multi_turn_tools = extract_from_multi_turn_func_doc()
    
    # Step 2: Extract from test files
    print("\n📦 Step 2: Extracting from test files...")
    test_file_tools = extract_from_test_files()
    
    # Step 3: Merge all tools
    print("\n📦 Step 3: Merging tool definitions...")
    all_tools = merge_tool_definitions(multi_turn_tools, test_file_tools)
    
    # Step 4: Analyze
    analyze_tools(all_tools)
    
    # Step 5: Write to file
    output_path = "bfcl_v3_all_tool_definitions.jsonl"
    print(f"\n💾 Writing to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for tool in all_tools:
            f.write(json.dumps(tool, ensure_ascii=False) + '\n')
    
    print(f"✅ Written {len(all_tools)} tool definitions")
    
    # Step 6: Create summary
    summary_path = "bfcl_v3_all_tools_summary.md"
    print(f"\n📄 Creating summary at {summary_path}...")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# BFCL_v3 Complete Tool Extraction\n\n")
        f.write(f"**Total Unique Tools**: {len(all_tools)}\n\n")
        
        f.write("## Sources\n\n")
        by_source = defaultdict(int)
        for tool in all_tools:
            by_source[tool['source']] += 1
        
        for source, count in sorted(by_source.items()):
            f.write(f"- **{source}**: {count} tools\n")
        
        f.write("\n## Categories (Multi-turn)\n\n")
        by_category = defaultdict(int)
        for tool in all_tools:
            if 'category' in tool and tool['category']:
                by_category[tool['category']] += 1
        
        for cat, count in sorted(by_category.items()):
            f.write(f"- **{cat}**: {count} tools\n")
        
        f.write("\n## Sample Tools\n\n")
        f.write("### First 20 Tools\n\n```\n")
        for i, tool in enumerate(all_tools[:20], 1):
            f.write(f"{i:3}. {tool['api_name']}\n")
        f.write("```\n")
        
        # Show diversity
        f.write(f"\n### Tools from 'live' files (real-world APIs)\n\n```\n")
        live_tools = [t for t in all_tools if 'live' in t['source']][:20]
        for i, tool in enumerate(live_tools, 1):
            f.write(f"{i:3}. {tool['api_name']}\n")
        f.write("```\n")
    
    print(f"✅ Summary created")
    
    print("\n" + "=" * 80)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  1. {output_path} - All tool definitions")
    print(f"  2. {summary_path} - Summary report")
    print("=" * 80)


if __name__ == "__main__":
    main()