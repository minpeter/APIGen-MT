#!/usr/bin/env python3
"""
Analyze BFCL tool pool to show top-5 most frequent tools per category.

Usage:
    python analyze_top_tools_by_category.py
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


def load_tool_pool(tool_pool_path: str) -> list:
    """Load tools from the JSONL file."""
    tools = []
    with open(tool_pool_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                tool = json.loads(line.strip())
                tools.append(tool)
            except json.JSONDecodeError:
                continue
    return tools


def extract_category(tool: dict) -> str:
    """Extract category from a tool definition."""
    # Try to get category from the tool data
    cat = tool.get('category', 'Unknown')
    
    if cat == 'Unknown':
        # Try to infer from source
        source = tool.get('source', 'Unknown')
        if 'live' in source.lower():
            cat = 'Live APIs'
        elif 'exec' in source.lower():
            cat = 'Executable'
        elif 'simple' in source.lower():
            cat = 'Simple Functions'
        elif 'multiple' in source.lower():
            cat = 'Multiple Functions'
        elif 'parallel' in source.lower():
            cat = 'Parallel Functions'
        else:
            cat = 'Other'
    
    return cat


def extract_tool_name(tool: dict) -> str:
    """Extract the tool/function name from a tool definition."""
    # BFCL format has 'tool_name' (system) and 'api_name' (function)
    if 'api_name' in tool and 'tool_name' in tool:
        # Combine tool_name and api_name for full context
        return f"{tool['tool_name']}.{tool['api_name']}"
    elif 'api_name' in tool:
        return tool['api_name']
    elif 'name' in tool:
        return tool['name']
    elif 'function' in tool and isinstance(tool['function'], dict):
        return tool['function'].get('name', 'unknown')
    elif 'function_name' in tool:
        return tool['function_name']
    else:
        return 'unknown'


def analyze_tools_by_category(tools: list) -> dict:
    """
    Analyze tools and group by category with frequency counts.
    
    Returns:
        Dictionary mapping category to Counter of tool names
    """
    category_tools = defaultdict(Counter)
    
    for tool in tools:
        category = extract_category(tool)
        tool_name = extract_tool_name(tool)
        category_tools[category][tool_name] += 1
    
    return category_tools


def print_top_tools_by_category(category_tools: dict, top_n: int = 5):
    """Print top-N tools for each category."""
    print("=" * 80)
    print(f"TOP-{top_n} MOST FREQUENT TOOLS BY CATEGORY")
    print("=" * 80)
    print()
    
    # Sort categories alphabetically
    sorted_categories = sorted(category_tools.keys())
    
    for category in sorted_categories:
        tool_counter = category_tools[category]
        total_tools = sum(tool_counter.values())
        unique_tools = len(tool_counter)
        
        print(f"📂 {category}")
        print(f"   Total: {total_tools} uses, {unique_tools} unique tools")
        print("-" * 80)
        
        # Get top N tools
        top_tools = tool_counter.most_common(top_n)
        
        for rank, (tool_name, count) in enumerate(top_tools, 1):
            percentage = (count / total_tools) * 100 if total_tools > 0 else 0
            bar_length = int(percentage / 2)  # Scale to max 50 chars
            bar = "█" * bar_length
            print(f"   {rank}. {tool_name:40s} {count:6d} ({percentage:5.1f}%) {bar}")
        
        print()


def generate_summary_stats(category_tools: dict):
    """Generate overall summary statistics."""
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()
    
    total_categories = len(category_tools)
    total_tools = sum(sum(counter.values()) for counter in category_tools.values())
    total_unique = sum(len(counter) for counter in category_tools.values())
    
    print(f"Total categories: {total_categories}")
    print(f"Total tool uses: {total_tools}")
    print(f"Total unique tools: {total_unique}")
    print(f"Average tools per category: {total_tools / total_categories if total_categories > 0 else 0:.1f}")
    print()
    
    # Top categories by tool count
    print("Top categories by total tool uses:")
    category_counts = [(cat, sum(counter.values())) for cat, counter in category_tools.items()]
    category_counts.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (category, count) in enumerate(category_counts[:10], 1):
        print(f"   {rank}. {category:30s} {count:6d}")


def export_to_json(category_tools: dict, output_path: str):
    """Export analysis to JSON file."""
    output_data = {}
    
    for category, tool_counter in category_tools.items():
        output_data[category] = {
            'total_uses': sum(tool_counter.values()),
            'unique_tools': len(tool_counter),
            'top_10_tools': [
                {'name': name, 'count': count}
                for name, count in tool_counter.most_common(10)
            ]
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Analysis exported to: {output_path}")


def main():
    """Main function."""
    # Configuration
    tool_pool_path = Path("~/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_all_tool_definitions.jsonl").expanduser()
    output_json = Path("~/data/APIGen-MT/magnet_tool_extraction/top_tools_by_category.json").expanduser()
    
    print("=" * 80)
    print("BFCL TOOL POOL ANALYSIS")
    print("=" * 80)
    print()
    
    # Check if file exists
    if not Path(tool_pool_path).exists():
        print(f"❌ Error: Tool pool file not found: {tool_pool_path}")
        return
    
    # Load tools
    print(f"📂 Loading tools from: {tool_pool_path}")
    tools = load_tool_pool(tool_pool_path)
    print(f"✅ Loaded {len(tools)} tools")
    print()
    
    # Analyze by category
    print("🔍 Analyzing tools by category...")
    category_tools = analyze_tools_by_category(tools)
    print(f"✅ Found {len(category_tools)} categories")
    print()
    
    # Print top tools per category
    print_top_tools_by_category(category_tools, top_n=5)
    
    # Print summary stats
    generate_summary_stats(category_tools)
    
    # Export to JSON
    export_to_json(category_tools, output_json)
    
    print()
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()