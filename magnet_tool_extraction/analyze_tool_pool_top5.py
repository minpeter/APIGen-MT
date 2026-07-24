#!/usr/bin/env python3
"""
Analyze BFCL tool pool showing top-5 most frequent tools per category.

ONLY uses: ~/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_all_tool_definitions.jsonl

Categories found in tool pool:
- Communication (10 tools)
- Events (9 tools)
- Finance (22 tools)
- Posting Api (14 tools)
- Science (17 tools)
- Storage (18 tools)
- Travel Booking (17 tools)
- Unknown (1,545 tools)
- Vehicle Control (22 tools)

Usage:
    python3 analyze_tool_pool_top5.py
"""

import json
from collections import Counter, defaultdict
from typing import Dict, List


def load_tool_pool(jsonl_path: str) -> List[dict]:
    """Load tool definitions from JSONL file."""
    tools = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                tool = json.loads(line.strip())
                tools.append(tool)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line: {e}")
                continue
    return tools


def extract_tools_by_category(tools: List[dict]) -> Dict[str, List[str]]:
    """Group tools by category."""
    category_tools = defaultdict(list)
    
    for tool in tools:
        # Extract category
        category = tool.get('category', 
                          tool.get('raw_definition', {}).get('category', 'Unknown'))
        
        # Extract tool name
        tool_name = tool.get('tool_name', '')
        api_name = tool.get('api_name', 'unknown')
        
        # Create full name
        if tool_name:
            full_name = f"{tool_name}.{api_name}"
        else:
            full_name = api_name
        
        category_tools[category].append(full_name)
    
    return category_tools


def print_top5_per_category(category_tools: Dict[str, List[str]], top_n: int = 5):
    """Print top-N tools for each category."""
    
    # Order categories logically
    category_order = [
        'Communication',
        'Events',
        'Finance',
        'Posting Api',
        'Science',
        'Storage',
        'Travel Booking',
        'Vehicle Control',
        'Unknown'
    ]
    
    print("=" * 80)
    print(f"TOP-{top_n} MOST FREQUENT TOOLS PER CATEGORY (FROM TOOL POOL)")
    print("=" * 80)
    print()
    
    for category in category_order:
        if category not in category_tools:
            continue
            
        tools = category_tools[category]
        counter = Counter(tools)
        total_unique = len(counter)
        total_instances = len(tools)
        
        print(f"📂 {category}")
        print("-" * 80)
        print(f"   Total tools: {total_unique} unique")
        print()
        
        # Show top N
        top_tools = counter.most_common(top_n)
        for rank, (name, count) in enumerate(top_tools, 1):
            # Since each tool appears only once in definitions, count is always 1
            print(f"   {rank}. {name}")
        
        # If there are more tools than shown
        if total_unique > top_n:
            remaining = total_unique - top_n
            print(f"   ... and {remaining} more tools")
        
        print()
    
    # Show any categories not in our ordered list
    other_categories = [c for c in category_tools.keys() if c not in category_order]
    if other_categories:
        print("=" * 80)
        print("OTHER CATEGORIES")
        print("=" * 80)
        print()
        for category in sorted(other_categories):
            tools = category_tools[category]
            counter = Counter(tools)
            print(f"📂 {category}: {len(counter)} tools")
            for rank, (name, count) in enumerate(counter.most_common(3), 1):
                print(f"   {rank}. {name}")
            print()


def generate_summary(category_tools: Dict[str, List[str]]):
    """Generate summary statistics."""
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    total_tools = sum(len(tools) for tools in category_tools.values())
    total_categories = len(category_tools)
    
    print(f"Total tools in pool: {total_tools}")
    print(f"Total categories: {total_categories}")
    print()
    
    print("Tools per category:")
    for category in sorted(category_tools.keys()):
        count = len(category_tools[category])
        bar_length = int(count / total_tools * 50)
        bar = "█" * bar_length
        print(f"  {category:30s}: {count:4d} {bar}")
    
    print()
    print("=" * 80)


def export_to_json(category_tools: Dict[str, List[str]], output_path: str):
    """Export analysis results to JSON."""
    
    results = {
        'metadata': {
            'total_tools': sum(len(t) for t in category_tools.values()),
            'total_categories': len(category_tools),
            'source': 'bfcl_v3_all_tool_definitions.jsonl'
        },
        'categories': {}
    }
    
    for category, tools in category_tools.items():
        counter = Counter(tools)
        results['categories'][category] = {
            'total_tools': len(counter),
            'top_tools': [
                {
                    'rank': rank,
                    'name': name,
                    'count': count
                }
                for rank, (name, count) in enumerate(counter.most_common(10), 1)
            ]
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported results to: {output_path}")


def export_to_markdown(category_tools: Dict[str, List[str]], output_path: str):
    """Export results to markdown format."""
    
    md = "# BFCL Tool Pool Analysis: Top-5 Tools per Category\n\n"
    md += f"**Source**: `bfcl_v3_all_tool_definitions.jsonl`\n\n"
    md += f"**Total Tools**: {sum(len(t) for t in category_tools.values())}\n\n"
    md += f"**Categories**: {len(category_tools)}\n\n"
    md += "---\n\n"
    
    category_order = [
        'Communication',
        'Events', 
        'Finance',
        'Posting Api',
        'Science',
        'Storage',
        'Travel Booking',
        'Vehicle Control',
        'Unknown'
    ]
    
    for category in category_order:
        if category not in category_tools:
            continue
            
        tools = category_tools[category]
        counter = Counter(tools)
        
        md += f"## {category}\n\n"
        md += f"**Total Tools**: {len(counter)}\n\n"
        md += "**Top-5:**\n\n"
        
        for rank, (name, count) in enumerate(counter.most_common(5), 1):
            md += f"{rank}. `{name}`\n"
        
        md += "\n---\n\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)
    
    print(f"✅ Exported markdown to: {output_path}")


def main():
    """Main function."""
    # Configuration - ONLY use this file
    tool_pool_path = Path("~/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_all_tool_definitions.jsonl").expanduser()
    output_json = Path("~/data/APIGen-MT/magnet_tool_extraction/tool_pool_top5_analysis.json").expanduser()
    output_md = Path("~/data/APIGen-MT/magnet_tool_extraction/TOOL_POOL_TOP5_BY_CATEGORY.md").expanduser()
    
    print("=" * 80)
    print("BFCL TOOL POOL ANALYSIS")
    print("=" * 80)
    print()
    print(f"Analyzing: {tool_pool_path}")
    print()
    
    # Load tool pool
    print("Loading tool definitions...")
    tools = load_tool_pool(tool_pool_path)
    print(f"✅ Loaded {len(tools)} tool definitions")
    print()
    
    # Extract tools by category
    print("Grouping tools by category...")
    category_tools = extract_tools_by_category(tools)
    print(f"✅ Found {len(category_tools)} categories")
    print()
    
    # Print top-5 per category
    print_top5_per_category(category_tools, top_n=5)
    
    # Generate summary
    generate_summary(category_tools)
    
    # Export results
    print()
    export_to_json(category_tools, output_json)
    export_to_markdown(category_tools, output_md)
    
    print()
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()