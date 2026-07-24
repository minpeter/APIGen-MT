#!/usr/bin/env python3
"""
Analyze BFCL tool usage frequency from actual test data.
Shows both:
1. Tools available in the pool (unique definitions)
2. Tools actually used in test cases (invocation frequency)

Usage:
    python analyze_tool_usage_frequency.py
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


def load_jsonl(file_path: str) -> list:
    """Load data from JSONL file."""
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                items.append(item)
            except json.JSONDecodeError:
                continue
    return items


def extract_tool_invocations(test_case: dict) -> list:
    """Extract all tool invocations from a test case."""
    invocations = []
    
    # Try different possible locations for ground truth
    if 'ground_truth' in test_case:
        gt = test_case['ground_truth']
        if isinstance(gt, list):
            for item in gt:
                if isinstance(item, dict):
                    tool_name = item.get('name') or item.get('api_name') or item.get('tool_name', 'unknown')
                    invocations.append(tool_name)
    
    # Try to find in 'function' field
    if 'function' in test_case:
        func = test_case['function']
        if isinstance(func, list):
            for item in func:
                if isinstance(item, dict):
                    tool_name = item.get('name') or item.get('api_name', 'unknown')
                    invocations.append(tool_name)
    
    # Try to find in answers
    if 'answers' in test_case:
        answers = test_case['answers']
        if isinstance(answers, list):
            for item in answers:
                if isinstance(item, dict):
                    tool_name = item.get('name') or item.get('api_name', 'unknown')
                    invocations.append(tool_name)
    
    return invocations


def analyze_tool_pool(tool_pool_path: str) -> dict:
    """Analyze unique tools available in the pool."""
    tools = load_jsonl(tool_pool_path)
    
    category_tools = defaultdict(list)
    
    for tool in tools:
        category = tool.get('category', 'Unknown')
        tool_name = tool.get('api_name', 'unknown')
        tool_system = tool.get('tool_name', '')
        
        full_name = f"{tool_system}.{tool_name}" if tool_system else tool_name
        category_tools[category].append(full_name)
    
    # Convert to counters
    category_counts = {}
    for category, tool_list in category_tools.items():
        category_counts[category] = Counter(tool_list)
    
    return category_counts


def analyze_test_data(test_data_dir: str) -> dict:
    """Analyze tool invocations from test data."""
    test_data_path = Path(test_data_dir)
    
    if not test_data_path.exists():
        print(f"Warning: Test data directory not found: {test_data_dir}")
        return {}
    
    category_invocations = defaultdict(Counter)
    
    # Look for BFCL test files
    for json_file in test_data_path.rglob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle different formats
                if isinstance(data, list):
                    test_cases = data
                elif isinstance(data, dict) and 'data' in data:
                    test_cases = data['data']
                else:
                    test_cases = [data] if isinstance(data, dict) else []
                
                # Extract invocations
                for test_case in test_cases:
                    if isinstance(test_case, dict):
                        invocations = extract_tool_invocations(test_case)
                        # Try to get category
                        category = test_case.get('category', 'Unknown')
                        for tool_name in invocations:
                            category_invocations[category][tool_name] += 1
                            
        except Exception as e:
            print(f"Warning: Could not process {json_file}: {e}")
            continue
    
    return category_invocations


def print_comparison(pool_counts: dict, test_counts: dict, top_n: int = 5):
    """Print comparison of available tools vs actually used tools."""
    print("=" * 80)
    print(f"TOP-{top_n} MOST FREQUENT TOOLS BY CATEGORY")
    print("=" * 80)
    print()
    
    # Get all categories
    all_categories = sorted(set(pool_counts.keys()) | set(test_counts.keys()))
    
    for category in all_categories:
        print(f"📂 {category}")
        print("-" * 80)
        
        # Pool tools
        if category in pool_counts:
            pool_counter = pool_counts[category]
            print(f"   Available in pool: {len(pool_counter)} unique tools")
            top_pool = pool_counter.most_common(top_n)
            for rank, (name, count) in enumerate(top_pool, 1):
                print(f"   {rank}. {name:50s} [Available]")
        else:
            print(f"   ⚠️  No tools in pool for this category")
        
        # Test invocations
        if category in test_counts:
            test_counter = test_counts[category]
            total_uses = sum(test_counter.values())
            print(f"\n   Used in tests: {total_uses} total invocations")
            top_used = test_counter.most_common(top_n)
            for rank, (name, count) in enumerate(top_used, 1):
                percentage = (count / total_uses * 100) if total_uses > 0 else 0
                bar = "█" * int(percentage / 2)
                print(f"   {rank}. {name:50s} {count:4d} ({percentage:5.1f}%) {bar}")
        else:
            print(f"   ℹ️  No test invocations found for this category")
        
        print()


def main():
    """Main function."""
    # Configuration
    tool_pool_path = Path("~/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_all_tool_definitions.jsonl").expanduser()
    test_data_dir = Path("~/data/APIGen-MT/magnet_tool_extraction").expanduser()
    
    print("=" * 80)
    print("BFCL TOOL USAGE FREQUENCY ANALYSIS")
    print("=" * 80)
    print()
    
    # Analyze tool pool
    print(f"📂 Analyzing tool pool: {tool_pool_path}")
    pool_counts = analyze_tool_pool(tool_pool_path)
    print(f"✅ Found {sum(len(c) for c in pool_counts.values())} unique tools")
    print()
    
    # Analyze test data
    print(f"📂 Analyzing test data: {test_data_dir}")
    test_counts = analyze_test_data(test_data_dir)
    if test_counts:
        print(f"✅ Found {sum(sum(c.values()) for c in test_counts.values())} tool invocations")
    print()
    
    # Print comparison
    print_comparison(pool_counts, test_counts, top_n=5)
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    print("Available Tools by Category (Pool):")
    for category in sorted(pool_counts.keys()):
        print(f"  {category:30s}: {len(pool_counts[category]):4d} unique tools")
    
    print()
    
    if test_counts:
        print("Tool Invocations by Category (Tests):")
        for category in sorted(test_counts.keys()):
            total = sum(test_counts[category].values())
            unique = len(test_counts[category])
            print(f"  {category:30s}: {total:4d} total, {unique:4d} unique")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()