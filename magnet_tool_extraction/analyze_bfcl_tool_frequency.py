#!/usr/bin/env python3
"""
Comprehensive BFCL tool usage analysis showing top-5 most frequent tools per category.

Analyzes:
1. Tool definitions available in the pool
2. Actual tool invocations from test data

Usage:
    python analyze_bfcl_tool_frequency.py
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def load_jsonl(file_path: str) -> List[dict]:
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


def load_json(file_path: str) -> List[dict]:
    """Load data from JSON or JSONL file."""
    path = Path(file_path)
    
    # Try JSONL first
    with open(path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        try:
            # Try to parse as single JSON
            f.seek(0)
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            return [data]
        except json.JSONDecodeError:
            # Must be JSONL
            f.seek(0)
            return [json.loads(line.strip()) for line in f if line.strip()]


def extract_tools_from_pool(pool_path: str) -> Dict[str, List[str]]:
    """Extract tools from pool grouped by category."""
    tools_data = load_jsonl(pool_path)
    
    category_tools = defaultdict(list)
    
    for tool in tools_data:
        category = tool.get('category', 'Unknown')
        tool_system = tool.get('tool_name', '')
        api_name = tool.get('api_name', 'unknown')
        
        # Create full tool name
        if tool_system:
            full_name = f"{tool_system}.{api_name}"
        else:
            full_name = api_name
        
        category_tools[category].append(full_name)
    
    return category_tools


def extract_invocations_from_answers(answers: dict) -> List[str]:
    """Extract tool names from ground truth answers."""
    invocations = []
    
    if 'ground_truth' in answers:
        gt = answers['ground_truth']
        if isinstance(gt, list):
            for item in gt:
                if isinstance(item, dict):
                    # Get the tool name (the key of the dict)
                    for tool_name in item.keys():
                        invocations.append(tool_name)
    
    return invocations


def analyze_test_invocations(bfcl_dir: str) -> Dict[str, Counter]:
    """Analyze actual tool invocations from test data."""
    bfcl_path = Path(bfcl_dir)
    possible_answer_dir = bfcl_path / 'possible_answer'
    
    if not possible_answer_dir.exists():
        print(f"Warning: possible_answer directory not found: {possible_answer_dir}")
        return {}
    
    # Map test files to categories
    file_category_map = {
        'simple': 'Simple Functions',
        'multiple': 'Multiple Functions',
        'parallel': 'Parallel Functions',
        'live': 'Live APIs',
        'multi_turn': 'Multi-Turn',
        'java': 'Java',
        'javascript': 'JavaScript',
        'sql': 'SQL',
        'exec': 'Executable'
    }
    
    category_invocations = defaultdict(Counter)
    
    # Process all answer files
    for answer_file in possible_answer_dir.glob('*.json'):
        file_name = answer_file.stem.lower()
        
        # Determine category from filename
        category = 'Unknown'
        for key, cat_name in file_category_map.items():
            if key in file_name:
                category = cat_name
                break
        
        # Load answers
        try:
            answers_list = load_json(str(answer_file))
            
            for answer in answers_list:
                invocations = extract_invocations_from_answers(answer)
                for tool_name in invocations:
                    category_invocations[category][tool_name] += 1
                    
        except Exception as e:
            print(f"Warning: Could not process {answer_file}: {e}")
            continue
    
    return category_invocations


def print_top_tools_per_category(
    pool_tools: Dict[str, List[str]], 
    test_invocations: Dict[str, Counter],
    top_n: int = 5
):
    """Print top-N tools for each category."""
    
    print("=" * 80)
    print(f"TOP-{top_n} MOST FREQUENT TOOLS BY CATEGORY")
    print("=" * 80)
    print()
    
    # Get all categories
    all_categories = sorted(set(pool_tools.keys()) | set(test_invocations.keys()))
    
    for category in all_categories:
        print(f"📂 {category}")
        print("-" * 80)
        
        # Show tools available in pool
        if category in pool_tools:
            pool_counter = Counter(pool_tools[category])
            total_unique = len(pool_counter)
            print(f"   Available in pool: {total_unique} unique tools")
            
            if total_unique <= top_n * 2:
                # Show all if not too many
                for rank, (name, count) in enumerate(pool_counter.most_common(top_n), 1):
                    print(f"   Pool {rank}. {name:50s}")
            else:
                # Show top N
                for rank, (name, count) in enumerate(pool_counter.most_common(top_n), 1):
                    print(f"   Pool {rank}. {name:50s}")
        else:
            print(f"   ⚠️  No tools in pool for this category")
        
        # Show actual invocations from tests
        if category in test_invocations and test_invocations[category]:
            inv_counter = test_invocations[category]
            total_uses = sum(inv_counter.values())
            unique_uses = len(inv_counter)
            
            print(f"\n   Used in tests: {total_uses} total invocations, {unique_uses} unique tools")
            
            top_used = inv_counter.most_common(top_n)
            for rank, (name, count) in enumerate(top_used, 1):
                percentage = (count / total_uses * 100) if total_uses > 0 else 0
                bar_length = int(percentage / 2)  # Scale to max 50 chars
                bar = "█" * bar_length
                print(f"   Test {rank}. {name:50s} {count:5d} ({percentage:5.1f}%) {bar}")
        else:
            print(f"   ℹ️  No test invocations found for this category")
        
        print()


def generate_summary(pool_tools: Dict[str, List[str]], test_invocations: Dict[str, Counter]):
    """Generate summary statistics."""
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()
    
    # Pool summary
    total_pool_tools = sum(len(tools) for tools in pool_tools.values())
    total_pool_categories = len(pool_tools)
    
    print("Tool Pool:")
    print(f"  Total unique tools: {total_pool_tools}")
    print(f"  Total categories: {total_pool_categories}")
    print()
    
    print("  Tools per category:")
    for category in sorted(pool_tools.keys()):
        count = len(pool_tools[category])
        print(f"    {category:30s}: {count:4d} tools")
    print()
    
    # Test invocation summary
    if test_invocations:
        total_invocations = sum(sum(counter.values()) for counter in test_invocations.values())
        total_unique_invoked = sum(len(counter) for counter in test_invocations.values())
        total_test_categories = len([c for c in test_invocations.values() if c])
        
        print("Test Invocations:")
        print(f"  Total invocations: {total_invocations}")
        print(f"  Unique tools used: {total_unique_invoked}")
        print(f"  Categories with usage: {total_test_categories}")
        print()
        
        print("  Invocations per category:")
        for category in sorted(test_invocations.keys()):
            if test_invocations[category]:
                total = sum(test_invocations[category].values())
                unique = len(test_invocations[category])
                print(f"    {category:30s}: {total:5d} total, {unique:4d} unique")
    
    print()
    print("=" * 80)


def export_results(
    pool_tools: Dict[str, List[str]],
    test_invocations: Dict[str, Counter],
    output_path: str
):
    """Export analysis results to JSON."""
    
    results = {
        'pool_analysis': {},
        'test_analysis': {},
        'metadata': {
            'total_pool_tools': sum(len(tools) for tools in pool_tools.values()),
            'total_pool_categories': len(pool_tools),
            'total_test_invocations': sum(sum(c.values()) for c in test_invocations.values()) if test_invocations else 0,
            'total_unique_invoked': sum(len(c) for c in test_invocations.values()) if test_invocations else 0
        }
    }
    
    # Pool data
    for category, tools in pool_tools.items():
        counter = Counter(tools)
        results['pool_analysis'][category] = {
            'total_tools': len(tools),
            'unique_tools': len(counter),
            'top_10': [
                {'name': name, 'available': True}
                for name, _ in counter.most_common(10)
            ]
        }
    
    # Test data
    for category, counter in test_invocations.items():
        if counter:
            results['test_analysis'][category] = {
                'total_invocations': sum(counter.values()),
                'unique_tools': len(counter),
                'top_10': [
                    {
                        'name': name,
                        'count': count,
                        'percentage': (count / sum(counter.values()) * 100)
                    }
                    for name, count in counter.most_common(10)
                ]
            }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results exported to: {output_path}")


def main():
    """Main function."""
    # Configuration
    tool_pool_path = "/home/ishalyminov/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_all_tool_definitions.jsonl"
    bfcl_test_dir = "/home/ishalyminov/data/magnet_mt/data/BFCL_v3"
    output_json = "/home/ishalyminov/data/APIGen-MT/magnet_tool_extraction/bfcl_tool_frequency_analysis.json"
    
    print("=" * 80)
    print("BFCL TOOL FREQUENCY ANALYSIS")
    print("=" * 80)
    print()
    
    # Analyze tool pool
    print(f"📂 Analyzing tool pool: {tool_pool_path}")
    pool_tools = extract_tools_from_pool(tool_pool_path)
    print(f"✅ Found {sum(len(t) for t in pool_tools.values())} tools in {len(pool_tools)} categories")
    print()
    
    # Analyze test invocations
    print(f"📂 Analyzing test invocations: {bfcl_test_dir}")
    test_invocations = analyze_test_invocations(bfcl_test_dir)
    if test_invocations:
        total_inv = sum(sum(c.values()) for c in test_invocations.values())
        print(f"✅ Found {total_inv} invocations in test data")
    print()
    
    # Print top tools per category
    print_top_tools_per_category(pool_tools, test_invocations, top_n=5)
    
    # Generate summary
    generate_summary(pool_tools, test_invocations)
    
    # Export results
    export_results(pool_tools, test_invocations, output_json)
    
    print()
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()