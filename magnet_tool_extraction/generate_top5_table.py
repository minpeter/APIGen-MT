#!/usr/bin/env python3
"""
Generate a visual summary table of top-5 most frequent tools per BFCL category.

Creates a formatted table showing both pool availability and actual test usage.

Usage:
    python generate_top5_table.py
"""

import json
from pathlib import Path


def load_analysis_results(json_path: str) -> dict:
    """Load the analysis results from JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_markdown_table(results: dict) -> str:
    """Generate a markdown table of top-5 tools per category."""
    
    md = "# BFCL Tool Pool Analysis: Top-5 Most Frequent Tools per Category\n\n"
    md += "## Summary\n\n"
    md += f"- **Total unique tools in pool**: {results['metadata']['total_pool_tools']}\n"
    md += f"- **Total test invocations**: {results['metadata']['total_test_invocations']}\n"
    md += f"- **Unique tools actually used**: {results['metadata']['total_unique_invoked']}\n\n"
    
    md += "## Top-5 Tools by Category\n\n"
    
    # Categories from pool
    categories = sorted(results['pool_analysis'].keys())
    
    for category in categories:
        pool_data = results['pool_analysis'].get(category, {})
        test_data = results['test_analysis'].get(category, {})
        
        md += f"### {category}\n\n"
        
        if pool_data:
            md += f"**Available in Pool** ({pool_data['unique_tools']} unique tools):\n\n"
            for i, tool in enumerate(pool_data['top_10'][:5], 1):
                md += f"{i}. `{tool['name']}`\n"
            md += "\n"
        
        if test_data:
            total_inv = test_data['total_invocations']
            unique = test_data['unique_tools']
            md += f"**Used in Tests** ({total_inv} total invocations, {unique} unique):\n\n"
            md += "| Rank | Tool Name | Count | Percentage |\n"
            md += "|------|-----------|-------|------------|\n"
            for i, tool in enumerate(test_data['top_10'][:5], 1):
                md += f"| {i} | `{tool['name']}` | {tool['count']} | {tool['percentage']:.1f}% |\n"
            md += "\n"
        
        if not pool_data and not test_data:
            md += "*No data available for this category.*\n\n"
        
        md += "---\n\n"
    
    return md


def generate_console_table(results: dict):
    """Print a formatted console table."""
    
    print("\n" + "=" * 80)
    print("BFCL TOOL POOL: TOP-5 MOST FREQUENT TOOLS PER CATEGORY")
    print("=" * 80)
    print()
    
    print("SUMMARY:")
    print(f"  Total unique tools in pool: {results['metadata']['total_pool_tools']}")
    print(f"  Total test invocations: {results['metadata']['total_test_invocations']}")
    print(f"  Unique tools used in tests: {results['metadata']['total_unique_invoked']}")
    print()
    
    # Process each category
    categories = sorted(set(results['pool_analysis'].keys()) | set(results['test_analysis'].keys()))
    
    for category in categories:
        print("─" * 80)
        print(f"📂 {category}")
        print("─" * 80)
        
        pool_data = results['pool_analysis'].get(category, {})
        test_data = results['test_analysis'].get(category, {})
        
        if pool_data:
            print(f"\n  Available in Pool: {pool_data['unique_tools']} unique tools")
            print("  Top-5:")
            for i, tool in enumerate(pool_data['top_10'][:5], 1):
                print(f"    {i}. {tool['name']}")
            print()
        
        if test_data:
            total_inv = test_data['total_invocations']
            unique = test_data['unique_tools']
            print(f"  Used in Tests: {total_inv} total invocations, {unique} unique tools")
            print(f"  {'Rank':<6}{'Tool Name':<45}{'Count':<8}{'%':<8}")
            print(f"  {'-'*6}{'-'*45}{'-'*8}{'-'*8}")
            for i, tool in enumerate(test_data['top_10'][:5], 1):
                print(f"  {i:<6}{tool['name']:<45}{tool['count']:<8}{tool['percentage']:.1f}%")
            print()
        
        if not pool_data and not test_data:
            print("  ⚠️  No data available\n")
    
    print("=" * 80)


def main():
    """Main function."""
    # Configuration
    analysis_json = "/home/ishalyminov/data/APIGen-MT/magnet_tool_extraction/bfcl_tool_frequency_analysis.json"
    output_md = "/home/ishalyminov/data/APIGen-MT/magnet_tool_extraction/TOP5_TOOLS_BY_CATEGORY.md"
    
    # Load results
    print(f"Loading analysis from: {analysis_json}")
    results = load_analysis_results(analysis_json)
    
    # Generate console table
    generate_console_table(results)
    
    # Generate markdown table
    md_content = generate_markdown_table(results)
    
    # Save markdown
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"\n✅ Markdown table saved to: {output_md}")


if __name__ == "__main__":
    main()