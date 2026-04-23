#!/usr/bin/env python3
"""
Extract aggregated token statistics from APIGen conversation log files.
Aggregates the token_usage field from each datapoint.

Usage:
python extract_token_statistics.py <input_file> [--output <output_file>]

Example:
python extract_token_statistics.py data/generated/step_by_step_10datapoints_3actions.jsonl
python extract_token_statistics.py data/generated/step_by_step_100_datapoints.jsonl --output stats.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def analyze_trajectory(datapoint: dict, index: int) -> dict:
    """Analyze a single trajectory datapoint and return token statistics."""
    trajectory = datapoint.get("trajectory", {})
    token_usage = datapoint.get("token_usage", {})
    metadata = datapoint.get("generation_metadata", {})

    stats = {
        "datapoint_index": index,
        "num_steps": len(trajectory.get("steps", [])),
        "num_tool_calls": sum(
            len(step.get("tool_calls", []))
            for step in trajectory.get("steps", [])
        ),
        "tools_used": trajectory.get("tools_used", []),
        "categories_used": trajectory.get("categories_used", []),
        "focus_category": metadata.get("focus_category", "unknown"),
        "expected_actions": metadata.get("num_actions", 0),
        # Token counts from token_usage field
        "prompt_tokens": token_usage.get("prompt_tokens", 0),
        "completion_tokens": token_usage.get("completion_tokens", 0),
        "total_tokens": token_usage.get("total_tokens", 0),
        "total_llm_calls": token_usage.get("total_llm_calls", 0),
    }

    return stats


def aggregate_statistics(all_stats: list[dict]) -> dict:
    """Aggregate statistics across all datapoints."""
    if not all_stats:
        return {}

    aggregated = {
        "total_datapoints": len(all_stats),
        "total_steps": sum(s["num_steps"] for s in all_stats),
        "total_tool_calls": sum(s["num_tool_calls"] for s in all_stats),

        # Token sums from token_usage
        "total_prompt_tokens": sum(s["prompt_tokens"] for s in all_stats),
        "total_completion_tokens": sum(s["completion_tokens"] for s in all_stats),
        "total_tokens": sum(s["total_tokens"] for s in all_stats),
        "total_llm_calls": sum(s["total_llm_calls"] for s in all_stats),

        # Averages per datapoint
        "avg_steps_per_datapoint": sum(s["num_steps"] for s in all_stats) / len(all_stats),
        "avg_tool_calls_per_datapoint": sum(s["num_tool_calls"] for s in all_stats) / len(all_stats),
        "avg_prompt_tokens": sum(s["prompt_tokens"] for s in all_stats) / len(all_stats),
        "avg_completion_tokens": sum(s["completion_tokens"] for s in all_stats) / len(all_stats),
        "avg_total_tokens": sum(s["total_tokens"] for s in all_stats) / len(all_stats),
        "avg_llm_calls": sum(s["total_llm_calls"] for s in all_stats) / len(all_stats),

        # Min/Max
        "min_total_tokens": min(s["total_tokens"] for s in all_stats),
        "max_total_tokens": max(s["total_tokens"] for s in all_stats),

        # Tool and category frequency
        "tool_usage_counts": defaultdict(int),
        "category_usage_counts": defaultdict(int),
        "focus_category_counts": defaultdict(int),
    }

    # Count tool and category frequencies
    for stats in all_stats:
        for tool in stats.get("tools_used", []):
            aggregated["tool_usage_counts"][tool] += 1
        for category in stats.get("categories_used", []):
            aggregated["category_usage_counts"][category] += 1
        aggregated["focus_category_counts"][stats.get("focus_category", "unknown")] += 1

    # Convert defaultdict to regular dict for JSON serialization
    aggregated["tool_usage_counts"] = dict(aggregated["tool_usage_counts"])
    aggregated["category_usage_counts"] = dict(aggregated["category_usage_counts"])
    aggregated["focus_category_counts"] = dict(aggregated["focus_category_counts"])

    return aggregated


def format_number(num: float) -> str:
    """Format a number with commas and 2 decimal places if needed."""
    if isinstance(num, int):
        return f"{num:,}"
    return f"{num:,.2f}"


def print_statistics(aggregated: dict, detailed_stats: list[dict] = None):
    """Print formatted statistics to stdout."""
    print("=" * 80)
    print("TOKEN STATISTICS REPORT")
    print("=" * 80)
    print()

    print("📊 OVERVIEW")
    print("-" * 40)
    print(f" Total Datapoints: {format_number(aggregated['total_datapoints'])}")
    print(f" Total Steps: {format_number(aggregated['total_steps'])}")
    print(f" Total Tool Calls: {format_number(aggregated['total_tool_calls'])}")
    print()

    print("📝 TOKEN COUNTS (Aggregated)")
    print("-" * 40)
    print(f" Prompt Tokens: {format_number(aggregated['total_prompt_tokens']):>15}")
    print(f" Completion Tokens: {format_number(aggregated['total_completion_tokens']):>15}")
    print("-" * 40)
    print(f" Total Tokens: {format_number(aggregated['total_tokens']):>15}")
    print()
    print(f" Total LLM Calls: {format_number(aggregated['total_llm_calls']):>15}")
    print()

    print("📈 AVERAGES (Per Datapoint)")
    print("-" * 40)
    print(f" Steps per Datapoint: {aggregated['avg_steps_per_datapoint']:>10.2f}")
    print(f" Tool Calls per Datapoint: {aggregated['avg_tool_calls_per_datapoint']:>10.2f}")
    print(f" Prompt Tokens: {aggregated['avg_prompt_tokens']:>10.2f}")
    print(f" Completion Tokens: {aggregated['avg_completion_tokens']:>10.2f}")
    print(f" Total Tokens: {aggregated['avg_total_tokens']:>10.2f}")
    print(f" LLM Calls: {aggregated['avg_llm_calls']:>10.2f}")
    print()

    print("📉 MIN/MAX")
    print("-" * 40)
    print(f" Min Total Tokens: {format_number(aggregated['min_total_tokens']):>15}")
    print(f" Max Total Tokens: {format_number(aggregated['max_total_tokens']):>15}")
    print()

    if aggregated.get("tool_usage_counts"):
        print("🔧 TOOL USAGE COUNTS")
        print("-" * 40)
        sorted_tools = sorted(
            aggregated["tool_usage_counts"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for tool, count in sorted_tools:
            print(f" {tool:.<30} {count:>5}")
        print()

    if aggregated.get("category_usage_counts"):
        print("📂 CATEGORY USAGE COUNTS")
        print("-" * 40)
        sorted_categories = sorted(
            aggregated["category_usage_counts"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for category, count in sorted_categories:
            print(f" {category:.<30} {count:>5}")
        print()

    if aggregated.get("focus_category_counts"):
        print("🎯 FOCUS CATEGORY DISTRIBUTION")
        print("-" * 40)
        sorted_focus = sorted(
            aggregated["focus_category_counts"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for category, count in sorted_focus:
            print(f" {category:.<30} {count:>5}")
        print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Extract aggregated token statistics from APIGen conversation log files"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the JSONL file containing conversation trajectories"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file for detailed statistics (optional)"
    )
    parser.add_argument(
        "--per-datapoint",
        action="store_true",
        help="Include per-datapoint statistics in output JSON"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Read and analyze all datapoints
    all_stats = []
    print(f"Reading {input_path}...", file=sys.stderr)

    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                datapoint = json.loads(line)
                stats = analyze_trajectory(datapoint, i)
                all_stats.append(stats)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {i}: {e}", file=sys.stderr)
                continue

    print(f"Processed {len(all_stats)} datapoints", file=sys.stderr)

    if not all_stats:
        print("Error: No valid datapoints found in file", file=sys.stderr)
        sys.exit(1)

    # Aggregate statistics
    aggregated = aggregate_statistics(all_stats)

    # Print summary to stdout
    print_statistics(aggregated, all_stats if args.per_datapoint else None)

    # Save to JSON if requested
    if args.output:
        output_data = {
            "aggregated": aggregated,
        }
        if args.per_datapoint:
            output_data["per_datapoint"] = all_stats

        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed statistics saved to: {output_path}")


if __name__ == "__main__":
    main()
