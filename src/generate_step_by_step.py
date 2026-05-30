#!/usr/bin/env python3
"""
Generate datapoints using step-by-step blueprint generation.

This script generates datapoints where each step is generated sequentially
with immediate tool execution simulation, resulting in a complete
conversation trajectory.

Usage:
    python generate_step_by_step.py [OPTIONS]

Options:
    --num-datapoints N    Number of datapoints to generate (default: 100)
    --num-actions N       Number of actions/steps per datapoint (default: 2)
    --output FILE         Output file path (default: step_by_step_datapoints.jsonl)
    --debug               Enable debug output
"""

import json
import os
import sys
import random
import argparse
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Force unbuffered output for nohup
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import LocalOpenAILLMClient
from tool_manager import ToolManager
from apigen_step_by_step import StepByStepGenerator, StepByStepDatapoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate datapoints using step-by-step blueprint generation.'
    )
    
    parser.add_argument(
        '--num-datapoints', '-n',
        type=int,
        default=100,
        help='Number of datapoints to generate (default: 100)'
    )
    
    parser.add_argument(
        '--num-actions', '-a',
        type=int,
        default=2,
        help='Number of actions/steps per datapoint (default: 2)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='step_by_step_datapoints.jsonl',
        help='Output file path (default: step_by_step_datapoints.jsonl)'
    )
    
    parser.add_argument(
        '--tool-pool',
        type=str,
        default='/home/ishalyminov/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl',
        help='Path to tool pool file'
    )

    parser.add_argument(
        '--invocation-examples',
        type=str,
        default='/home/ishalyminov/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl',
        help='Path to invocation examples file (for Python tool implementations)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='deepseek-v4-flash-free',
        help='Model to use for generation (default: deepseek-v4-flash-free)'
    )

    return parser.parse_args()


def load_tool_categories(tool_pool_path: str) -> dict:
    """Load tools and group them by category."""
    tools_by_category = {}
    
    with open(tool_pool_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                tool = json.loads(line.strip())
                category = tool.get('category', 'Unknown')
                
                if category not in tools_by_category:
                    tools_by_category[category] = []
                tools_by_category[category].append(tool)
            except json.JSONDecodeError:
                continue
    
    return tools_by_category


def main():
    args = parse_args()
    
    print("=" * 70)
    print("STEP-BY-STEP DATAPOINT GENERATION")
    print("=" * 70)
    print(f"Target: {args.num_datapoints} datapoints")
    print(f"Actions per datapoint: {args.num_actions}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print("=" * 70)
    
    # Configuration
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    
    if not api_key or not api_base:
        print("ERROR: OPENAI_API_KEY or OPENAI_API_BASE not set")
        sys.exit(1)
    
    # Initialize LLM client
    llm_client = LocalOpenAILLMClient(
        url=api_base,
        api_key=api_key,
        api_model=args.model,
        hf_tokenizer_id=None
    )
    
    # Load tool categories for uniform sampling
    print("\nLoading tools...")
    tools_by_category = load_tool_categories(args.tool_pool)
    total_tools = sum(len(t) for t in tools_by_category.values())
    print(f"Loaded {total_tools} tools across {len(tools_by_category)} categories")
    
    for cat, tools in sorted(tools_by_category.items()):
        print(f"  {cat:30s}: {len(tools):3d} tools")
    
    # Initialize tool manager (with Python tool implementations)
    tool_manager = ToolManager(
        llm=llm_client,
        tool_pool_path=args.tool_pool,
        invocation_examples_path=args.invocation_examples
    )
    
    # Initialize generator
    generator = StepByStepGenerator(
        llm_client=llm_client,
        tool_manager=tool_manager,
        num_actions=args.num_actions
    )
    
    # Output setup
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track generated datapoints
    datapoints = []
    categories = list(tools_by_category.keys())
    
    # Generation loop - run until target number of datapoints is reached
    attempt = 0

    while len(datapoints) < args.num_datapoints:
        remaining = args.num_datapoints - len(datapoints)

        print(f"\n{'='*70}")
        print(f"Generated: {len(datapoints)}/{args.num_datapoints} | Remaining: {remaining}")
        print("=" * 70)
        
        # Select random category to focus on
        focus_category = random.choice(categories)
        print(f"Focus category: {focus_category}")
        
        # Generate datapoint
        datapoint = generator.generate_datapoint(
            focus_category=focus_category
        )

        if datapoint:
            # Add timestamp
            datapoint_dict = datapoint.model_dump()
            datapoint_dict['timestamp'] = datetime.now().isoformat()
            datapoint_dict['generation_attempt'] = attempt

            # Save immediately - only verified datapoints
            with open(args.output, 'a', encoding='utf-8') as f:
                f.write(json.dumps(datapoint_dict, ensure_ascii=False) + '\n')

            datapoints.append(datapoint)
            print(f"\n✓ Successfully generated and verified datapoint {len(datapoints)}")
            print(f" Query: {datapoint.trajectory.query}")
            print(f" Tools used: {datapoint.trajectory.tools_used}")
        else:
            # Generation failed or verification failed - don't count towards target
            print(f"\n✗ Failed to generate datapoint (verification failed or generation error)")
    
    # Summary
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total generated: {len(datapoints)}/{args.num_datapoints}")
    print(f"Output file: {args.output}")

    if datapoints:
        # Tool statistics
        tools_used_all = []
        for dp in datapoints:
            tools_used_all.extend(dp.trajectory.tools_used)

        from collections import Counter
        tool_counts = Counter(tools_used_all)

        print(f"\nTop 10 tools used:")
        for tool, count in tool_counts.most_common(10):
            print(f"  {tool}: {count}")

        # Token usage statistics
        total_llm_calls = sum(dp.token_usage.total_llm_calls for dp in datapoints)
        total_prompt_tokens = sum(dp.token_usage.prompt_tokens for dp in datapoints)
        total_completion_tokens = sum(dp.token_usage.completion_tokens for dp in datapoints)
        total_tokens = sum(dp.token_usage.total_tokens for dp in datapoints)

        print(f"\nToken Usage Statistics:")
        print(f"  Total LLM calls: {total_llm_calls}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"    - Prompt tokens: {total_prompt_tokens:,}")
        print(f"    - Completion tokens: {total_completion_tokens:,}")
        print(f"  Average per datapoint:")
        print(f"    - LLM calls: {total_llm_calls / len(datapoints):.1f}")
        print(f"    - Tokens: {total_tokens / len(datapoints):.0f}")

    print("=" * 70)


if __name__ == "__main__":
    main()