#!/usr/bin/env python3
"""
Generate datapoints using step-by-step blueprint generation.

Supports two modes:
  - multi-turn (default): Each user turn is a separate exchange with its own query.
    Use --num-turns to control total turns and --num-actions for tools per turn.
  - step-by-step (legacy): Single user query with multiple action steps.

Usage:
    python generate_step_by_step.py [OPTIONS]

Options:
    --mode MODE             Generation mode: multi-turn (default) or step-by-step
    --num-datapoints N      Number of datapoints to generate (default: 100)
    --num-turns N            Number of user-assistant turns for multi-turn (default: 10)
    --num-actions N         Actions per turn (default: 1)
    --output FILE           Output file path (default: step_by_step_datapoints.jsonl)
    --category CATEGORY     Filter tools to a specific category
    --model MODEL           Model name (default: minimaxai/minimax-m2.7)
"""

import json
import os
import sys
import random
import argparse
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import LocalOpenAILLMClient
from tool_manager import ToolManager
from apigen_step_by_step import StepByStepGenerator, StepByStepDatapoint
from apigen_multi_turn import MultiTurnGenerator, MultiTurnDatapoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate datapoints using step-by-step blueprint generation.'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='multi-turn',
        choices=['multi-turn', 'step-by-step'],
        help='Generation mode: multi-turn (default) or step-by-step (legacy)'
    )

    parser.add_argument(
        '--num-datapoints', '-n',
        type=int,
        default=100,
        help='Number of datapoints to generate (default: 100)'
    )

    parser.add_argument(
        '--num-turns', '-t',
        type=int,
        default=10,
        help='Number of user-assistant turns for multi-turn mode (default: 10)'
    )

    parser.add_argument(
        '--num-actions', '-a',
        type=int,
        default=1,
        help='Number of actions per turn (default: 1)'
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
        default='~/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl',
        help='Path to tool pool file'
    )

    parser.add_argument(
        '--invocation-examples',
        type=str,
        default='~/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl',
        help='Path to invocation examples file (for Python tool implementations)'
    )

    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='Filter tools to a specific category'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='minimaxai/minimax-m2.7',
        help='Model to use for generation (default: minimaxai/minimax-m2.7)'
    )

    parser.add_argument(
        '--judge-model',
        type=str,
        default=None,
        help='Model to use for judge tasks (state verification, sequence validation). Defaults to --model if not set.'
    )

    parser.add_argument(
        '--judge-api-base',
        type=str,
        default=None,
        help='API base URL for judge model. Defaults to OPENAI_API_BASE if not set.'
    )

    parser.add_argument(
        '--judge-api-key',
        type=str,
        default=None,
        help='API key for judge model. Defaults to OPENAI_API_KEY if not set.'
    )

    parser.add_argument(
        '--num-actions-range',
        type=int,
        nargs=2,
        default=None,
        metavar=('MIN', 'MAX'),
        help='Randomize num_actions per datapoint between MIN and MAX (inclusive). Overrides -a.'
    )

    parser.add_argument(
        '--config-pool',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use diverse config pool for initial API states (default: True). Use --no-config-pool to disable.'
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


def run_step_by_step(args, llm_client, tool_manager, categories, output_path, judge_client=None):
    """Run step-by-step (legacy single-query) generation."""
    generator = StepByStepGenerator(
        llm_client=llm_client,
        tool_manager=tool_manager,
        num_actions=args.num_actions,
        judge_client=judge_client
    )

    datapoints = []
    attempt = 0

    while len(datapoints) < args.num_datapoints:
        remaining = args.num_datapoints - len(datapoints)

        print(f"\n{'='*70}")
        print(f"Generated: {len(datapoints)}/{args.num_datapoints} | Remaining: {remaining}")
        print("=" * 70)

        focus_category = random.choice(categories)
        print(f"Focus category: {focus_category}")

        # Randomize num_actions if range is specified
        if args.num_actions_range:
            num_actions = random.randint(args.num_actions_range[0], args.num_actions_range[1])
            generator.num_actions = num_actions
            print(f"Actions for this datapoint: {num_actions}")
        
        # Generate datapoint
        datapoint = generator.generate_datapoint(
            focus_category=focus_category
        )

        if datapoint:
            datapoint_dict = datapoint.model_dump()
            datapoint_dict['timestamp'] = datetime.now().isoformat()
            datapoint_dict['generation_attempt'] = attempt

            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(datapoint_dict, ensure_ascii=False) + '\n')

            datapoints.append(datapoint)
            print(f"\n✓ Successfully generated and verified datapoint {len(datapoints)}")
            print(f" Query: {datapoint.trajectory.query}")
            print(f" Tools used: {datapoint.trajectory.tools_used}")
        else:
            print(f"\n✗ Failed to generate datapoint")

        attempt += 1

    return datapoints


def run_multi_turn(args, llm_client, tool_manager, categories, output_path):
    """Run multi-turn (multiple user exchanges) generation."""
    generator = MultiTurnGenerator(
        llm_client=llm_client,
        tool_manager=tool_manager,
        num_turns=args.num_turns,
        actions_per_turn=args.num_actions,
    )

    datapoints = []

    while len(datapoints) < args.num_datapoints:
        remaining = args.num_datapoints - len(datapoints)

        print(f"\n{'='*70}")
        print(f"Generated: {len(datapoints)}/{args.num_datapoints} | Remaining: {remaining}")
        print("=" * 70)

        focus_category = random.choice(categories)
        print(f"Focus category: {focus_category}")

        dp = generator.generate_multi_turn_datapoint(focus_category=focus_category)

        if dp is None:
            print(f"\n✗ Failed to generate datapoint, retrying...")
            continue

        dp_dict = dp.model_dump()
        dp_dict['timestamp'] = datetime.now().isoformat()

        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(dp_dict, ensure_ascii=False) + '\n')

        datapoints.append(dp)
        print(f"\n✓ Successfully generated datapoint {len(datapoints)}")
        print(f" Task: {dp.conversation.overall_task[:80]}")
        print(f" Turns: {len(dp.conversation.turns)}")
        print(f" Tools: {dp.conversation.tools_used}")

    return datapoints


def main():
    args = parse_args()

    tool_pool_path = str(Path(args.tool_pool).expanduser())
    invocation_examples_path = str(Path(args.invocation_examples).expanduser())

    mode_label = "MULTI-TURN" if args.mode == "multi-turn" else "STEP-BY-STEP"
    print("=" * 70)
    print(f"{mode_label} DATAPOINT GENERATION")
    print("=" * 70)
    print(f"Target: {args.num_datapoints} datapoints")
    if args.mode == "multi-turn":
        print(f"Turns per conversation: {args.num_turns}")
        print(f"Actions per turn: {args.num_actions}")
    else:
        print(f"Actions per datapoint: {args.num_actions}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print("=" * 70)

    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")

    if not api_key or not api_base:
        print("ERROR: OPENAI_API_KEY or OPENAI_API_BASE not set")
        sys.exit(1)

    llm_client = LocalOpenAILLMClient(
        url=api_base,
        api_key=api_key,
        api_model=args.model,
        hf_tokenizer_id=None
    )

    print("\nLoading tools...")
    tools_by_category = load_tool_categories(tool_pool_path)

    if args.category:
        filtered = {args.category: tools_by_category.get(args.category)}
        if filtered[args.category] is None:
            print(f"Error: Category '{args.category}' not found")
            available = list(tools_by_category.keys())
            print(f"Available categories: {available}")
            return
        tools_by_category = filtered

    total_tools = sum(len(t) for t in tools_by_category.values())
    print(f"Loaded {total_tools} tools across {len(tools_by_category)} categories")

    for cat, tools in sorted(tools_by_category.items()):
        print(f"  {cat:30s}: {len(tools):3d} tools")

    tool_manager = ToolManager(
        llm=llm_client,
        tool_pool_path=tool_pool_path,
        invocation_examples_path=invocation_examples_path,
        use_config_pool=args.config_pool,
    )

    # Initialize judge client if specified, otherwise reuse generator client
    if args.judge_model:
        judge_api_base = args.judge_api_base or api_base
        judge_api_key = args.judge_api_key or api_key
        judge_client = LocalOpenAILLMClient(
            url=judge_api_base,
            api_key=judge_api_key,
            api_model=args.judge_model,
            hf_tokenizer_id=None
        )
    else:
        judge_client = llm_client

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = list(tools_by_category.keys())

    if args.mode == "multi-turn":
        datapoints = run_multi_turn(args, llm_client, tool_manager, categories, args.output)
    else:
        datapoints = run_step_by_step(args, llm_client, tool_manager, categories, args.output, judge_client=judge_client)

    # Summary
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total generated: {len(datapoints)}/{args.num_datapoints}")
    print(f"Output file: {args.output}")

    if datapoints:
        from collections import Counter

        if args.mode == "multi-turn":
            tools_used_all = []
            for dp in datapoints:
                tools_used_all.extend(dp.conversation.tools_used)
            tool_counts = Counter(tools_used_all)

            print(f"\nTop 10 tools used:")
            for tool, count in tool_counts.most_common(10):
                print(f"  {tool}: {count}")

            total_calls = sum(dp.token_usage.total_llm_calls for dp in datapoints)
            total_tokens = sum(dp.token_usage.total_tokens for dp in datapoints)
        else:
            tools_used_all = []
            for dp in datapoints:
                tools_used_all.extend(dp.trajectory.tools_used)
            tool_counts = Counter(tools_used_all)

            print(f"\nTop 10 tools used:")
            for tool, count in tool_counts.most_common(10):
                print(f"  {tool}: {count}")

            total_calls = sum(dp.token_usage.total_llm_calls for dp in datapoints)
            total_tokens = sum(dp.token_usage.total_tokens for dp in datapoints)

        print(f"\nToken Usage Statistics:")
        print(f"  Total LLM calls: {total_calls}")
        print(f"  Total tokens: {total_tokens:,}")
        if datapoints:
            print(f"  Average per datapoint:")
            print(f"    - LLM calls: {total_calls / len(datapoints):.1f}")
            print(f"    - Tokens: {total_tokens // len(datapoints):,}")

    print("=" * 70)


if __name__ == "__main__":
    main()