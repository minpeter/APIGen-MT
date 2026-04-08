#!/usr/bin/env python3
"""
Generate datapoints using the BFCL tool pool.
This script selects random subsets of tools with uniform category coverage.

Usage:
    python generate_datapoints_bfcl.py [OPTIONS]

Options:
--num-datapoints N       Number of datapoints to generate (default: 100)
--num-actions N         Number of actions/steps to generate per datapoint (default: 2)
--debug                 Enable debug mode to print LLM blueprint generation calls
    --output FILE         Output file path for generated datapoints (default: apigen_phase1_100_datapoints_bfcl.jsonl)
    --help               Show this help message
"""

import json
import os
import random
import argparse
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_client import LocalOpenAILLMClient
from tool_manager import ToolManager

# Import from apigen-phase1 using importlib
import importlib.util
spec = importlib.util.spec_from_file_location(
    "apigen_phase1",
    os.path.join(os.path.dirname(__file__), "apigen-phase1.py")
)
apigen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apigen_module)
APIGenMTPhase1Generator = apigen_module.APIGenMTPhase1Generator

# Global debug flag
DEBUG_MODE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate datapoints using the BFCL tool pool with uniform category coverage.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 100 datapoints (default)
    python generate_datapoints_bfcl.py

    # Generate 50 datapoints
    python generate_datapoints_bfcl.py --num-datapoints 50

# Generate 10 datapoints with 3 actions each
python generate_datapoints_bfcl.py --num-datapoints 10 --num-actions 3

# Generate 10 datapoints with debug output
python generate_datapoints_bfcl.py --num-datapoints 10 --debug

# Generate datapoints with custom output file
python generate_datapoints_bfcl.py --output my_datapoints.jsonl
        """
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
        help='Number of actions/steps to generate per datapoint (default: 2)'
    )

    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode to print LLM blueprint generation calls and tool execution details'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='apigen_phase1_100_datapoints_bfcl.jsonl',
        help='Output file path for generated datapoints (default: apigen_phase1_100_datapoints_bfcl.jsonl)'
    )

    return parser.parse_args()


def load_tool_categories(tool_pool_path: str) -> dict:
    """Load tools and group them by category and source.
    
    For tools without explicit category, infer from source file.
    """
    tools_by_category = {}
    
    with open(tool_pool_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                tool = json.loads(line.strip())
                category = tool.get('category', 'Unknown')
                
                # For tools with 'Unknown' category, use source-based category
                if category == 'Unknown':
                    source = tool.get('source', 'Unknown')
                    # Extract category from source name
                    if 'live' in source.lower():
                        category = 'Live APIs'
                    elif 'exec' in source.lower():
                        category = 'Executable'
                    elif 'simple' in source.lower():
                        category = 'Simple Functions'
                    elif 'multiple' in source.lower():
                        category = 'Multiple Functions'
                    elif 'parallel' in source.lower():
                        category = 'Parallel Functions'
                    else:
                        category = 'Other'
                
                if category not in tools_by_category:
                    tools_by_category[category] = []
                tools_by_category[category].append(tool)
            except json.JSONDecodeError:
                continue
    
    return tools_by_category


def select_tool_subset(tools_by_category: dict, max_tools: int = 80) -> list:
    """Select a random subset of tools with uniform category coverage.
    
    This ensures each category is represented equally to promote diversity
    in the generated datapoints.
    """
    selected_tools = []
    categories = list(tools_by_category.keys())
    
    if not categories:
        return selected_tools
    
    # Calculate how many tools to select per category (uniform distribution)
    tools_per_category = max(1, max_tools // len(categories))
    
    # Select tools uniformly from each category
    for category in categories:
        category_tools = tools_by_category[category]
        # Sample min of: tools_per_category, or available tools in category
        num_to_sample = min(tools_per_category, len(category_tools))
        sampled = random.sample(category_tools, num_to_sample)
        selected_tools.extend(sampled)
    
    # If we still have slots and more tools available, fill randomly
    remaining_slots = max_tools - len(selected_tools)
    if remaining_slots > 0:
        all_tools = []
        for cat_tools in tools_by_category.values():
            all_tools.extend(cat_tools)
        
        available = [t for t in all_tools if t not in selected_tools]
        if available:
            additional = random.sample(available, min(remaining_slots, len(available)))
            selected_tools.extend(additional)
    
    # Shuffle to randomize the order
    random.shuffle(selected_tools)
    
    return selected_tools[:max_tools]


def create_temp_tool_pool(tools: list, temp_path: str):
    """Create a temporary tool pool file with selected tools."""
    with open(temp_path, 'w', encoding='utf-8') as f:
        for tool in tools:
            f.write(json.dumps(tool) + '\n')


def load_existing_datapoints(filepath: str) -> list:
    """Load existing datapoints from a JSONL file."""
    datapoints = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    datapoints.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return datapoints


def get_category_from_tool(tool: dict) -> str:
    """Extract category from a tool definition."""
    cat = tool.get('category', 'Unknown')
    if cat == 'Unknown':
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


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set global debug flag
    global DEBUG_MODE
    DEBUG_MODE = args.debug
    
    target_count = args.num_datapoints
    
    print("=" * 60)
    print(f"Generating {target_count} unique datapoints using BFCL tool pool")
    if DEBUG_MODE:
        print("🔧 DEBUG MODE ENABLED")
    print("=" * 60)

    # Configuration
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    tool_pool_path = "/home/ishalyminov/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_all_tool_definitions.jsonl"
    output_dir = "/home/ishalyminov/data/APIGen-MT/data/generated"
    temp_pool_dir = "/home/ishalyminov/data/APIGen-MT/data/generated"
    
    # Use output file from command line argument
    output_filename = args.output
    if not os.path.isabs(output_filename):
        # If it's not an absolute path, make it relative to output_dir
        final_output_file = os.path.join(output_dir, output_filename)
    else:
        final_output_file = output_filename
    
    existing_file = final_output_file  # Use the same file for existing datapoints

    os.makedirs(output_dir, exist_ok=True)

    if not api_key or not api_base:
        print("ERROR: OPENAI_API_KEY or OPENAI_API_BASE not set in .env file.")
        sys.exit(1)

    # Load existing datapoints
    datapoints = load_existing_datapoints(existing_file)
    print(f"Loaded {len(datapoints)} existing datapoints")

    # Initialize LLM client
    llm_client = LocalOpenAILLMClient(
        url=api_base,
        api_key=api_key,
        api_model="nvidia/nemotron-3-super-120b-a12b",
        hf_tokenizer_id=None,
    )

    # Load tool categories
    print("Loading tool categories...")
    tools_by_category = load_tool_categories(tool_pool_path)
    total_tools = sum(len(t) for t in tools_by_category.values())
    print(f"Loaded {total_tools} tools across {len(tools_by_category)} categories")
    
    # Display category distribution
    print("\nCategory distribution:")
    for cat, tools in sorted(tools_by_category.items()):
        print(f"  {cat:30s}: {len(tools):4d} tools")

    attempt = 0
    max_attempts = (target_count - len(datapoints)) * 3  # Allow for some failures
    temp_pool_path = os.path.join(temp_pool_dir, "temp_tool_pool.jsonl")

    category_list = list(tools_by_category.keys())

    while len(datapoints) < target_count and attempt < max_attempts:
        attempt += 1
        remaining = target_count - len(datapoints)
        print(f"\n--- Attempt {attempt}/{max_attempts} ---")
        print(f"Generated: {len(datapoints)}/{target_count} datapoints ({remaining} remaining)")

        # Select random tool subset with uniform category coverage
        selected_tools = select_tool_subset(tools_by_category, max_tools=80)
        create_temp_tool_pool(selected_tools, temp_pool_path)
        
        # Track which categories are in this selection
        selected_categories = set()
        for tool in selected_tools:
            cat = get_category_from_tool(tool)
            selected_categories.add(cat)

        # Initialize generator with temp tool pool
        tool_manager = ToolManager(llm=llm_client, tool_pool_path=temp_pool_path)
        phase1_generator = APIGenMTPhase1Generator(llm_client=llm_client, tool_manager=tool_manager, num_actions=args.num_actions)

        # Generate query based on available tools and their categories
        # Pick a random category to focus the query on
        focus_category = random.choice(list(selected_categories))
        
        # Generate a diverse query
        query = f"Using tools from {focus_category}, perform a multi-step operation that requires retrieving information and then creating or updating a record. (variation #{attempt + len(datapoints)})"
        print(f"Query: {query[:80]}...")

        # Generate blueprint
        try:
            if DEBUG_MODE:
                print("\n" + "=" * 70)
                print("🔧 DEBUG: Generating blueprint...")
                print(f"   Query: {query}")
                print(f"   Focus category: {focus_category}")
                print(f"   Tools available: {len(selected_tools)}")
                print(f"   Categories in pool: {', '.join(sorted(selected_categories))}")
                print("=" * 70)
            
            verified_bp = phase1_generator.generate_verified_blueprint(query, max_attempts=2)

            if verified_bp:
                if DEBUG_MODE:
                    print("\n" + "=" * 70)
                    print("🔧 DEBUG: Blueprint generated successfully")
                    print(f"   Blueprint query: {verified_bp.blueprint.q}")
                    print(f"   Number of steps: {len(verified_bp.blueprint.a_gt_steps)}")
                    if verified_bp.llm_review_history:
                        print(f"   Quality: {verified_bp.llm_review_history[-1].quality_assessment}")
                    print("   Tool calls:")
                    for idx, step in enumerate(verified_bp.blueprint.a_gt_steps):
                        print(f"     Step {idx}:")
                        for tc in step.tool_calls:
                            print(f"       - {tc.tool_name}({json.dumps(tc.arguments)})")
                    print("=" * 70 + "\n")
                
                print("✓ Successfully generated blueprint")

                data_point = {
                    "query": query,
                    "blueprint": verified_bp.blueprint.model_dump(),
                    "validation_result": verified_bp.validation_result.model_dump(),
                    "llm_review_history": [review.model_dump() for review in verified_bp.llm_review_history],
                    "generation_attempts": verified_bp.generation_attempts,
                    "timestamp": datetime.now().isoformat(),
                    "tools_used": list(set(
                        tc.tool_name
                        for step in verified_bp.blueprint.a_gt_steps
                        for tc in step.tool_calls
                    )),
                    "categories_in_pool": list(selected_categories),
                    "focus_category": focus_category
                }

                datapoints.append(data_point)

                # Save every datapoint by appending to file
                with open(final_output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data_point, ensure_ascii=False) + "\n")
                print(f"Progress saved: {len(datapoints)} datapoints")
            else:
                print("✗ Failed to generate blueprint")

        except Exception as e:
            print(f"✗ Error: {e}")

    print(f"\n{'='*60}")
    print("Generation complete!")
    print(f"Total datapoints: {len(datapoints)}")
    print(f"Output file: {final_output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()