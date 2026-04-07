#!/usr/bin/env python3
"""
Generate 100 unique datapoints using the BFCL tool pool.
This script selects random subsets of tools for each query to avoid token limits.
"""

import json
import os
import random
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


def main():
    print("=" * 60)
    print("Generating 100 unique datapoints using BFCL tool pool")
    print("=" * 60)

    # Configuration
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    tool_pool_path = "/home/ishalyminov/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_all_tool_definitions.jsonl"
    output_dir = "/home/ishalyminov/data/APIGen-MT/data/generated"
    temp_pool_dir = "/home/ishalyminov/data/APIGen-MT/data/generated"
    existing_file = os.path.join(output_dir, "apigen_phase1_31_datapoints.jsonl")
    final_output_file = os.path.join(output_dir, "apigen_phase1_100_datapoints_bfcl.jsonl")

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
    print(f"Loaded {sum(len(t) for t in tools_by_category.values())} tools across {len(tools_by_category)} categories")

    # Generate datapoints until we have 100
    target_count = 100
    attempt = 0
    max_attempts = (target_count - len(datapoints)) * 3  # Allow for some failures
    temp_pool_path = os.path.join(temp_pool_dir, "temp_tool_pool.jsonl")

    # Query templates based on categories
    query_templates = [
        "Find information about {cat} and create a summary event for tracking purposes.",
        "Search for {cat} related data and register a new entry in the system.",
        "Look up {cat} resources and schedule a review meeting.",
        "Get data about {cat} and create a calendar event for follow-up.",
        "Search {cat} database and create a new record with the findings.",
        "Retrieve {cat} information and set up a reminder event.",
        "Find {cat} details and create an event to discuss the results.",
        "Look up {cat} data and schedule a meeting to review the findings.",
        "Get {cat} statistics and create a calendar entry for the presentation.",
        "Search for {cat} records and create an event to analyze the data.",
    ]

    category_list = list(tools_by_category.keys())

    while len(datapoints) < target_count and attempt < max_attempts:
        attempt += 1
        remaining = target_count - len(datapoints)
        print(f"\n--- Attempt {attempt}/{max_attempts} ---")
        print(f"Generated: {len(datapoints)}/{target_count} datapoints ({remaining} remaining)")

        # Select random tool subset
        selected_tools = select_tool_subset(tools_by_category, max_tools=80)
        create_temp_tool_pool(selected_tools, temp_pool_path)

        # Initialize generator with temp tool pool
        tool_manager = ToolManager(llm=llm_client, tool_pool_path=temp_pool_path)
        phase1_generator = APIGenMTPhase1Generator(llm_client=llm_client, tool_manager=tool_manager)

        # Generate query using template with variation
        category = random.choice(category_list)
        base_template = random.choice(query_templates)
        query = base_template.format(cat=category)

        # Add unique variation
        query = f"{query} (variation #{attempt + len(datapoints)}, category: {category})"
        print(f"Query: {query[:80]}...")

        # Generate blueprint
        try:
            verified_bp = phase1_generator.generate_verified_blueprint(query, max_attempts=2)

            if verified_bp:
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
                    ))
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
