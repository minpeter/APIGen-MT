#!/usr/bin/env python3
"""
Debug script to print all rendered prompts for the tool manager and invocation simulator.
This script runs generate_datapoints_bfcl.py for 1 datapoint and prints all LLM prompts.
"""

import json
import os
import sys
import random

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from llm_client import LocalOpenAILLMClient
from tool_manager import ToolManager

# Import apigen-phase1 using importlib
import importlib.util
spec = importlib.util.spec_from_file_location(
    "apigen_phase1",
    os.path.join(os.path.dirname(__file__), "apigen-phase1.py")
)
apigen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apigen_module)
APIGenMTPhase1Generator = apigen_module.APIGenMTPhase1Generator

# Track all prompts
captured_prompts = []

# ============================================================================
# Monkey-patch LLM client to capture prompts
# ============================================================================
original_json_output = LocalOpenAILLMClient.json_output

def patched_json_output(self, prompt, system_prompt=None, reasoning=False, schema=None, **kwargs):
    """Patched json_output that captures prompts."""
    global captured_prompts
    
    prompt_info = {
        "type": "LLM Call",
        "system_prompt": system_prompt,
        "user_prompt": prompt,
        "reasoning": reasoning,
        "schema": schema.__name__ if schema else None
    }
    captured_prompts.append(prompt_info)
    
    # Call original
    return original_json_output(self, prompt=prompt, system_prompt=system_prompt, reasoning=reasoning, schema=schema, **kwargs)

LocalOpenAILLMClient.json_output = patched_json_output

# ============================================================================
# Monkey-patch ToolManager to capture tool executor prompts
# ============================================================================
original_virtual_tool_executor = ToolManager._ToolManager__virtual_tool_executor

def patched_virtual_tool_executor(self, tool_name, params, schema):
    """Patched virtual tool executor that captures the prompt."""
    global captured_prompts
    
    # Build the prompt the same way the original does
    output_type = schema.get('output_type', 'unknown')
    output_description = schema.get('output_description', '')
    
    output_guidance = ""
    if output_type and output_type != 'unknown':
        output_guidance += f"\n- Expected Output Type: {output_type}"
    if output_description:
        output_guidance += f"\n- Output Description: {output_description}"
    
    prompt = f"""You are an expert function simulator. Based on the following function description and the provided arguments, simulate the execution of this function call.

Function Name: {tool_name}

Function Description: {schema["description"]}

Function Schema:
{json.dumps(schema, indent=2)}
{output_guidance}
Arguments Provided:
{json.dumps(params, indent=2)}

Current Date/Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} (Assume this is the time of execution)

Task:
Generate a plausible JSON response string that represents what the function '{tool_name}' would return if it were actually executed with the given arguments.
- Consider the function's description (e.g., does it fetch data, create something, authorize, search?).
- Consider the argument values (e.g., dates, search terms).
- IMPORTANT: If Output Type and Output Description are provided above, ensure your response matches those specifications exactly.
- If the function description mentions potential errors (like needing authorization for 'fetch_calendar_events'), sometimes simulate those error responses.
- If the function returns nothing on success (like 'create_calendar_event' or 'authorize_calendar_access'), return a JSON indicating success, like '{{"status": "success"}}' or an empty JSON object '{{}}'.
- For functions returning data (like 'fetch_calendar_events' or 'web_search'), generate realistic-looking example data formatted as a JSON string.
- Ensure your entire output is ONLY the JSON string, without any introductory text, explanations, or markdown formatting like ```json ... ```. Just the raw JSON string.
"""
    
    prompt_info = {
        "type": "Tool Execution Simulation",
        "tool_name": tool_name,
        "system_prompt": "You are an expert function simulator outputting only JSON strings.",
        "user_prompt": prompt,
        "params": params,
        "output_type": output_type,
        "output_description": output_description
    }
    captured_prompts.append(prompt_info)
    
    # Call original
    return original_virtual_tool_executor(self, tool_name, params, schema)

ToolManager._ToolManager__virtual_tool_executor = patched_virtual_tool_executor

import datetime


def print_separator(title, char="=", width=80):
    """Print a separator with a title."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}\n")


def print_prompt(prompt_info, idx, total):
    """Print a captured prompt in a readable format."""
    print(f"\n{'─' * 80}")
    print(f" PROMPT {idx + 1}/{total}: {prompt_info.get('type', 'Unknown')}")
    if 'tool_name' in prompt_info:
        print(f" Tool: {prompt_info['tool_name']}")
    print(f"{'─' * 80}")
    
    print("\n📝 SYSTEM PROMPT:")
    print("-" * 40)
    system = prompt_info.get('system_prompt')
    if system:
        print(system)
    else:
        print("(None)")
    
    print("\n📝 USER PROMPT:")
    print("-" * 40)
    user = prompt_info.get('user_prompt')
    if user:
        # Print with line numbers
        for i, line in enumerate(user.split('\n'), 1):
            print(f"{i:4d} | {line}")
    else:
        print("(None)")
    
    if prompt_info.get('schema'):
        print(f"\n📋 Expected Schema: {prompt_info['schema']}")
    
    if 'output_type' in prompt_info:
        print(f"\n📤 Expected Output Type: {prompt_info['output_type']}")
    if 'output_description' in prompt_info:
        print(f"📤 Output Description: {prompt_info['output_description']}")


def main():
    global captured_prompts
    captured_prompts = []
    
    print_separator("DEBUG SCRIPT: Capturing All LLM Prompts")
    
    # Configuration
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    tool_pool_path = "/home/ishalyminov/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl"
    temp_pool_dir = "/home/ishalyminov/data/APIGen-MT/data/generated"
    
    if not api_key or not api_base:
        print("ERROR: OPENAI_API_KEY or OPENAI_API_BASE not set in .env file.")
        sys.exit(1)
    
    # Initialize LLM client
    llm_client = LocalOpenAILLMClient(
        url=api_base,
        api_key=api_key,
        api_model="nvidia/nemotron-3-super-120b-a12b",
        hf_tokenizer_id=None,
    )
    
    # Load tool pool
    print(f"Loading tools from: {tool_pool_path}")
    
    tools = []
    with open(tool_pool_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 20:  # Use first 20 tools for this test
                break
            tools.append(json.loads(line.strip()))
    
    # Create temp tool pool
    os.makedirs(temp_pool_dir, exist_ok=True)
    temp_pool_path = os.path.join(temp_pool_dir, "debug_temp_tool_pool.jsonl")
    with open(temp_pool_path, 'w') as f:
        for tool in tools:
            f.write(json.dumps(tool) + '\n')
    
    print(f"Loaded {len(tools)} tools")
    
    # Show sample tool format
    print("\n📦 SAMPLE TOOL FORMAT:")
    print("-" * 40)
    sample_tool = tools[0]
    print(json.dumps(sample_tool, indent=2))
    
    # Initialize ToolManager
    tool_manager = ToolManager(llm=llm_client, tool_pool_path=temp_pool_path)
    
    # Get tool schemas to verify output fields are present
    schemas = tool_manager.get_tools_json_schema()
    print(f"\n📋 Tool schemas loaded: {len(schemas)}")
    
    # Show sample schema with output fields
    if schemas:
        print("\n📦 SAMPLE SCHEMA (with output fields):")
        print("-" * 40)
        print(json.dumps(schemas[0], indent=2))
    
    # Initialize generator
    phase1_generator = APIGenMTPhase1Generator(
        llm_client=llm_client,
        tool_manager=tool_manager,
        num_actions=2
    )
    
    # Generate a simple test query
    query = "Search for weather information for New York today, then create a calendar event based on the result."
    
    print_separator(f"GENERATING BLUEPRINT FOR QUERY")
    print(f"Query: {query}")
    print()
    
    # Generate blueprint (this will capture all prompts)
    verified_bp = phase1_generator.generate_verified_blueprint(query, max_attempts=1)
    
    # Print all captured prompts
    print_separator("CAPTURED PROMPTS SUMMARY")
    print(f"Total prompts captured: {len(captured_prompts)}")
    
    for i, p in enumerate(captured_prompts):
        print(f"\n{i+1}. {p.get('type', 'Unknown')}")
        if 'tool_name' in p:
            print(f"   Tool: {p['tool_name']}")
    
    # Print each prompt in detail
    print_separator("DETAILED PROMPT CONTENTS")
    
    for idx, prompt_info in enumerate(captured_prompts):
        print_prompt(prompt_info, idx, len(captured_prompts))
    
    # Print result
    print_separator("GENERATION RESULT")
    if verified_bp:
        print("✅ Blueprint generated successfully!")
        print(f"\nBlueprint Query: {verified_bp.blueprint.q}")
        print(f"Number of steps: {len(verified_bp.blueprint.a_gt_steps)}")
        print("\nTool calls:")
        for i, step in enumerate(verified_bp.blueprint.a_gt_steps):
            print(f"  Step {i+1}:")
            for tc in step.tool_calls:
                print(f"    - {tc.tool_name}({json.dumps(tc.arguments)})")
    else:
        print("❌ Blueprint generation failed")
    
    # Cleanup
    if os.path.exists(temp_pool_path):
        os.remove(temp_pool_path)
    
    print_separator("DEBUG COMPLETE")


if __name__ == "__main__":
    main()