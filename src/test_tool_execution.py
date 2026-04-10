#!/usr/bin/env python3
"""
Test script to verify tool execution is added to datapoints.
"""

import json
import os
import sys

# Load environment variables manually
env_path = '/home/ishalyminov/data/APIGen-MT/.env'
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_client import LocalOpenAILLMClient
from tool_manager import ToolManager

# Import APIGenMTPhase1Generator
import importlib.util
spec = importlib.util.spec_from_file_location(
    "apigen_phase1",
    os.path.join(os.path.dirname(__file__), "apigen-phase1.py")
)
apigen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apigen_module)
APIGenMTPhase1Generator = apigen_module.APIGenMTPhase1Generator


def test_tool_execution():
    """Test that tool execution works and outputs are captured."""
    
    print("=" * 70)
    print("Testing Tool Execution in Datapoint Generation")
    print("=" * 70)
    
    # Configuration
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    tool_pool_path = "/home/ishalyminov/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_all_tool_definitions.jsonl"
    
    if not api_key or not api_base:
        print("ERROR: Missing API credentials")
        return False
    
    # Initialize LLM client
    llm_client = LocalOpenAILLMClient(
        url=api_base,
        api_key=api_key,
        api_model="nvidia/nemotron-3-super-120b-a12b",
        hf_tokenizer_id=None,
    )
    
    # Load a small subset of tools
    print("\n1. Loading tool pool...")
    tools = []
    with open(tool_pool_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 80:  # Use 80 tools like the main script
                break
            tools.append(json.loads(line.strip()))
    
    print(f"   Loaded {len(tools)} tools")
    
    # Create temp tool pool
    temp_pool = "/tmp/test_tool_execution_pool.jsonl"
    with open(temp_pool, 'w') as f:
        for tool in tools:
            f.write(json.dumps(tool) + '\n')
    
    # Initialize tool manager
    print("\n2. Initializing ToolManager...")
    tool_manager = ToolManager(llm=llm_client, tool_pool_path=temp_pool)
    phase1_generator = APIGenMTPhase1Generator(llm_client=llm_client, tool_manager=tool_manager)
    
    # Test query
    query = "Get weather data for New York and then create a weather alert"
    
    print(f"\n3. Generating blueprint for query:")
    print(f"   '{query}'")
    print()
    
    # Generate blueprint
    verified_bp = phase1_generator.generate_verified_blueprint(query, max_attempts=1)
    
    if not verified_bp:
        print("   ✗ Failed to generate blueprint")
        return False
    
    print(f"   ✓ Blueprint generated")
    print(f"   Blueprint query: {verified_bp.blueprint.q}")
    print(f"   Steps: {len(verified_bp.blueprint.a_gt_steps)}")
    
    # Execute tools
    print(f"\n4. Executing tool calls...")
    tool_execution_results = []
    
    for step_idx, step in enumerate(verified_bp.blueprint.a_gt_steps):
        print(f"\n   Step {step_idx + 1}:")
        step_results = {
            "step_index": step_idx,
            "tool_calls": []
        }
        
        for tool_call in step.tool_calls:
            print(f"     Tool: {tool_call.tool_name}")
            print(f"     Arguments: {json.dumps(tool_call.arguments)}")
            
            try:
                # Execute the tool
                output = tool_manager.invoke_tool(tool_call.tool_name, tool_call.arguments)
                
                step_results["tool_calls"].append({
                    "tool_name": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                    "output": output
                })
                
                # Show truncated output
                output_str = json.dumps(output) if isinstance(output, dict) else str(output)
                print(f"     Output: {output_str[:150]}...")
                
            except Exception as e:
                print(f"     Error: {e}")
                step_results["tool_calls"].append({
                    "tool_name": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                    "output": {"error": str(e)}
                })
        
        tool_execution_results.append(step_results)
    
    # Create datapoint
    print(f"\n5. Creating datapoint with tool execution results...")
    datapoint = {
        "query": query,
        "blueprint": verified_bp.blueprint.model_dump(),
        "validation_result": verified_bp.validation_result.model_dump(),
        "llm_review_history": [review.model_dump() for review in verified_bp.llm_review_history],
        "generation_attempts": verified_bp.generation_attempts,
        "tools_used": list(set(
            tc.tool_name
            for step in verified_bp.blueprint.a_gt_steps
            for tc in step.tool_calls
        )),
        "tool_execution_results": tool_execution_results
    }
    
    # Save to file
    output_file = "/tmp/test_datapoint_with_execution.json"
    with open(output_file, 'w') as f:
        json.dump(datapoint, f, indent=2)
    
    print(f"   ✓ Datapoint saved to: {output_file}")
    
    # Show structure
    print(f"\n6. Datapoint structure:")
    print(f"   - query: '{datapoint['query']}'")
    print(f"   - blueprint: {len(datapoint['blueprint']['a_gt_steps'])} steps")
    print(f"   - tools_used: {datapoint['tools_used']}")
    print(f"   - tool_execution_results: {len(datapoint['tool_execution_results'])} step results")
    
    # Show first tool execution result
    if datapoint['tool_execution_results']:
        print(f"\n7. Example tool execution result:")
        first_result = datapoint['tool_execution_results'][0]
        print(f"   Step {first_result['step_index']}:")
        for tc in first_result['tool_calls']:
            print(f"     - {tc['tool_name']}: output captured = {bool(tc['output'])}")
    
    print(f"\n" + "=" * 70)
    print("✓ Test completed successfully!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_tool_execution()
    sys.exit(0 if success else 1)