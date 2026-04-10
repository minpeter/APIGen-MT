#!/usr/bin/env python3
"""
Example script showing enhanced datapoints with simulated execution traces.
"""

import json
from tool_simulation import simulate_execution_trace

# Example blueprint from generate_datapoints_bfcl.py
example_blueprint = {
    "q": "Using tools from Storage and Communication, retrieve a file and send its contents via message.",
    "a_gt_steps": [
        {
            "tool_calls": [
                {
                    "tool_name": "ls",
                    "arguments": {"path": "/workspace"}
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "tool_name": "cat",
                    "arguments": {"file_name": "report.txt"}
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "tool_name": "send_message",
                    "arguments": {
                        "receiver_id": "user_123",
                        "message": "Here's the report content: [content from cat]"
                    }
                }
            ]
        }
    ]
}

print("=" * 80)
print("ENHANCED DATAPOINT EXAMPLE")
print("=" * 80)
print()

# Create the enhanced datapoint structure
print("1. ORIGINAL BLUEPRINT:")
print("-" * 80)
print(json.dumps(example_blueprint, indent=2)[:300] + "...")
print()

# Simulate execution trace
print("2. SIMULATED EXECUTION TRACE:")
print("-" * 80)

tool_calls = []
for step in example_blueprint["a_gt_steps"]:
    for tc in step["tool_calls"]:
        tool_calls.append(tc)

execution_trace = simulate_execution_trace(tool_calls)

for step in execution_trace:
    print(f"\nStep {step['step_index']}: {step['function_name']}")
    print(f"  Arguments: {json.dumps(step['arguments'])}")
    print(f"  Simulated Return:")
    print(f"    {json.dumps(step['simulated_return'], indent=4)[:200]}...")

print()
print("=" * 80)
print("COMPLETE ENHANCED DATAPOINT STRUCTURE:")
print("=" * 80)

enhanced_datapoint = {
    "query": "Retrieve a file and send its contents via message.",
    "blueprint": example_blueprint,
    "simulated_execution_trace": execution_trace,
    "timestamp": "2025-01-15T10:30:00Z",
    "tools_used": ["ls", "cat", "send_message"],
    "categories_in_pool": ["Storage", "Communication"]
}

print(json.dumps(enhanced_datapoint, indent=2)[:800] + "...")

print()
print("=" * 80)
print("KEY ENHANCEMENTS:")
print("=" * 80)
print("✅ Blueprint still contains the original tool calls")
print("✅ NEW: simulated_execution_trace shows what each tool would return")
print("✅ Each trace step includes:")
print("   - step_index: Position in the execution flow")
print("   - function_name: Tool that was called")
print("   - arguments: Parameters passed to the tool")
print("   - simulated_return: Realistic return value based on tool type")
print("   - timestamp: When the execution occurred")
print()
print("This makes datapoints suitable for:")
print("  • Training models to predict tool outputs")
print("  • Understanding multi-step workflow execution")
print("  • Debugging tool interactions")
print("  • Building multi-turn conversation systems")
print("=" * 80)