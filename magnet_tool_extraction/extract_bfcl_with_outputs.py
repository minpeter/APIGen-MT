#!/usr/bin/env python3
"""
Enhanced BFCL tool extraction with LLM-predicted output fields.
Extracts tool definitions and predicts output_type and output_description using LLM.
Downloads BFCL_v3 if it doesn't exist.

This script only works with:
- BFCL_v3 (version is hardcoded)
- NVIDIA LLM client (client type is hardcoded)
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

# Import existing modules
sys.path.insert(0, str(Path(__file__).parent))
from parse_bfcl import parse_bfcl_func_doc
from download_bfcl_v4 import download_bfcl_v4
from llm_output_predictor import predict_outputs_for_tools

# Fixed constants - no configuration options
BFCL_VERSION = "v3"
LLM_CLIENT_TYPE = "nvidia"


def extract_tools_from_bfcl(
    bfcl_data_path: Path,
    verbose: bool = True
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract tool definitions and invocations from BFCL_v3 data.

    Args:
        bfcl_data_path: Path to BFCL data directory
        verbose: Print progress information

    Returns:
        Tuple of (tool_definitions, invocations)
    """
    if verbose:
        print(f"\n📦 Step 1: Extracting tool definitions from multi_turn_func_doc...")

    func_doc_path = bfcl_data_path / "multi_turn_func_doc"

    if func_doc_path.exists():
        # Convert ToolDefinition objects to dicts
        tool_defs = parse_bfcl_func_doc(func_doc_path)
        tool_definitions = []
        for td in tool_defs:
            tool_dict = {
                'category': td.category,
                'tool_name': td.tool_name,
                'tool_description': td.tool_description,
                'api_name': td.api_name,
                'api_description': td.api_description,
                'parameters': td.parameters.to_dict() if hasattr(td.parameters, 'to_dict') else td.parameters,
            }
            tool_definitions.append(tool_dict)

        if verbose:
            print(f" ✅ Extracted {len(tool_definitions)} tool definitions")
    else:
        tool_definitions = []
        if verbose:
            print(f" ⚠️ No multi_turn_func_doc directory found")

    # Extract invocations from test files
    if verbose:
        print(f"\n📝 Step 2: Extracting invocation examples...")

    invocations = []

    # Find all test JSON files for v3
    test_files_pattern = f"BFCL_{BFCL_VERSION}_multi_turn_*.json"
    test_files = list(bfcl_data_path.glob(test_files_pattern))

    if not test_files:
        # Try without version prefix
        test_files = list(bfcl_data_path.glob("multi_turn_*.json"))

    for test_file in test_files:
        if verbose:
            print(f" 📄 Processing {test_file.name}...")

        try:
            with open(test_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        # Extract invocations from test cases
                        if 'test_case' in data:
                            test_case = data['test_case']

                            # Process each turn in the test case
                            if isinstance(test_case, list):
                                for turn in test_case:
                                    if isinstance(turn, dict):
                                        invocation = {
                                            'tool_name': data.get('tool_name', 'unknown'),
                                            'user_message': turn.get('user_message', ''),
                                            'assistant_message': turn.get('assistant_message', ''),
                                            'tool_calls': turn.get('tool_calls', [])
                                        }
                                        invocations.append(invocation)

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            if verbose:
                print(f" ⚠️ Error processing {test_file.name}: {e}")

    if verbose:
        print(f" ✅ Extracted {len(invocations)} invocation examples")

    return tool_definitions, invocations


def main():
    """Main entry point"""
    # Load environment variables
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Extract BFCL_v3 tools with LLM-predicted output fields (NVIDIA client only)"
    )
    parser.add_argument(
        "--data-dir",
        default="/home/ishalyminov/data/magnet_mt/data",
        help="Base data directory (default: /home/ishalyminov/data/magnet_mt/data)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: bfcl_v3_tools_with_outputs.jsonl)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for LLM"
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=5,
        help="Maximum invocation contexts per tool (default: 5)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of tools to process (for testing)"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of BFCL data"
    )

    args = parser.parse_args()

    # Ensure BFCL data is available, download if needed
    data_dir = Path(args.data_dir)

    print(f"\n{'='*80}")
    print(f"BFCL_{BFCL_VERSION.upper()} TOOL EXTRACTION WITH OUTPUT PREDICTION")
    print(f"{'='*80}")

    # Download BFCL if needed
    bfcl_data_path = download_bfcl_v4(
        data_dir=data_dir,
        force_download=args.force_download
    )

    # Extract tools and invocations
    tools, invocations = extract_tools_from_bfcl(
        bfcl_data_path=bfcl_data_path,
        verbose=True
    )

    if not tools:
        print(f"\n❌ No tools found in {bfcl_data_path}")
        sys.exit(1)

    # Limit for testing
    if args.limit:
        tools = tools[:args.limit]
        print(f"\n⚠️ Limiting to {args.limit} tools for testing")

    # Predict outputs using LLM (always uses NVIDIA client)
    enhanced_tools = predict_outputs_for_tools(
        tools=tools,
        invocations=invocations,
        client_type=LLM_CLIENT_TYPE,  # Hardcoded to "nvidia"
        debug=args.debug,
        max_contexts=args.max_contexts
    )

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"bfcl_{BFCL_VERSION}_tools_with_outputs.jsonl")

    # Save results
    print(f"\n{'='*80}")
    print("💾 SAVING RESULTS")
    print(f"{'='*80}")

    with open(output_path, 'w') as f:
        for tool in enhanced_tools:
            f.write(json.dumps(tool) + '\n')

    print(f"✅ Saved {len(enhanced_tools)} enhanced tools to {output_path}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("📊 SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total tools extracted: {len(enhanced_tools)}")
    print(f"Total invocations available: {len(invocations)}")

    # Count tools with invocation contexts
    tools_with_contexts = sum(
        1 for tool in enhanced_tools
        if any(inv.get('tool_name') == tool.get('tool_name') for inv in invocations)
    )
    print(f"Tools with invocation contexts: {tools_with_contexts}")

    # Sample output types
    output_types = {}
    for tool in enhanced_tools:
        otype = tool.get('output_type', 'unknown')
        output_types[otype] = output_types.get(otype, 0) + 1

    print(f"\nOutput types distribution:")
    for otype, count in sorted(output_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f" {otype:30s}: {count:3d} tools")

    print(f"\n{'='*80}")
    print("✅ EXTRACTION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()