#!/usr/bin/env python3
"""
Extract tool definitions and invocation examples from BFCL_v3 dataset.
Outputs two files:
1. bfcl_v3_tool_definitions.jsonl - Tool definitions in Magnet format
2. bfcl_v3_invocation_examples.jsonl - Tool invocations with actual arguments and outputs
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Any

# Import the parser from our existing code
import sys
sys.path.insert(0, str(Path(__file__).parent))
from parse_bfcl import parse_bfcl_func_doc, _resolve_category


def extract_tool_definitions(output_path: str = "bfcl_v3_tool_definitions.jsonl"):
    """Extract all tool definitions from BFCL_v3 multi_turn_func_doc."""
    
    func_doc_dir = Path("/home/ishalyminov/data/magnet_mt/data/BFCL_v3/multi_turn_func_doc")
    
    # Parse all function documentation
    definitions = parse_bfcl_func_doc(func_doc_dir, require_parameters=False)
    
    # Write to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for defn in definitions:
            f.write(json.dumps(defn.to_dict(), ensure_ascii=False) + '\n')
    
    print(f"✅ Extracted {len(definitions)} tool definitions to {output_path}")
    return definitions


def parse_function_call(call_str: str) -> dict:
    """Parse a function call string like "cd(folder='document')" into structured format."""
    
    # Match function name and arguments
    match = re.match(r'(\w+)\((.*)\)', call_str.strip())
    if not match:
        return None
    
    function_name = match.group(1)
    args_str = match.group(2)
    
    # Parse arguments
    args = {}
    if args_str:
        # Split by comma, but handle nested quotes/brackets
        # Simple approach: split by ',' followed by word boundary and '='
        arg_parts = []
        current = ""
        in_quotes = False
        quote_char = None
        paren_depth = 0
        
        for char in args_str:
            if char in '"\'':
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
            
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            
            if char == ',' and not in_quotes and paren_depth == 0:
                arg_parts.append(current.strip())
                current = ""
            else:
                current += char
        
        if current.strip():
            arg_parts.append(current.strip())
        
        # Parse each argument
        for arg in arg_parts:
            if '=' in arg:
                key, value = arg.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes from string values
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                # Try to parse as JSON for complex types
                try:
                    args[key] = json.loads(value)
                except:
                    args[key] = value
    
    return {
        "function": function_name,
        "arguments": args
    }


def extract_invocation_examples(output_path: str = "bfcl_v3_invocation_examples.jsonl"):
    """Extract tool invocation examples with actual arguments and outputs."""
    
    data_dir = Path("/home/ishalyminov/data/magnet_mt/data/BFCL_v3")
    
    # Focus on multi-turn datasets as they have the best examples
    test_files = [
        "BFCL_v3_multi_turn_base.json",
        "BFCL_v3_multi_turn_composite.json",
        "BFCL_v3_multi_turn_long_context.json",
    ]
    
    examples = []
    example_id = 0
    
    for test_file in test_files:
        test_path = data_dir / test_file
        answer_path = data_dir / "possible_answer" / test_file
        
        if not test_path.exists():
            print(f"⚠️  Skipping {test_file} - not found")
            continue
        
        print(f"📄 Processing {test_file}...")
        
        # Load test data
        test_data = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
        
        # Load ground truth answers
        answers = {}
        if answer_path.exists():
            with open(answer_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        ans = json.loads(line)
                        answers[ans['id']] = ans['ground_truth']
        
        # Process each test case
        for test_case in test_data:
            case_id = test_case.get('id', '')
            question = test_case.get('question', [])
            ground_truth = answers.get(case_id, [])
            initial_config = test_case.get('initial_config', {})
            involved_classes = test_case.get('involved_classes', [])
            
            # Extract invocation examples from ground truth
            for turn_idx, turn_calls in enumerate(ground_truth):
                # Safely get user message
                if turn_idx < len(question) and len(question[turn_idx]) > 0:
                    user_message = question[turn_idx][0].get('content', '')
                else:
                    user_message = ''
                
                for call_str in turn_calls:
                    parsed = parse_function_call(call_str)
                    if not parsed:
                        continue
                    
                    function_name = parsed['function']
                    arguments = parsed['arguments']
                    
                    # Determine category and tool_name
                    # Try to find the class this function belongs to
                    tool_name = None
                    category = None
                    
                    # Check function doc to get category
                    func_doc_dir = data_dir / "multi_turn_func_doc"
                    for json_file in func_doc_dir.glob("*.json"):
                        with open(json_file, 'r') as f:
                            content = f.read()
                            # Check if function name appears in this file
                            if f'"name": "{function_name}"' in content or f'"name":"{function_name}"' in content:
                                tool_name = json_file.stem
                                category = _resolve_category(tool_name)
                                break
                    
                    if not tool_name:
                        # Fallback: use involved classes
                        if involved_classes:
                            tool_name = involved_classes[0]
                            category = _resolve_category(tool_name)
                    
                    example = {
                        "id": f"{case_id}_turn{turn_idx}_{function_name}",
                        "test_case_id": case_id,
                        "turn_index": turn_idx,
                        "category": category,
                        "tool_name": tool_name,
                        "function_name": function_name,
                        "arguments": arguments,
                        "user_message": user_message,
                        "call_string": call_str,
                        "initial_config": initial_config,
                        "involved_classes": involved_classes
                    }
                    
                    examples.append(example)
                    example_id += 1
    
    # Write to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✅ Extracted {len(examples)} invocation examples to {output_path}")
    return examples


def generate_statistics(definitions, examples):
    """Generate comprehensive statistics."""
    
    print("\n" + "=" * 80)
    print("EXTRACTION STATISTICS")
    print("=" * 80)
    
    # Tool definition statistics
    print(f"\n📦 TOOL DEFINITIONS")
    print(f"   Total tools: {len(definitions)}")
    
    categories = defaultdict(int)
    for defn in definitions:
        categories[defn.category] += 1
    
    print(f"   Categories: {len(categories)}")
    for cat, count in sorted(categories.items()):
        print(f"      {cat:30} {count:3} tools")
    
    # Invocation example statistics
    print(f"\n📝 INVOCATION EXAMPLES")
    print(f"   Total examples: {len(examples)}")
    
    function_counts = defaultdict(int)
    for ex in examples:
        function_counts[ex['function_name']] += 1
    
    print(f"   Unique functions: {len(function_counts)}")
    print(f"\n   Top 10 most called functions:")
    for func, count in sorted(function_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"      {func:30} {count:3} calls")
    
    # Arguments statistics
    args_counts = []
    for ex in examples:
        args_counts.append(len(ex['arguments']))
    
    if args_counts:
        print(f"\n   Argument statistics:")
        print(f"      Average args per call: {sum(args_counts)/len(args_counts):.2f}")
        print(f"      Min args: {min(args_counts)}")
        print(f"      Max args: {max(args_counts)}")
        print(f"      Calls with 0 args: {args_counts.count(0)}")
        print(f"      Calls with 1 arg: {args_counts.count(1)}")
        print(f"      Calls with 2+ args: {sum(1 for c in args_counts if c >= 2)}")


def create_sample_outputs_file(definitions, examples):
    """Create a human-readable sample outputs file."""
    
    output_path = "bfcl_v3_samples_human_readable.md"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# BFCL_v3 Extraction - Sample Outputs\n\n")
        f.write("This file contains sample tool definitions and invocation examples extracted from BFCL_v3.\n\n")
        
        # Part 1: Tool Definitions
        f.write("## Part 1: Tool Definitions\n\n")
        f.write(f"Total: {len(definitions)} tools\n\n")
        
        # Group by category
        by_category = defaultdict(list)
        for defn in definitions:
            by_category[defn.category].append(defn)
        
        for category in sorted(by_category.keys()):
            tools = by_category[category]
            f.write(f"### {category} ({len(tools)} tools)\n\n")
            
            for tool in tools[:3]:  # Show first 3
                f.write(f"#### {tool.api_name}\n\n")
                f.write(f"**Description**: {tool.api_description[:150]}...\n\n")
                f.write(f"**Parameters**:\n")
                
                if tool.parameters.required:
                    f.write(f"- Required: {', '.join(tool.parameters.required)}\n")
                if tool.parameters.optional:
                    f.write(f"- Optional: {', '.join(tool.parameters.optional)}\n")
                
                f.write(f"\n**Example**:\n```json\n")
                f.write(json.dumps(tool.to_dict(), indent=2))
                f.write("\n```\n\n")
            
            if len(tools) > 3:
                f.write(f"... and {len(tools) - 3} more tools\n\n")
        
        # Part 2: Invocation Examples
        f.write("## Part 2: Invocation Examples\n\n")
        f.write(f"Total: {len(examples)} examples\n\n")
        
        # Show diverse examples
        shown_functions = set()
        example_count = 0
        
        for ex in examples:
            if ex['function_name'] in shown_functions:
                continue
            if example_count >= 20:  # Limit to 20 examples
                break
            
            shown_functions.add(ex['function_name'])
            example_count += 1
            
            f.write(f"### Example {example_count}: {ex['function_name']}\n\n")
            f.write(f"**User Message**: {ex['user_message'][:200]}...\n\n")
            f.write(f"**Category**: {ex['category']}\n")
            f.write(f"**Tool**: {ex['tool_name']}\n\n")
            
            f.write(f"**Function Call**:\n```\n{ex['call_string']}\n```\n\n")
            
            f.write(f"**Arguments**:\n```json\n")
            f.write(json.dumps(ex['arguments'], indent=2))
            f.write("\n```\n\n")
            
            if ex['initial_config']:
                f.write(f"**Initial Config**: {', '.join(ex['involved_classes'])}\n\n")
            
            f.write("---\n\n")
    
    print(f"✅ Created human-readable samples file: {output_path}")


def main():
    """Main extraction function."""
    
    print("=" * 80)
    print("BFCL_v3 TOOL EXTRACTION")
    print("=" * 80)
    
    # Step 1: Extract tool definitions
    print("\n📦 Step 1: Extracting tool definitions...")
    definitions = extract_tool_definitions("bfcl_v3_tool_definitions.jsonl")
    
    # Step 2: Extract invocation examples
    print("\n📝 Step 2: Extracting invocation examples...")
    examples = extract_invocation_examples("bfcl_v3_invocation_examples.jsonl")
    
    # Step 3: Generate statistics
    generate_statistics(definitions, examples)
    
    # Step 4: Create human-readable samples
    print("\n📄 Step 3: Creating human-readable samples...")
    create_sample_outputs_file(definitions, examples)
    
    print("\n" + "=" * 80)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 80)
    print("\nOutput files:")
    print("  1. bfcl_v3_tool_definitions.jsonl      - Tool definitions in Magnet format")
    print("  2. bfcl_v3_invocation_examples.jsonl   - Tool invocations with arguments")
    print("  3. bfcl_v3_samples_human_readable.md   - Human-readable samples")
    print("=" * 80)


if __name__ == "__main__":
    main()