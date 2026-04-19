#!/usr/bin/env python3
"""Add --frequent argument and filtering logic"""

import sys

# Read the file
with open('extract_bfcl_with_outputs.py', 'r') as f:
    lines = f.readlines()

# 1. Add the --frequent argument after --limit
insert_index = None
for i, line in enumerate(lines):
    if 'help="Limit number of tools to process (for testing)"' in line:
        # Find the closing parenthesis
        for j in range(i, min(i+5, len(lines))):
            if lines[j].strip() == ')':
                insert_index = j + 1
                break
        break

if insert_index is None:
    print("❌ Could not find --limit argument")
    sys.exit(1)

print(f"1. Adding --frequent argument at line {insert_index}")

# Add the --frequent argument with correct indentation
new_arg = [
    '    parser.add_argument(\n',
    '        "--frequent",\n',
    '        action="store_true",\n',
    '        help="Process top 100 most frequent tools (from frequency analysis)"\n',
    '    )\n',
]

lines[insert_index:insert_index] = new_arg

# 2. Add the filtering logic after the limit check
insert_index = None
for i, line in enumerate(lines):
    if 'Limiting to {args.limit} tools for testing' in line:
        # Find the closing parenthesis of the print statement
        for j in range(i, min(i+5, len(lines))):
            if ')' in lines[j]:
                insert_index = j + 1
                break
        break

if insert_index is None:
    print("❌ Could not find limit check")
    sys.exit(1)

print(f"2. Adding filtering logic at line {insert_index}")

# Add the filtering logic with correct indentation (4 spaces for main() level)
filter_code = [
    '\n',
    '    # Filter by most frequent tools\n',
    '    if args.frequent:\n',
    '        frequent_tools_file = Path(__file__).parent / "top_100_frequent_tools.txt"\n',
    '        if frequent_tools_file.exists():\n',
    '            with open(frequent_tools_file, \'r\') as f:\n',
    '                frequent_tool_names = [line.strip() for line in f if line.strip()]\n',
    '            \n',
    '            # Filter tools to only include frequent ones\n',
    '            filtered_tools = []\n',
    '            for tool in tools:\n',
    '                tool_name = tool.get(\'tool_name\', \'\')\n',
    '                api_name = tool.get(\'api_name\', \'\')\n',
    '                full_name = f"{tool_name}.{api_name}" if tool_name else api_name\n',
    '            \n',
    '                if full_name in frequent_tool_names:\n',
    '                    filtered_tools.append(tool)\n',
    '            \n',
    '            tools = filtered_tools\n',
    '            print(f"\\n📊 Filtered to {len(tools)} most frequent tools")\n',
    '        else:\n',
    '            print(f"\\n⚠️ Warning: {frequent_tools_file} not found, ignoring --frequent flag")\n',
]

lines[insert_index:insert_index] = filter_code

# Write the file
with open('extract_bfcl_with_outputs.py', 'w') as f:
    f.writelines(lines)

print(f"✅ Added {len(new_arg) + len(filter_code)} lines")

# Verify syntax
import ast
try:
    with open('extract_bfcl_with_outputs.py', 'r') as f:
        ast.parse(f.read())
    print("✅ Syntax is valid")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
    sys.exit(1)

print("\n🎉 Successfully added --frequent argument and filtering logic")