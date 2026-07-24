#!/usr/bin/env python3
"""
Sample extraction demonstration showing how the Magnet tool pool extraction
works with BFCL_v3 data.

This script demonstrates:
1. Loading BFCL_v3 function documentation
2. Parsing and transforming to Magnet canonical format
3. Displaying sample outputs
"""

import json
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from parse_bfcl import parse_bfcl_func_doc, discover_bfcl_classes
from tool_definition import ToolDefinition


def print_separator(title: str = ""):
    """Print a visual separator."""
    print("\n" + "=" * 80)
    if title:
        print(f" {title}")
        print("=" * 80)


def print_json(data: dict):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    """Run the sample extraction demonstration."""
    
    # Use the actual BFCL_v3 data path
    bfcl_data_dir = Path("~/data/magnet_mt/data/BFCL_v3").expanduser()
    func_doc_dir = bfcl_data_dir / "multi_turn_func_doc"
    
    print_separator("MAGNET TOOL POOL EXTRACTION - BFCL_v3 DEMONSTRATION")
    
    # Step 1: Discover classes from test files
    print("\n1. DISCOVERING TOOL CLASSES FROM TEST FILES")
    print("-" * 80)
    
    discovered_classes = discover_bfcl_classes(bfcl_data_dir)
    print(f"\nFound {len(discovered_classes)} tool classes (from test files):")
    for cls in sorted(discovered_classes):
        print(f"  • {cls}")
    
    # Convert PascalCase class names to snake_case to match file names
    import re
    def to_snake_case(name):
        """Convert PascalCase to snake_case."""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    class_names = [to_snake_case(cls) for cls in discovered_classes]
    print(f"\nNormalized to file name format:")
    for cls in sorted(class_names):
        print(f"  • {cls}")
    
    # Step 2: Parse function documentation
    print_separator("2. PARSING FUNCTION DOCUMENTATION")
    
    definitions = parse_bfcl_func_doc(
        func_doc_dir,
        class_names=class_names,  # Use normalized names
        require_parameters=True
    )
    
    print(f"\nExtracted {len(definitions)} function definitions")
    
    # Step 3: Show sample outputs
    print_separator("3. SAMPLE OUTPUTS BY CATEGORY")
    
    # Group by category
    from collections import defaultdict
    by_category = defaultdict(list)
    for defn in definitions:
        by_category[defn.category].append(defn)
    
    # Show samples from each category
    for category in sorted(by_category.keys()):
        tools = by_category[category]
        print(f"\n{category.upper()} ({len(tools)} functions)")
        print("-" * 80)
        
        # Show first 2 examples from each category
        for i, defn in enumerate(tools[:2], 1):
            print(f"\n  Example {i}: {defn.api_name}")
            print("  " + "-" * 76)
            print_json(defn.to_dict())
    
    # Step 4: Detailed analysis of specific examples
    print_separator("4. DETAILED EXAMPLE: GORILLA FILE SYSTEM")
    
    gorilla_tools = [d for d in definitions if d.tool_name == "gorilla_file_system"]
    print(f"\nGorilla File System contains {len(gorilla_tools)} functions:")
    
    # Show a function with required parameters
    cat_func = next((d for d in gorilla_tools if d.api_name == "cat"), None)
    if cat_func:
        print("\n  Function: cat (with required parameters)")
        print("  " + "-" * 76)
        print_json(cat_func.to_dict())
    
    # Show a function with optional parameters
    find_func = next((d for d in gorilla_tools if d.api_name == "find"), None)
    if find_func:
        print("\n  Function: find (with optional parameters)")
        print("  " + "-" * 76)
        print_json(find_func.to_dict())
    
    # Step 5: Statistics
    print_separator("5. EXTRACTION STATISTICS")
    
    print("\nFunctions by Category:")
    for category in sorted(by_category.keys()):
        print(f"  {category:<30} {len(by_category[category]):>3} functions")
    
    print(f"\n{'TOTAL':<30} {len(definitions):>3} functions")
    
    # Count parameters
    total_required = sum(len(d.parameters.required) for d in definitions)
    total_optional = sum(len(d.parameters.optional) for d in definitions)
    
    print(f"\nParameter Statistics:")
    print(f"  Total required parameters:  {total_required}")
    print(f"  Total optional parameters:  {total_optional}")
    print(f"  Average required per API:   {total_required/len(definitions):.2f}")
    print(f"  Average optional per API:   {total_optional/len(definitions):.2f}")
    
    # Functions with no required params
    no_required = [d for d in definitions if not d.parameters.required]
    print(f"\n  Functions with no required params: {len(no_required)}")
    
    # Functions with optional params
    with_optional = [d for d in definitions if d.parameters.optional]
    print(f"  Functions with optional params:    {len(with_optional)}")
    
    print_separator("EXTRACTION COMPLETE")
    
    return definitions


if __name__ == "__main__":
    definitions = main()