# Magnet Tool Pool Extraction Scripts

This directory contains the tool pool extraction scripts from the Magnet paper (arXiv 2503.07826), adapted for use with APIGen-MT.

## Overview

The Magnet paper describes a systematic approach to collecting tool definitions from multiple sources to create a comprehensive tool pool for multi-turn dataset generation. This implementation extracts tool definitions from the **BFCL_v3 (Berkeley Function Calling Leaderboard)** dataset and transforms them into the Magnet canonical format.

## Files

### Core Scripts

- **`tool_definition.py`** - Data model for the Magnet canonical format
- **`parse_bfcl.py`** - Parser for BFCL-v3 multi-turn function documentation
- **`parse_stabletoolbench.py`** - Parser for StableToolBench/ToolEnv2404 (original)
- **`collect_tools.py`** - Main orchestration script for collecting all tools
- **`sample_extraction.py`** - Demonstration script with sample outputs

### Documentation

- **`ANALYSIS.md`** - Comprehensive analysis with sample outputs from BFCL_v3
- **`README.md`** - This file

## Quick Start

### Run Sample Extraction

```bash
python sample_extraction.py
```

This will:
1. Discover tool classes from BFCL_v3 test files
2. Parse function documentation from multi_turn_func_doc/
3. Display sample outputs by category
4. Show detailed examples with parameter analysis
5. Print extraction statistics

### Collect All Tools

```bash
python collect_tools.py \
  --bfcl-func-doc /path/to/BFCL_v3/multi_turn_func_doc \
  --bfcl-data-dir /path/to/BFCL_v3 \
  --output tool_pool.jsonl \
  --stats
```

## Magnet Canonical Format

Each tool definition follows this structure:

```json
{
  "category": "Storage",
  "tool_name": "gorilla_file_system",
  "tool_description": "Functions provided by the gorilla file system toolkit.",
  "api_name": "cat",
  "api_description": "Display the contents of a file...",
  "parameters": {
    "type": "dict",
    "properties": {
      "file_name": {
        "type": "string",
        "description": "The name of the file..."
      }
    },
    "required": ["file_name"],
    "optional": []
  }
}
```

## Sample Outputs

### Category Distribution

The extraction from BFCL_v3 yields:

- **Storage** (17 functions) - File system operations (cat, ls, cd, etc.)
- **Science** (17 functions) - Mathematical operations (add, subtract, etc.)
- **Finance** (16 functions) - Trading bot operations (buy_stock, sell_stock, etc.)
- **Events** (7 functions) - Ticket management system
- **Communication** (6 functions) - Message and contact management

### Example: Storage Category

**Function: cat**
```json
{
  "category": "Storage",
  "tool_name": "gorilla_file_system",
  "api_name": "cat",
  "api_description": "Display the contents of a file of any extension from current directory.",
  "parameters": {
    "type": "dict",
    "properties": {
      "file_name": {
        "type": "string",
        "description": "The name of the file from current directory to display."
      }
    },
    "required": ["file_name"],
    "optional": []
  }
}
```

**Function: find**
```json
{
  "category": "Storage",
  "tool_name": "gorilla_file_system",
  "api_name": "find",
  "api_description": "Find any file or directories under specific path...",
  "parameters": {
    "type": "dict",
    "properties": {
      "path": {
        "type": "string",
        "description": "The directory path to start the search.",
        "default": "."
      },
      "name": {
        "type": "string",
        "description": "The name of the file or directory to search for.",
        "default": "None"
      }
    },
    "required": [],
    "optional": ["path", "name"]
  }
}
```

## Key Features

### 1. Class Discovery
- Automatically discovers tool classes from test files
- Maps PascalCase to snake_case for file matching
- Filters to only include referenced classes

### 2. Category Mapping
```python
_BFCL_CATEGORY_MAP = {
    "gorilla_file_system": "Storage",
    "trading_bot": "Finance",
    "ticket_api": "Events",
    "weather_api": "Weather",
    "math_api": "Science",
    "message_api": "Communication",
    "calendar_api": "Business_Software",
}
```

### 3. Parameter Normalization
- Converts OpenAI-style `"type": "object"` to `"type": "dict"`
- Separates `required` and `optional` parameters
- Preserves default values and nested schemas

### 4. Quality Filtering
- Excludes functions with no parameters (by default)
- Validates required fields before extraction
- Logs skipped entries for transparency

## Extraction Statistics

From BFCL_v3 multi-turn dataset:

```
Functions by Category:
  Communication                 6 functions
  Events                        7 functions
  Finance                      16 functions
  Science                      17 functions
  Storage                      17 functions

TOTAL                          63 functions

Parameter Statistics:
  Total required parameters:    89
  Total optional parameters:    14
  Average required per API:     1.41
  Average optional per API:     0.22
  
  Functions with no required params: 5
  Functions with optional params:    11
```

## Transformation Examples

### From BFCL to Magnet Format

**Input (BFCL original):**
```json
{
  "name": "buy_stock",
  "description": "Buy a stock with a given symbol and quantity.",
  "parameters": {
    "type": "dict",
    "properties": {
      "symbol": {"type": "string", "description": "The stock symbol to buy."},
      "quantity": {"type": "integer", "description": "The quantity of stocks to buy."}
    },
    "required": ["symbol", "quantity"]
  }
}
```

**Output (Magnet canonical):**
```json
{
  "category": "Finance",
  "tool_name": "trading_bot",
  "tool_description": "Functions provided by the trading bot toolkit.",
  "api_name": "buy_stock",
  "api_description": "Buy a stock with a given symbol and quantity.",
  "parameters": {
    "type": "dict",
    "properties": {
      "symbol": {"type": "string", "description": "The stock symbol to buy."},
      "quantity": {"type": "integer", "description": "The quantity of stocks to buy."}
    },
    "required": ["symbol", "quantity"],
    "optional": []
  }
}
```

## Integration with APIGen-MT

### Use Cases

1. **Tool Pool for Dataset Generation**
   - Use extracted tool definitions as input for multi-turn conversation generation
   - Leverage categorization for domain-specific tool selection

2. **Schema Validation**
   - Apply the Magnet canonical format as a validation standard
   - Ensure consistency across different tool sources

3. **Tool Discovery**
   - Use category information to select relevant tools per domain
   - Filter tools based on parameter complexity

### Example Workflow

```python
from magnet_tool_extraction.parse_bfcl import parse_bfcl_func_doc
from magnet_tool_extraction.tool_definition import ToolDefinition

# Load tool definitions
definitions = parse_bfcl_func_doc(
    func_doc_dir="path/to/multi_turn_func_doc",
    require_parameters=True
)

# Filter by category
storage_tools = [d for d in definitions if d.category == "Storage"]

# Convert to your format
for tool in storage_tools:
    print(f"{tool.api_name}: {tool.api_description}")
```

## Technical Notes

### JSONL Format Handling

The parser handles both formats:
- JSON arrays: `[{"name": "func1", ...}, {"name": "func2", ...}]`
- JSONL: One JSON object per line

### Class Name Conversion

BFCL test files use PascalCase (e.g., `GorillaFileSystem`) while function doc files use snake_case (e.g., `gorilla_file_system.json`). The extraction script automatically handles this conversion.

### Response Schema

The original BFCL format includes a `response` field that describes the return value. This is preserved in the data but not included in the Magnet canonical output, which focuses on the function signature.

## References

- **Magnet Paper**: [arXiv 2503.07826](https://arxiv.org/abs/2503.07826)
- **BFCL Dataset**: [gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
- **StableToolBench**: [stabletoolbench/ToolEnv2404](https://huggingface.co/datasets/stabletoolbench/ToolEnv2404)

## License

These scripts are provided for research purposes, following the terms of the original Magnet paper and associated datasets.