# Magnet Tool Pool Extraction - Summary

## What Was Done

Successfully analyzed and copied the tool pool extraction scripts from the Magnet paper (arXiv 2503.07826) to APIGen-MT.

## Files Added

### Core Scripts (from Magnet)
1. **`tool_definition.py`** - Data model for Magnet canonical format
2. **`parse_bfcl.py`** - Parser for BFCL-v3 function documentation (enhanced to handle JSONL)
3. **`parse_stabletoolbench.py`** - Parser for StableToolBench/ToolEnv2404
4. **`collect_tools.py`** - Main orchestration script

### Demonstration Scripts
5. **`sample_extraction.py`** - Comprehensive demonstration with sample outputs
6. **`usage_examples.py`** - Multiple usage examples for different scenarios

### Documentation
7. **`ANALYSIS.md`** - Comprehensive analysis with detailed sample outputs from BFCL_v3
8. **`README.md`** - Quick start guide and reference
9. **`SUMMARY.md`** - This file

## Key Findings

### BFCL_v3 Dataset Structure

The BFCL_v3 multi-turn dataset contains:
- **8 tool classes** discovered from test files
- **105 total functions** extracted (including those without parameter requirements)
- **63 functions** with parameters (when filtered)

### Category Distribution

Functions by category:
- **Storage** (17 functions) - Gorilla file system operations
- **Science** (17 functions) - Mathematical operations
- **Finance** (16 functions) - Trading bot operations
- **Vehicle Control** (16 functions) - Car control operations
- **Travel Booking** (14 functions) - Travel booking system
- **Posting API** (12 functions) - Social media operations
- **Events** (7 functions) - Ticket management
- **Communication** (6 functions) - Message/contact management

### Sample Output Format

Each tool definition follows the Magnet canonical format:

```json
{
  "category": "Storage",
  "tool_name": "gorilla_file_system",
  "tool_description": "Functions provided by the gorilla file system toolkit.",
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

## Key Transformations

### 1. Format Normalization
- **Input**: BFCL JSONL format (one JSON object per line)
- **Output**: Magnet canonical format with unified schema
- **Change**: Normalized `"type": "object"` to `"type": "dict"`

### 2. Category Mapping
- Mapped BFCL class names to semantic categories
- Example: `gorilla_file_system` → `Storage`, `trading_bot` → `Finance`

### 3. Parameter Separation
- Split parameters into `required` and `optional` lists
- Preserved default values for optional parameters

### 4. Class Name Conversion
- Converted PascalCase test file references to snake_case file names
- Example: `GorillaFileSystem` → `gorilla_file_system`

## Example Transformations

### Example 1: Simple Required Parameter

**Input (BFCL):**
```json
{
  "name": "buy_stock",
  "description": "Buy a stock with a given symbol and quantity.",
  "parameters": {
    "type": "dict",
    "properties": {
      "symbol": {"type": "string", "description": "The stock symbol."},
      "quantity": {"type": "integer", "description": "The quantity to buy."}
    },
    "required": ["symbol", "quantity"]
  }
}
```

**Output (Magnet):**
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
      "symbol": {"type": "string", "description": "The stock symbol."},
      "quantity": {"type": "integer", "description": "The quantity to buy."}
    },
    "required": ["symbol", "quantity"],
    "optional": []
  }
}
```

### Example 2: Optional Parameters

**Input (BFCL):**
```json
{
  "name": "find",
  "description": "Find files or directories...",
  "parameters": {
    "type": "dict",
    "properties": {
      "path": {"type": "string", "default": "."},
      "name": {"type": "string", "default": "None"}
    },
    "required": []
  }
}
```

**Output (Magnet):**
```json
{
  "category": "Storage",
  "tool_name": "gorilla_file_system",
  "api_name": "find",
  "parameters": {
    "type": "dict",
    "properties": {
      "path": {"type": "string", "default": "."},
      "name": {"type": "string", "default": "None"}
    },
    "required": [],
    "optional": ["path", "name"]
  }
}
```

## Integration Opportunities for APIGen-MT

### 1. Tool Pool for Dataset Generation
- Use extracted tools as input for multi-turn conversation generation
- Leverage categorization for domain-specific selection
- Apply parameter schemas for validation

### 2. Enhanced Tool Discovery
- Use category information to select relevant tools
- Filter by parameter complexity
- Search descriptions for semantic matching

### 3. Schema Standardization
- Adopt Magnet canonical format as standard
- Ensure consistency across tool sources
- Enable cross-dataset compatibility

## Usage Examples

### Basic Extraction
```python
from parse_bfcl import parse_bfcl_func_doc

definitions = parse_bfcl_func_doc(
    func_doc_dir="path/to/multi_turn_func_doc",
    require_parameters=True
)
```

### Filter by Category
```python
storage_tools = [d for d in definitions if d.category == "Storage"]
```

### Export to JSONL
```python
import json

with open("tools.jsonl", "w") as f:
    for defn in definitions:
        f.write(json.dumps(defn.to_dict()) + "\n")
```

## Running the Scripts

### Sample Extraction (Recommended)
```bash
cd magnet_tool_extraction
python sample_extraction.py
```

### Usage Examples
```bash
python usage_examples.py
```

### Full Collection
```bash
python collect_tools.py \
  --bfcl-func-doc /path/to/BFCL_v3/multi_turn_func_doc \
  --bfcl-data-dir /path/to/BFCL_v3 \
  --output tool_pool.jsonl \
  --stats
```

## Output Files Generated

- **`exported_tools.jsonl`** - All tool definitions in Magnet format (105 entries)
- **`tool_pool.jsonl`** - Generated by collect_tools.py (if run)

## Next Steps

1. **Integrate with APIGen-MT pipeline**: Use the extracted tool definitions as input for multi-turn conversation generation
2. **Extend to other sources**: Add parsers for additional tool datasets
3. **Implement name rewriting**: Add LLM-based name rewriting to avoid contamination (as mentioned in the Magnet paper)
4. **Add response schemas**: Optionally include the `response` field from BFCL for complete specifications

## References

- **Magnet Paper**: [arXiv 2503.07826](https://arxiv.org/abs/2503.07826)
- **BFCL Dataset**: [HuggingFace](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
- **Original Implementation**: `/home/ishalyminov/data/magnet_mt/`

## Contact

For questions about the extraction process or integration with APIGen-MT, refer to:
- `ANALYSIS.md` for detailed technical documentation
- `README.md` for quick start guide
- `sample_extraction.py` for working examples