# Magnet Tool Pool Extraction - Complete Package

## Quick Navigation

| Document | Description | Use Case |
|----------|-------------|----------|
| **[SUMMARY.md](SUMMARY.md)** | Executive summary of what was done | Overview and key findings |
| **[ANALYSIS.md](ANALYSIS.md)** | Detailed technical analysis with sample outputs | Deep dive into extraction process |
| **[README.md](README.md)** | Quick start guide and reference | Getting started and usage |
| **[This File](INDEX.md)** | Navigation and complete file listing | Finding the right document |

## Quick Start

### 1. View Sample Extraction
```bash
python sample_extraction.py
```

### 2. Run Usage Examples
```bash
python usage_examples.py
```

### 3. View Extracted Data
```bash
head -1 exported_tools.jsonl | python -m json.tool
```

## Complete File Listing

### Documentation (4 files)
- `SUMMARY.md` - Executive summary with key findings and integration opportunities
- `ANALYSIS.md` - Comprehensive technical analysis with detailed sample outputs
- `README.md` - Quick start guide, usage examples, and reference
- `INDEX.md` - This navigation file

### Core Scripts (4 files)
- `tool_definition.py` - Data model for Magnet canonical format
- `parse_bfcl.py` - BFCL-v3 parser (enhanced to handle JSONL)
- `parse_stabletoolbench.py` - StableToolBench/ToolEnv2404 parser
- `collect_tools.py` - Main orchestration script

### Demonstration Scripts (2 files)
- `sample_extraction.py` - Comprehensive demonstration with statistics
- `usage_examples.py` - Multiple programmatic usage examples

### Generated Data (1 file)
- `exported_tools.jsonl` - All 105 extracted tool definitions in Magnet format

## Extraction Results Summary

```
Total tools extracted: 105
Categories: 8

Breakdown by category:
  Storage                        17 tools
  Science                        17 tools
  Finance                        16 tools
  Vehicle Control                16 tools
  Travel Booking                 14 tools
  Posting Api                    12 tools
  Events                          7 tools
  Communication                   6 tools
```

## Sample Tool Definition

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

## Key Insights

### From BFCL_v3 Dataset
1. **105 tool definitions** extracted across 8 categories
2. **Average 1.41 required parameters** per tool
3. **15 tools with optional parameters** (14%)
4. **41 tools with 2+ required parameters** (39%)

### Category Highlights
- **Storage**: File system operations (cat, ls, cd, find, grep, etc.)
- **Science**: Mathematical operations (add, subtract, multiply, etc.)
- **Finance**: Trading operations (buy_stock, sell_stock, etc.)
- **Vehicle Control**: Car control (climate, brakes, lights, etc.)

### Transformation Features
1. ✅ Format normalization (JSONL to Magnet format)
2. ✅ Category mapping (class names to semantic categories)
3. ✅ Parameter separation (required vs optional)
4. ✅ Quality filtering (parameterless tools excluded by default)

## Integration with APIGen-MT

### Recommended Approach
1. Use `exported_tools.jsonl` as initial tool pool
2. Apply category-based filtering for domain-specific tasks
3. Validate generated conversations against parameter schemas
4. Extend with additional tool sources as needed

### Example Integration
```python
from magnet_tool_extraction.parse_bfcl import parse_bfcl_func_doc

# Load tool definitions
tools = parse_bfcl_func_doc("path/to/BFCL_v3/multi_turn_func_doc")

# Filter for specific domain
storage_tools = [t for t in tools if t.category == "Storage"]

# Use in your pipeline
for tool in storage_tools:
    # Generate multi-turn conversations using this tool
    conversations = generate_conversations(tool)
```

## Technical Enhancements Made

### Fixed Issues
1. ✅ JSONL format handling (original parser expected JSON arrays)
2. ✅ Class name conversion (PascalCase to snake_case)
3. ✅ Enhanced error handling and logging

### Added Features
1. ✅ Comprehensive documentation with sample outputs
2. ✅ Multiple usage examples
3. ✅ Statistical analysis scripts
4. ✅ Ready-to-use exported data

## Next Steps

1. **Review** the extracted tools in `exported_tools.jsonl`
2. **Run** the sample extraction to see live results: `python sample_extraction.py`
3. **Read** the detailed analysis in `ANALYSIS.md`
4. **Integrate** with your APIGen-MT pipeline
5. **Extend** with additional tool sources

## References

- **Magnet Paper**: [arXiv 2503.07826](https://arxiv.org/abs/2503.07826)
- **BFCL Dataset**: [HuggingFace](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
- **StableToolBench**: [HuggingFace](https://huggingface.co/datasets/stabletoolbench/ToolEnv2404)

## Questions?

- For **overview**: Read `SUMMARY.md`
- For **technical details**: Read `ANALYSIS.md`
- For **usage**: Read `README.md`
- For **examples**: Run `sample_extraction.py` or `usage_examples.py`

---

**Total Package**: 11 files (4 docs + 4 scripts + 2 demos + 1 data file)