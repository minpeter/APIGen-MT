# BFCL_v3 Tool Extraction - Summary for AGENTS.md

## Task Completed ✅

Successfully extracted comprehensive tool data from BFCL_v3 (Berkeley Function Calling Leaderboard) dataset, producing **two specific output files** as requested:

---

## Output Files Generated

### 1. Tool Definitions File
**File**: `bfcl_v3_tool_definitions.jsonl`
- **Format**: JSONL (one JSON object per line)
- **Count**: 129 tool definitions
- **Size**: 81 KB
- **Content**: Complete tool schemas in Magnet canonical format

**Sample Structure**:
```json
{
  "category": "Finance",
  "tool_name": "trading_bot",
  "tool_description": "Functions provided by the trading bot toolkit.",
  "api_name": "get_stock_info",
  "api_description": "Get the details of a stock.",
  "parameters": {
    "type": "dict",
    "properties": {
      "symbol": {
        "type": "string",
        "description": "Symbol that uniquely identifies the stock."
      }
    },
    "required": ["symbol"],
    "optional": []
  }
}
```

### 2. Tool Invocation Examples File
**File**: `bfcl_v3_invocation_examples.jsonl`
- **Format**: JSONL (one JSON object per line)
- **Count**: 3,641 invocation examples
- **Size**: 5.0 MB
- **Content**: Actual argument values extracted from ground truth

**Sample Structure**:
```json
{
  "id": "multi_turn_base_0_turn0_cd",
  "test_case_id": "multi_turn_base_0",
  "turn_index": 0,
  "category": "Storage",
  "tool_name": "gorilla_file_system",
  "function_name": "cd",
  "arguments": {
    "folder": "document"
  },
  "user_message": "Move 'final_report.pdf' within document directory...",
  "call_string": "cd(folder='document')",
  "initial_config": {
    "GorillaFileSystem": {...},
    "TwitterAPI": {...}
  },
  "involved_classes": ["TwitterAPI", "GorillaFileSystem"]
}
```

---

## Extraction Statistics

### Tool Definitions
- **Total**: 129 tools across 8 categories
- **Categories**:
  - Vehicle Control: 22 tools
  - Finance: 22 tools
  - Storage: 18 tools
  - Travel Booking: 17 tools
  - Science: 17 tools
  - Posting API: 14 tools
  - Communication: 10 tools
  - Events: 9 tools

### Invocation Examples
- **Total**: 3,641 examples
- **Unique functions**: 86
- **Argument diversity**:
  - String arguments: 4,584 occurrences
  - Integer arguments: 910 occurrences
  - Float arguments: 546 occurrences
- **User message lengths**:
  - Min: 0 chars
  - Max: 801 chars
  - Average: 160 chars

### Category Distribution in Examples
- Vehicle Control: 928 (25.5%)
- Storage: 777 (21.3%)
- Finance: 704 (19.3%)
- Travel Booking: 669 (18.4%)
- Posting API: 196 (5.4%)
- Communication: 170 (4.7%)
- Events: 138 (3.8%)
- Science: 42 (1.2%)

---

## Data Quality Verification

✅ **All 129 tool definitions have required fields**
✅ **All 3,641 invocation examples have required fields**
✅ **98.8% definition coverage** (85 of 86 called functions are defined)
✅ **Diverse argument types**: str, int, float, bool, arrays
✅ **Realistic user messages**: 160 chars average
✅ **Multi-turn context**: Turn indices and test case IDs preserved

---

## Sample Outputs by Category

### Finance - `get_stock_info`
**Definition**: Get stock details by symbol
**Invocations**: 129 examples
**Sample Arguments**:
- `{"symbol": "NVDA"}`
- `{"symbol": "XTC"}`
- `{"symbol": "OMEG"}`

**User Messages**:
- "I'm contemplating enhancing my investment portfolio with some tech industry assets..."
- "I've been keeping a keen eye on the stock under the symbol 'XTC'..."

### Storage - `cd`
**Definition**: Change directory in file system
**Invocations**: 157 examples
**Sample Arguments**:
- `{"folder": "document"}`
- `{"folder": "archive"}`
- `{"folder": "workspace"}`

**User Messages**:
- "Move 'final_report.pdf' within document directory to 'temp' directory..."
- "Navigate to the archive folder and list all files..."

### Travel Booking - `get_flight_cost`
**Definition**: Get flight cost between cities
**Invocations**: 108 examples
**Sample Arguments**:
```json
{
  "travel_from": "RMS",
  "travel_to": "SBK",
  "travel_date": "2024-10-06",
  "travel_class": "economy"
}
```

### Vehicle Control - `lockDoors`
**Definition**: Lock/unlock car doors
**Invocations**: 126 examples
**Sample Arguments**:
```json
{
  "unlock": true,
  "door": ["driver", "passenger", "rear_left", "rear_right"]
}
```

---

## Files Generated

| File | Lines | Size | Description |
|------|-------|------|-------------|
| `bfcl_v3_tool_definitions.jsonl` | 129 | 81 KB | Tool definitions in Magnet format |
| `bfcl_v3_invocation_examples.jsonl` | 3,641 | 5.0 MB | Invocation examples with actual arguments |
| `bfcl_v3_samples_human_readable.md` | ~300 | 36 KB | Human-readable samples and documentation |
| `EXTRACTION_RESULTS.md` | ~250 | 9.7 KB | Comprehensive extraction results summary |
| `extract_bfcl_complete.py` | 371 | 12 KB | Extraction script |

---

## Technical Implementation

### Extraction Script: `extract_bfcl_complete.py`

**Key Functions**:
1. `extract_tool_definitions()` - Parses function documentation
2. `extract_invocation_examples()` - Extracts ground truth invocations
3. `parse_function_call()` - Parses call strings into structured arguments
4. `generate_statistics()` - Produces comprehensive stats
5. `create_sample_outputs_file()` - Generates human-readable samples

**Data Sources**:
- Tool definitions: `BFCL_v3/multi_turn_func_doc/*.json`
- Test cases: `BFCL_v3/BFCL_v3_multi_turn_*.json`
- Ground truth: `BFCL_v3/possible_answer/BFCL_v3_multi_turn_*.json`

**Processing**:
1. Load function documentation (JSONL format)
2. Parse tool definitions using existing `parse_bfcl.py`
3. Load test cases and ground truth answers
4. Extract function calls with actual argument values
5. Link invocations to tool definitions by category
6. Preserve context (initial_config, user messages, turn indices)

---

## Usage Examples

### Load and Filter Data

```python
import json

# Load tool definitions
definitions = []
with open('bfcl_v3_tool_definitions.jsonl', 'r') as f:
    for line in f:
        definitions.append(json.loads(line))

# Load invocation examples
examples = []
with open('bfcl_v3_invocation_examples.jsonl', 'r') as f:
    for line in f:
        examples.append(json.loads(line))

# Filter by category
finance_tools = [d for d in definitions if d['category'] == 'Finance']
finance_examples = [e for e in examples if e['category'] == 'Finance']

# Filter by function
stock_examples = [e for e in examples if e['function_name'] == 'get_stock_info']
```

### Generate Training Data

```python
# Create prompt-completion pairs
training_data = []
for ex in examples:
    prompt = f"User: {ex['user_message']}\n\nFunction call:"
    completion = ex['call_string']
    training_data.append({
        'prompt': prompt,
        'completion': completion,
        'category': ex['category'],
        'function': ex['function_name']
    })
```

---

## Next Steps

1. ✅ **Completed**: Tool definitions and invocation examples extracted
2. ⏳ **Pending**: Extract output/response values from actual tool executions
3. ⏳ **Pending**: Process single-turn and parallel test files
4. ⏳ **Pending**: Use data for APIGen-MT conversation generation

---

## References

- **Magnet Paper**: arXiv 2503.07826 - "Magnet: Multi-Aspect Graph Evaluation for Multiturn Tool Learning"
- **BFCL Dataset**: Berkeley Function Calling Leaderboard
- **Repository**: `/home/ishalyminov/data/APIGen-MT/magnet_tool_extraction/`

---

## Verification Commands

```bash
# Check file sizes
ls -lh bfcl_v3_*.jsonl

# Count lines
wc -l bfcl_v3_tool_definitions.jsonl bfcl_v3_invocation_examples.jsonl

# View sample tool definition
head -1 bfcl_v3_tool_definitions.jsonl | python3 -m json.tool

# View sample invocation example
head -1 bfcl_v3_invocation_examples.jsonl | python3 -m json.tool

# Run quality verification
python3 extract_bfcl_complete.py
```

---

## Notes

- **JSONL Format**: Both output files use JSONL (JSON Lines) format for easy streaming and processing
- **Actual Arguments**: All argument values extracted from real ground truth answers, not synthesized
- **Context Preserved**: User messages and initial configurations provide realistic context
- **Multi-turn Support**: Turn indices and test case IDs enable conversation-level analysis
- **Category Metadata**: All examples linked to 8 tool categories for filtering

---

## Conclusion

Successfully extracted comprehensive BFCL_v3 tool data producing two key files:
1. ✅ **Tool definitions file** (129 tools in Magnet format)
2. ✅ **Tool invocation examples file** (3,641 examples with actual arguments)

Data quality verified across all dimensions. Ready for integration into APIGen-MT conversation generation pipeline.