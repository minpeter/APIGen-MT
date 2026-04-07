# BFCL_v3 Tool Extraction Pipeline

## Overview

This document describes the complete extraction pipeline from BFCL_v3 raw data to the two output files.

```
┌─────────────────────────────────────────────────────────────────┐
│                    BFCL_v3 Raw Data                             │
└─────────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
┌──────────────────────┐    ┌──────────────────────┐
│  multi_turn_func_doc │    │  Test Cases +        │
│  (Tool Definitions)  │    │  Ground Truth        │
└──────────────────────┘    └──────────────────────┘
                │                       │
                │                       │
                ▼                       ▼
┌──────────────────────┐    ┌──────────────────────┐
│  parse_bfcl.py       │    │  Parse Function      │
│  - Parse JSONL       │    │  Call Strings        │
│  - Build schemas     │    │  - Extract args      │
│  - Resolve cats      │    │  - Link to defs      │
└──────────────────────┘    └──────────────────────┘
                │                       │
                └───────────┬───────────┘
                            │
                            ▼
            ┌──────────────────────────────┐
            │  extract_bfcl_complete.py    │
            │  - Coordinate extraction     │
            │  - Generate statistics       │
            │  - Create human-readable     │
            └──────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
┌──────────────────────┐    ┌──────────────────────┐
│  OUTPUT FILE 1:      │    │  OUTPUT FILE 2:      │
│  Tool Definitions    │    │  Invocation Examples │
│  (129 tools)         │    │  (3,641 examples)    │
└──────────────────────┘    └──────────────────────┘
```

## Data Flow

### Step 1: Tool Definition Extraction

**Input**: `BFCL_v3/multi_turn_func_doc/*.json` (JSONL files)

**Processing**:
1. Read each JSONL line containing function documentation
2. Parse JSON structure to extract:
   - Category
   - Tool name
   - API name
   - API description
   - Parameter schema (properties, required, optional)
3. Resolve category names (e.g., "GorillaFileSystem" → "Storage")
4. Convert to Magnet canonical format

**Output**: `bfcl_v3_tool_definitions.jsonl`

**Sample**:
```json
{
  "category": "Finance",
  "tool_name": "trading_bot",
  "api_name": "get_stock_info",
  "api_description": "Get the details of a stock.",
  "parameters": {
    "type": "dict",
    "properties": {
      "symbol": {"type": "string", "description": "..."}
    },
    "required": ["symbol"],
    "optional": []
  }
}
```

### Step 2: Invocation Example Extraction

**Input**: 
- `BFCL_v3/BFCL_v3_multi_turn_*.json` (test cases)
- `BFCL_v3/possible_answer/BFCL_v3_multi_turn_*.json` (ground truth)

**Processing**:
1. Load test cases with user messages and initial configs
2. Load ground truth answers with function calls
3. Parse function call strings (e.g., `get_stock_info(symbol='NVDA')`)
4. Extract argument values from call strings
5. Link to tool definitions by category
6. Preserve context:
   - Test case ID
   - Turn index
   - User message
   - Initial configuration

**Output**: `bfcl_v3_invocation_examples.jsonl`

**Sample**:
```json
{
  "id": "multi_turn_base_0_turn0_cd",
  "test_case_id": "multi_turn_base_0",
  "turn_index": 0,
  "category": "Storage",
  "function_name": "cd",
  "arguments": {"folder": "document"},
  "user_message": "Move 'final_report.pdf'...",
  "call_string": "cd(folder='document')",
  "initial_config": {...},
  "involved_classes": ["GorillaFileSystem"]
}
```

## Quality Assurance

### Automated Checks

✅ **Field Completeness**
- All 129 definitions have required fields
- All 3,641 examples have required fields

✅ **Schema Validity**
- 105/129 definitions have parameter properties
- 99/129 definitions have required parameters
- 15/129 definitions have optional parameters

✅ **Cross-Reference Integrity**
- 98.8% of called functions have definitions
- 85 of 86 unique functions are defined

✅ **Data Diversity**
- Argument types: str (4,584), int (910), float (546)
- User message lengths: 0-801 chars (avg 160)
- Category distribution: 8 categories represented

### Manual Verification

✅ **Sample Inspection**
- Tool definitions match Magnet format
- Invocation examples have realistic arguments
- User messages provide appropriate context
- Categories correctly assigned

## Statistics Summary

| Metric | Value |
|--------|-------|
| **Tool Definitions** | 129 |
| **Invocation Examples** | 3,641 |
| **Unique Functions** | 86 |
| **Categories** | 8 |
| **Argument Diversity** | 3 types (str, int, float) |
| **Avg Args per Call** | 1.66 |
| **Definition Coverage** | 98.8% |

## File Sizes

| File | Size | Lines |
|------|------|-------|
| Tool Definitions | 81 KB | 129 |
| Invocation Examples | 5.0 MB | 3,641 |
| Human-Readable Samples | 36 KB | ~300 |
| Documentation | 20 KB | ~500 |

## Usage

### Load Data

```python
import json

# Load tool definitions
with open('bfcl_v3_tool_definitions.jsonl', 'r') as f:
    definitions = [json.loads(line) for line in f]

# Load invocation examples
with open('bfcl_v3_invocation_examples.jsonl', 'r') as f:
    examples = [json.loads(line) for line in f]
```

### Filter by Category

```python
finance_tools = [d for d in definitions if d['category'] == 'Finance']
finance_examples = [e for e in examples if e['category'] == 'Finance']
```

### Generate Training Data

```python
training_pairs = [
    {
        'prompt': f"User: {ex['user_message']}\n\nFunction:",
        'completion': ex['call_string'],
        'metadata': {
            'category': ex['category'],
            'function': ex['function_name']
        }
    }
    for ex in examples
]
```

## Integration with APIGen-MT

### Tool Pool Construction

```python
# Build tool pool from definitions
tool_pool = {}
for defn in definitions:
    tool_name = defn['tool_name']
    if tool_name not in tool_pool:
        tool_pool[tool_name] = []
    tool_pool[tool_name].append(defn)
```

### Few-Shot Example Selection

```python
# Select diverse few-shot examples
few_shots = []
seen_functions = set()
for ex in examples:
    if ex['function_name'] not in seen_functions:
        few_shots.append(ex)
        seen_functions.add(ex['function_name'])
    if len(few_shots) >= 10:
        break
```

## Next Steps

1. ✅ Extract tool definitions and invocation examples
2. ⏳ Extract tool outputs from actual executions
3. ⏳ Process single-turn and parallel test files
4. ⏳ Generate new multi-turn conversations
5. ⏳ Evaluate using Magnet framework

---

**Generated**: 2026-04-05
**Script**: `extract_bfcl_complete.py`
**Data Source**: BFCL_v3 (Berkeley Function Calling Leaderboard)
**Format**: Magnet Canonical Format
