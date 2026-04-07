# BFCL_v3 Complete Extraction Results

## Executive Summary

Successfully extracted comprehensive tool data from the BFCL_v3 (Berkeley Function Calling Leaderboard) dataset:

- **129 tool definitions** in Magnet canonical format
- **3,641 invocation examples** with actual arguments and user messages
- **8 categories** of tools from multi-turn conversations

---

## Output Files

### 1. `bfcl_v3_tool_definitions.jsonl`

**Format**: JSONL (one JSON object per line)

**Content**: Tool definitions in Magnet canonical format

**Structure**:
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

**Statistics**:
- Total tools: 129
- Categories: 8
- Tools by category:
  - Vehicle Control: 22 tools
  - Finance: 22 tools
  - Storage: 18 tools
  - Travel Booking: 17 tools
  - Science: 17 tools
  - Posting API: 14 tools
  - Communication: 10 tools
  - Events: 9 tools

---

### 2. `bfcl_v3_invocation_examples.jsonl`

**Format**: JSONL (one JSON object per line)

**Content**: Tool invocation examples with actual arguments, user messages, and context

**Structure**:
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

**Statistics**:
- Total examples: 3,641
- Unique functions: 86
- Average args per call: 1.66
- Min args: 0
- Max args: 7
- Calls with 0 args: 508 (14%)
- Calls with 1 arg: 1,613 (44%)
- Calls with 2+ args: 1,520 (42%)

**Top 10 Most Called Functions**:
1. cd - 157 calls
2. startEngine - 132 calls
3. get_stock_info - 129 calls
4. lockDoors - 126 calls
5. book_flight - 123 calls
6. get_zipcode_based_on_city - 108 calls
7. get_flight_cost - 108 calls
8. post_tweet - 103 calls
9. get_order_details - 97 calls
10. fillFuelTank - 96 calls

---

### 3. `bfcl_v3_samples_human_readable.md`

**Format**: Markdown

**Content**: Human-readable presentation of tool definitions and invocation examples

**Sections**:
- Part 1: Tool Definitions (organized by category)
- Part 2: Invocation Examples (diverse sample of 20 different functions)

---

## Sample Data by Category

### Communication (170 examples)

**Example**: `message_login`
```json
{
  "function_name": "message_login",
  "arguments": {"user_id": "USR001"},
  "user_message": "Logging in as USR001...",
  "call_string": "message_login(user_id='USR001')"
}
```

### Events (138 examples)

**Example**: `resolve_ticket`
```json
{
  "function_name": "resolve_ticket",
  "arguments": {
    "ticket_id": 7423,
    "resolution": ""
  },
  "user_message": "There's a minor snag in our ticketing system...",
  "call_string": "resolve_ticket(ticket_id=7423,resolution='')"
}
```

### Finance (704 examples)

**Example**: `get_stock_info`
```json
{
  "function_name": "get_stock_info",
  "arguments": {"symbol": "NVDA"},
  "user_message": "I'm contemplating enhancing my investment portfolio...",
  "call_string": "get_stock_info(symbol='NVDA')"
}
```

### Storage (777 examples)

**Example**: `cd`
```json
{
  "function_name": "cd",
  "arguments": {"folder": "document"},
  "user_message": "Move 'final_report.pdf' within document directory...",
  "call_string": "cd(folder='document')"
}
```

### Travel Booking (669 examples)

**Example**: `get_flight_cost`
```json
{
  "function_name": "get_flight_cost",
  "arguments": {
    "travel_from": "RMS",
    "travel_to": "SBK",
    "travel_date": "2024-10-06",
    "travel_class": "economy"
  },
  "user_message": "Wanderlust is calling, and I'm mapping out my travel...",
  "call_string": "get_flight_cost(travel_from='RMS', travel_to='SBK', travel_date='2024-10-06', travel_class='economy')"
}
```

### Vehicle Control (928 examples)

**Example**: `lockDoors`
```json
{
  "function_name": "lockDoors",
  "arguments": {
    "unlock": true,
    "door": ["driver", "passenger", "rear_left", "rear_right"]
  },
  "user_message": "Hey there, I noticed that all of my car doors seem to have locked themselves up...",
  "call_string": "lockDoors(unlock=True, door=['driver', 'passenger', 'rear_left', 'rear_right'])"
}
```

---

## Key Features

### Tool Definitions Include:
✅ Complete parameter schemas with types
✅ Required vs optional parameter separation
✅ Default values for optional parameters
✅ Detailed descriptions for each parameter
✅ Category and tool name metadata

### Invocation Examples Include:
✅ Actual argument values from real test cases
✅ User messages providing context
✅ Original call strings in Python syntax
✅ Initial system configuration/state
✅ Test case IDs for traceability
✅ Turn index for multi-turn context

---

## Data Quality

### Coverage
- **Multi-turn conversations**: 3 test files processed
  - `BFCL_v3_multi_turn_base.json`
  - `BFCL_v3_multi_turn_composite.json`
  - `BFCL_v3_multi_turn_long_context.json`

### Diversity
- **86 unique functions** called across all examples
- **8 categories** represented
- **Complex argument types**: strings, integers, floats, booleans, arrays
- **Parameter ranges**: 0 to 7 arguments per call

### Authenticity
- All examples extracted from actual BFCL_v3 test cases
- Ground truth answers used for accuracy
- User messages provide realistic context
- Initial configurations show system state

---

## Usage Examples

### Load Tool Definitions
```python
import json

definitions = []
with open('bfcl_v3_tool_definitions.jsonl', 'r') as f:
    for line in f:
        definitions.append(json.loads(line))

# Filter by category
storage_tools = [d for d in definitions if d['category'] == 'Storage']
```

### Load Invocation Examples
```python
examples = []
with open('bfcl_v3_invocation_examples.jsonl', 'r') as f:
    for line in f:
        examples.append(json.loads(line))

# Group by function
from collections import defaultdict
by_function = defaultdict(list)
for ex in examples:
    by_function[ex['function_name']].append(ex)

# Get examples for specific function
cd_examples = by_function['cd']
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
        'metadata': {
            'category': ex['category'],
            'function': ex['function_name'],
            'arguments': ex['arguments']
        }
    })
```

---

## Integration with APIGen-MT

### Recommended Workflow

1. **Use tool definitions** to create a tool pool for generation
2. **Use invocation examples** as few-shot examples for prompting
3. **Use initial_config** to set up realistic scenarios
4. **Use user messages** as templates for generation

### Example Integration

```python
# Step 1: Build tool pool
tool_pool = {}
for defn in definitions:
    tool_name = defn['tool_name']
    if tool_name not in tool_pool:
        tool_pool[tool_name] = []
    tool_pool[tool_name].append(defn)

# Step 2: Select tools for generation
target_category = 'Storage'
storage_tools = [d for d in definitions if d['category'] == target_category]

# Step 3: Get few-shot examples
few_shot_examples = [ex for ex in examples if ex['category'] == target_category][:5]

# Step 4: Generate new conversations
for tool in storage_tools[:3]:
    # Use tool definition and few-shot examples to generate
    new_conversations = generate_conversations(
        tool=tool,
        few_shot_examples=few_shot_examples,
        num_conversations=10
    )
```

---

## Technical Notes

### Extraction Process
1. Parsed all function documentation from `multi_turn_func_doc/`
2. Loaded test cases from multi-turn JSON files
3. Parsed ground truth answers for actual function calls
4. Extracted argument values from call strings
5. Linked examples to tool definitions by category
6. Preserved initial configurations for context

### Data Sources
- **Tool definitions**: `BFCL_v3/multi_turn_func_doc/*.json`
- **Test cases**: `BFCL_v3/BFCL_v3_multi_turn_*.json`
- **Ground truth**: `BFCL_v3/possible_answer/BFCL_v3_multi_turn_*.json`

### Known Limitations
- Only multi-turn test files processed (not simple/multiple/parallel)
- Function call parsing uses regex (may miss edge cases)
- Output values not included (only input arguments)
- Response schemas not extracted

---

## Next Steps

1. ✅ Tool definitions and invocation examples extracted
2. ⏳ Extract output/response values from test executions
3. ⏳ Process single-turn and parallel test files
4. ⏳ Create conversation-level datasets
5. ⏳ Generate new multi-turn conversations using this data

---

## Files Generated

| File | Lines | Size | Description |
|------|-------|------|-------------|
| `bfcl_v3_tool_definitions.jsonl` | 129 | ~80KB | Tool definitions in Magnet format |
| `bfcl_v3_invocation_examples.jsonl` | 3,641 | ~2MB | Invocation examples with arguments |
| `bfcl_v3_samples_human_readable.md` | ~300 | ~25KB | Human-readable samples |

---

## References

- **Magnet Paper**: arXiv 2503.07826
- **BFCL Dataset**: gorilla-llm/Berkeley-Function-Calling-Leaderboard
- **Extraction Scripts**: `extract_bfcl_complete.py`