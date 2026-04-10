# Tool Execution in Datapoint Generation

## Overview

The `generate_100_datapoints_bfcl.py` script has been updated to **execute all tools** defined in each blueprint and capture their outputs in the generated datapoints.

## What Changed

### Before
- Only generated the blueprint (sequence of tool calls)
- Did not execute the tools
- No tool outputs in the datapoints

### After
- Generates the blueprint
- **Executes each tool call** using `ToolManager.invoke_tool()`
- **Captures tool outputs** in `tool_execution_results` field
- Stores complete execution trace in each datapoint

## Implementation Details

### 1. Tool Execution Logic

After generating a verified blueprint, the script now:

```python
# Execute all tool calls and collect outputs
tool_execution_results = []

for step_idx, step in enumerate(verified_bp.blueprint.a_gt_steps):
    step_results = {
        "step_index": step_idx,
        "tool_calls": []
    }
    
    for tool_call in step.tool_calls:
        # Execute the tool
        tool_output = tool_manager.invoke_tool(
            tool_call.tool_name,
            tool_call.arguments
        )
        
        # Store the output
        step_results["tool_calls"].append({
            "tool_name": tool_call.tool_name,
            "arguments": tool_call.arguments,
            "output": tool_output  # <-- Captured output
        })
    
    tool_execution_results.append(step_results)
```

### 2. Virtual Tool Execution

The `ToolManager.invoke_tool()` method uses a **virtual executor** for BFCL tools:

- For each tool call, it prompts the LLM with:
  - Tool name
  - Tool description
  - Tool schema (parameters)
  - Arguments provided
  - Current date/time

- The LLM generates a **realistic simulated output**:
  - For data-fetching tools: Returns realistic data
  - For creation tools: Returns success status
  - For error-prone tools: Sometimes simulates errors

### 3. Datapoint Structure

Each generated datapoint now includes:

```json
{
  "query": "User query text",
  "blueprint": {
    "q": "Refined query",
    "a_gt_steps": [
      {
        "tool_calls": [
          {"tool_name": "get_weather", "arguments": {"city": "NYC"}},
          {"tool_name": "create_alert", "arguments": {"message": "Storm warning"}}
        ]
      }
    ],
    "o_gt": "Expected outcome description"
  },
  "validation_result": {...},
  "llm_review_history": [...],
  "tools_used": ["get_weather", "create_alert"],
  "categories_in_pool": ["Weather", "Alerts", ...],
  "focus_category": "Weather",
  "tool_execution_results": [
    {
      "step_index": 0,
      "tool_calls": [
        {
          "tool_name": "get_weather",
          "arguments": {"city": "NYC"},
          "output": {
            "temperature": 72,
            "condition": "Partly Cloudy",
            "humidity": 65,
            "wind_speed": 12
          }
        },
        {
          "tool_name": "create_alert",
          "arguments": {"message": "Storm warning"},
          "output": {
            "status": "success",
            "alert_id": "alert_12345",
            "created_at": "2026-04-06T17:30:00Z"
          }
        }
      ]
    }
  ]
}
```

## Benefits

### 1. Complete Training Data
- Tool outputs can be used for **training multi-turn dialog models**
- Models learn how tool outputs look and how to process them
- Enables training on **sequential reasoning** (output → input transformation)

### 2. Validation
- Ensures tool calls are **executable** (not just plausible)
- Catches errors in tool definitions early
- Verifies argument compatibility

### 3. Context for Downstream Tasks
- Tool outputs can be used for:
  - Response generation
  - Follow-up queries
  - Multi-step reasoning
  - Error handling scenarios

## Execution Flow

```
User Query
    ↓
Generate Blueprint (LLM)
    ↓
Validate Blueprint
    ↓
For Each Tool Call:
    ├─ Invoke Tool
    ├─ LLM Simulates Output
    └─ Capture Result
    ↓
Store in Datapoint
    ↓
Save to Output File
```

## Example: Multi-Step Tool Execution

### Query
"Retrieve my calendar events for today and create a summary document"

### Blueprint
1. `fetch_calendar_events(start_date="2026-04-06", end_date="2026-04-06")`
2. `create_document(title="Daily Summary", content="{{step_1_output}}")`

### Execution
**Step 1:** Execute `fetch_calendar_events`
- Output: `{"events": [{"time": "10:00", "title": "Team Meeting"}, ...]}`

**Step 2:** Execute `create_document`
- Output: `{"document_id": "doc_789", "status": "created"}`

### Captured in Datapoint
```json
"tool_execution_results": [
  {
    "step_index": 0,
    "tool_calls": [
      {
        "tool_name": "fetch_calendar_events",
        "arguments": {"start_date": "2026-04-06", "end_date": "2026-04-06"},
        "output": {"events": [...]}
      }
    ]
  },
  {
    "step_index": 1,
    "tool_calls": [
      {
        "tool_name": "create_document",
        "arguments": {"title": "Daily Summary", "content": "..."},
        "output": {"document_id": "doc_789", "status": "created"}
      }
    ]
  }
]
```

## Testing

Run the test script to verify tool execution:

```bash
cd /home/ishalyminov/data/APIGen-MT/src
python3 test_tool_execution.py
```

This will:
1. Load 80 tools from the BFCL pool
2. Generate a blueprint
3. Execute all tool calls
4. Save a sample datapoint to `/tmp/test_datapoint_with_execution.json`

## Performance Considerations

- Each tool execution requires an LLM API call (~1-3 seconds)
- For 2-step blueprints: ~2-6 additional seconds per datapoint
- Total generation time: ~30-90 seconds per datapoint (including validation)

## Output Files

- **Main output**: `data/generated/apigen_phase1_100_datapoints_bfcl.jsonl`
- **Temp tool pools**: `data/generated/temp_tool_pool.jsonl`
- **Test output**: `/tmp/test_datapoint_with_execution.json`

## Date
Updated: 2026-04-06