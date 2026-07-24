# Agents Documentation

## Current LLM Configuration (source of truth)

The primary generation CLI uses the OpenAI-compatible settings in
`src/runtime_config.py`:

- Provider: OpenRouter by default; any compatible endpoint may be selected.
- API base: `OPENAI_API_BASE`, defaulting to `https://openrouter.ai/api/v1`.
- API key: required in `OPENAI_API_KEY`; keep it in the environment or a secure vault.
- Default model: `minimax/minimax-m2.7`.

Use `uv run python main.py --help` to inspect the CLI without credentials or a
network call. Historical extraction notes below are not a second configuration
source; their runnable defaults use the same shared constants where applicable.

## BFCL Tool Extraction - Completed (2026-04-10)

Successfully extracted and enhanced tool definitions from BFCL_v3 dataset with LLM-predicted outputs.

### Changes Implemented
1. ✅ Removed `-v` command line argument from `extract_bfcl_with_outputs.py`
2. ✅ Hardcoded to BFCL_v3 dataset only (no version choice)
3. ✅ The extraction workflow uses the shared OpenAI-compatible runtime settings
4. ✅ Added 5-second timeout after each LLM invocation (in `llm_output_predictor.py`)
5. ✅ Implemented 3-attempt retry logic for LLM errors
6. ✅ Added `--frequent` argument for processing top 100 most frequent tools

### Extraction Results
- **Total tools processed**: 66 (out of 89 in frequent tools list)
- **Success rate**: 98.5% (65/66 tools, 1 failed after 3 retries)
- **Output file**: `magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl` (60KB)
- **Processing time**: ~27 minutes

### Output Statistics
- All 66 tools have both `output_type` and `output_description` fields
- Output types: dict (30), float (14), string (8), list (8), other (6)
- Categories: Storage (10), Science (10), Posting API (9), Travel Booking (9), Vehicle Control (8), Events (7), Finance (7), Communication (6)

### Key Files
- `extract_bfcl_with_outputs.py` - Main extraction script with all improvements
- `llm_output_predictor.py` - LLM output prediction with 5-second timeout
- `bfcl_v3_tools_with_outputs.jsonl` - Extracted tools with predicted outputs
- `EXTRACTION_FREQUENT_TOOLS_SUMMARY.md` - Detailed summary report

## Step-by-Step Blueprint Generation - Completed (2026-04-11)

### Overview
Implemented a new step-by-step blueprint generation approach where:
1. User query is generated first based on available tools
2. Each step selects a tool based on current dialog trajectory
3. Tool execution is simulated immediately
4. Output is validated against declared type/description
5. Result is appended to dialog trajectory
6. Final response is generated after all steps

### Key Differences from Original Approach

**Original (Single-Shot)**:
- Entire blueprint generated in one LLM call
- No real simulation during generation
- Output validation happens after generation
- No dialog trajectory context

**New (Step-by-Step)**:
- Query generated first, then tools selected step-by-step
- Tool execution simulated immediately after selection
- Output validation at each step
- Full dialog trajectory context for tool selection
- Placeholders resolved using actual simulated outputs

### Implementation

**New Files**:
- `src/apigen_step_by_step.py` - Core step-by-step generator
- `src/generate_step_by_step.py` - CLI script for batch generation

**Key Models**:
- `ToolCallWithOutput` - Single tool call with simulated output
- `TrajectoryStep` - Step with tool calls and reasoning
- `ConversationTrajectory` - Complete conversation with query, steps, final response
- `StepByStepDatapoint` - Full datapoint with trajectory and metadata

### Generation Process

```
Step 1: Generate user query
  └─ LLM generates query based on available tools and focus category

Step 2..N+1: For each action step
  ├─ Build context with query + previous steps + outputs
  ├─ LLM selects next tool and arguments
  ├─ Process placeholders from previous outputs
  ├─ Simulate tool execution via ToolManager
  ├─ Validate output against declared type/description
  └─ Append to trajectory

Step N+2: Generate final response
  └─ LLM summarizes what was accomplished
```

### Output Format

```json
{
  "trajectory": {
    "query": "User's request",
    "steps": [
      {
        "step_number": 1,
        "tool_calls": [
          {
            "tool_name": "get_user_id",
            "arguments": {"user": "alice"},
            "output": {"user_id": "U12345"}
          }
        ],
        "reasoning": "Need to get user ID first..."
      },
      {
        "step_number": 2,
        "tool_calls": [
          {
            "tool_name": "send_message",
            "arguments": {"receiver_id": "U12345", "message": "Hello!"},
            "output": {"success": true, "message_id": "M789"}
          }
        ],
        "reasoning": "Send message using the ID from step 1..."
      }
    ],
    "final_response": "I've sent your message to Alice.",
    "tools_used": ["get_user_id", "send_message"],
    "categories_used": []
  },
  "generation_metadata": {
    "num_actions": 2,
    "focus_category": "Communication",
    "query_intent": "User wants to send a message to Alice...",
    "expected_tools": ["get_user_id", "send_message"]
  }
}
```

### Generation Results

Successfully generated 100 datapoints using step-by-step approach:

**Statistics**:
- Total generated: 100/100
- Output file: `data/generated/step_by_step_100_datapoints.jsonl`
- Processing time: ~4 hours (with API retries)

**Top Tools Used**:
- get_ticket: 20
- edit_ticket: 17
- get_user_id: 16
- send_message: 16
- echo: 13
- displayCarStatus: 8
- ls: 8
- get_flight_cost: 7
- resolve_ticket: 7
- get_user_tickets: 7

### Benefits of Step-by-Step Approach

1. **Realistic Tool Selection**: Tools selected based on actual context
2. **Proper Dependencies**: Placeholders resolved using real simulated outputs
3. **Verifiable Execution**: Each step's output validated immediately
4. **Complete Trajectory**: Full conversation history for training
5. **Error Handling**: Failures at individual steps don't ruin entire generation

### Usage

```bash
# Generate 100 datapoints with 2 actions each
python src/generate_step_by_step.py --num-datapoints 100 --num-actions 2

# Custom output path
python src/generate_step_by_step.py --num-datapoints 50 --output custom_output.jsonl
```
