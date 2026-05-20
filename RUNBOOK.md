# APIGen-MT Data Synthesis Pipeline Runbook

This document describes the three-stage data synthesis pipeline: extracting tool schemas from BFCL, generating Python implementations with unit tests, and producing step-by-step training datapoints using those implementations.

## Prerequisites

### Environment

Create a `.env` file in the project root with:

```
OPENAI_API_KEY=<your NVIDIA API key>
OPENAI_API_BASE=https://integrate.api.nvidia.com/v1
```

The pipeline uses NVIDIA's API endpoint with OpenAI-compatible clients. All three stages require `OPENAI_API_KEY`; stages 1, 2 and 3 use `z-ai/glm-5.1` by default.

### Dependencies

```bash
pip install openai pydantic python-dotenv requests pytest tiktoken
```

---

## Pipeline Overview

```
Stage 1: BFCL Tool Extraction          Stage 2: Tool Implementation          Stage 3: Step-by-Step Generation
┌──────────────────────────┐      ┌──────────────────────────────┐      ┌─────────────────────────────────┐
│ BFCL_v3 dataset          │      │ bfcl_v3_tools_with_outputs   │      │ bfcl_v3_tools_with_outputs      │
│  ├─ multi_turn_func_doc/ │ ──── │ bfcl_v3_invocation_examples  │ ──── │ bfcl_v3_invocation_examples     │
│  └─ multi_turn test files│      │ BFCL func_doc response schemas│      │ tools/*.py (Python impls)       │
│                          │      │                              │      │                                 │
│ Output:                  │      │ Output:                      │      │ Output:                         │
│ bfcl_v3_tools_with_      │      │ tools/{class}.py             │      │ step_by_step_datapoints.jsonl   │
│ outputs.jsonl            │      │ tools/schemas.py             │      │                                 │
│                          │      │ tests/tools/test_{class}.py  │      │                                 │
└──────────────────────────┘      └──────────────────────────────┘      └─────────────────────────────────┘
```

---

## Stage 1: Extracting Tools with Outputs from BFCL

### Purpose

Parse the BFCL_v3 dataset to extract structured tool definitions (name, description, parameters, category) and use an LLM to predict each tool's `output_type` and `output_description` -- fields that BFCL does not provide natively.

### Script

```bash
python magnet_tool_extraction/extract_bfcl_with_outputs.py [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `../magnet_mt/data` | Base directory where BFCL_v3 is stored/downloaded |
| `--output` | auto (`bfcl_v3_tools_with_outputs.jsonl`) | Output JSONL path |
| `--frequent` | off | Filter to top 89 most frequent tools (from `top_100_frequent_tools.txt`) |
| `--limit N` | all | Process only N tools (for testing) |
| `--max-contexts N` | 5 | Max invocation contexts per tool sent to the LLM |
| `--debug` | off | Print full LLM prompts and responses |
| `--force-download` | off | Force re-download of BFCL data |

### Workflow

```
1. Download/verify BFCL_v3 data
   └─ Checks for {data_dir}/BFCL_v3/multi_turn_func_doc/
   └─ Downloads from HuggingFace if missing

2. Extract tool definitions
   └─ Parse multi_turn_func_doc/*.json → ToolDefinition list
   └─ Map filenames to categories:
       gorilla_file_system → Storage
       math_api → Science
       trading_bot → Finance
       ticket_api → Events
       message_api → Communication
       posting_api → Posting Api
       travel_booking → Travel Booking
       vehicle_control → Vehicle Control

3. Extract invocation contexts
   └─ Scan BFCL_v3_multi_turn_*.json test files
   └─ Collect (user_message, assistant_message, tool_calls) per tool

4. Optional: --frequent filter
   └─ Read top_100_frequent_tools.txt (89 entries in "tool_name.api_name" format)
   └─ Keep only matching tools

5. LLM output prediction (per tool)
   └─ Build prompt with tool schema + invocation contexts
   └─ Call nvidia/nemotron-3-super-120b-a12b (temp=0.7, max_tokens=500)
   └─ Parse JSON response for output_type + output_description
   └─ 5-second sleep between calls (rate limiting)
   └─ 3-attempt retry on failure; fallback to "unknown"

6. Write results to JSONL
```

### Prompts

**System prompt** (`llm_output_predictor.py:159-183`):

```
You are an expert at analyzing function/tool schemas and predicting what they return.

Given information about a function/tool:
1. Its name and description
2. Its parameters (arguments it accepts)
3. Example invocations showing how it's used in practice

Your task is to predict:
1. output_type: The type of data the function returns (e.g., "string", "integer",
   "boolean", "dict", "list", "file content", "API response", "operation status", etc.)
2. output_description: A clear description of what the function returns,
   including the structure if it's a complex type

Guidelines:
- Be specific about the output type (e.g., "weather data dict" instead of just "dict")
- Include important fields if returning a structured type
- Mention if the function returns success/failure status
- Consider what makes sense given the function's purpose and parameters
- Use the invocation contexts to understand real-world usage patterns

Respond ONLY with a valid JSON object matching the schema:
{
  "output_type": "string",
  "output_description": "string"
}
```

**User prompt** (`llm_output_predictor.py:185-254`):

```markdown
# Function Information

**Tool Name**: {tool_name}
**API Name**: {api_name}
**Description**: {api_description}

## Parameters
```json
{parameters_schema}
```

## Example Invocations

### Context 1
**User**: {user_message}
**Assistant**: {assistant_message}
**Tool Calls**:
- {tool_name}: {arguments_json}

...

## Task

Based on the function information and example invocations above, predict:
1. The output type this function returns
2. A clear description of what the output contains

Respond with a JSON object containing 'output_type' and 'output_description'.
```

### Output

File: `magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl` (105 lines, one JSON object per line)

```json
{
  "category": "Storage",
  "tool_name": "gorilla_file_system",
  "tool_description": "Functions provided by the gorilla file system toolkit.",
  "api_name": "cat",
  "api_description": "Display the contents of a file...",
  "parameters": {
    "type": "dict",
    "properties": { "file_name": {"type": "string", "description": "..."} },
    "required": ["file_name"],
    "optional": []
  },
  "output_type": "string",
  "output_description": "The contents of the specified file from the current directory, returned as a string."
}
```

The two LLM-added fields are `output_type` and `output_description`. Typical output type distribution: dict (50), float (19), string (18), list (12), boolean (2), number (2), integer (1), unknown (1).

### Estimated Runtime

~30 minutes for all 105 tools (dominated by the 5-second sleep between each LLM call).

---

## Stage 2: Generating Tool Implementations with Unit Tests

### Purpose

Use an LLM to generate executable Python class implementations for the 8 BFCL tool classes, Pydantic input schemas, and pytest unit tests. These implementations replace LLM-based virtual tool execution with deterministic real code during datapoint generation.

### Script

```bash
python scripts/generate_tool_implementations.py [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--classes` | all 8 | Comma-separated class keys to generate |
| `--output-dir` | `tools/` | Output directory for class modules |
| `--test-dir` | `tests/tools/` | Output directory for test files |
| `--model` | `z-ai/glm-5.1` | LLM model |
| `--api-base` | `$OPENAI_API_BASE` | API endpoint URL |
| `--api-key` | `$OPENAI_API_KEY` | API key |
| `--skip-existing` | off | Skip classes with existing output files |
| `--verbose` | off | Print LLM reasoning and file details |
| `--max-retries` | 2 | Max retries per generation step |
| `--skip-tests` | off | Skip unit test generation |
| `--only-tests` | off | Generate tests only for existing class files |
| `--only-schemas` | off | Generate/update schemas.py only |

### Input Data

Three input sources are combined:

| Source | Path | Purpose |
|---|---|---|
| Tool definitions | `magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl` | Per-API schemas, parameters, output types |
| Response schemas | `{data_dir}/BFCL_v3/multi_turn_func_doc/{class}.json` | Exact return dict shapes (`response` field) |
| Invocation examples | `magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl` | Real call examples with arguments and initial configs |

### Data Preparation

Before prompting, the script:

1. **Groups tools by class** -- `group_tools_by_class()` groups the 105 tool definitions by `tool_name` field (e.g., all `math_api` tools together)
2. **Groups invocation examples by function** -- `group_examples_by_function()` for per-method call examples
3. **Extracts canonical initial configs** -- `get_canonical_initial_configs()` picks the largest (by JSON size) `initial_config` per class from invocation examples
4. **Selects diverse examples** -- `select_diverse_examples()` picks up to N examples prioritizing different test cases and argument patterns (3 for class generation, 2 for tests)

### 8 Tool Classes

| Class Key | Class Name | Config Key | Category | Example Methods |
|---|---|---|---|---|
| `gorilla_file_system` | `GorillaFileSystem` | `GorillaFileSystem` | Storage | cat, cd, cp, ls, mkdir, grep, find, diff |
| `math_api` | `MathAPI` | `MathAPI` | Science | add, divide, mean, min_value, logarithm |
| `message_api` | `MessageAPI` | `MessageAPI` | Communication | send_message, get_user_id, list_users |
| `posting_api` | `PostingAPI` | `TwitterAPI` | Posting Api | post_tweet, follow_user, get_tweet |
| `ticket_api` | `TicketAPI` | `TicketAPI` | Events | create_ticket, edit_ticket, resolve_ticket |
| `trading_bot` | `TradingBot` | `TradingBot` | Finance | place_order, get_stock_info, add_to_watchlist |
| `travel_booking` | `TravelBooking` | `TravelAPI` | Travel Booking | book_flight, get_flight_cost, authenticate_travel |
| `vehicle_control` | `VehicleControl` | `VehicleControlAPI` | Vehicle Control | displayCarStatus, startEngine, set_cruise_control |

Note: `posting_api` uses config key `TwitterAPI` (from BFCL data) but class name `PostingAPI`; `travel_booking` uses `TravelAPI` config key but `TravelBooking` class name.

### Generation Workflow (per class)

```
┌─ Step 1: Generate class code ──────────────────────────────────┐
│  build_class_prompt() → call_llm() → extract_code_block()      │
│  validate_python_code() (compile check)                         │
│  Check all methods present (def {api_name})                     │
│  Retry up to max_retries with error/missing-method feedback     │
│  → tools/{class_key}.py                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓ 1s delay
┌─ Step 2: Generate Pydantic schemas ────────────────────────────┐
│  build_schemas_prompt() → call_llm() → extract_code_block()    │
│  validate_python_code() (with pydantic import prepended)        │
│  Retry on syntax errors                                         │
│  → stored in memory (combined into schemas.py at end)           │
└─────────────────────────────────────────────────────────────────┘
                              ↓ 1s delay
┌─ Step 3: Generate unit tests ──────────────────────────────────┐
│  build_tests_prompt() → call_llm(max_tokens=8192)              │
│  Includes class code + schemas code as reference context        │
│  validate_python_code()                                         │
│  Retry on syntax errors                                         │
│  → tests/tools/test_{class_key}.py                             │
└─────────────────────────────────────────────────────────────────┘
```

After all 8 classes: write combined `tools/schemas.py` (with deduplicated imports and section headers) and `tools/__init__.py` (with `TOOL_CLASSES` registry and `create_tool_instance()` factory).

### Prompts

All three generation steps share a **system prompt** (`generate_tool_implementations.py:333-347`):

```
You are a Python code generator producing production-quality, working Python code.

Rules:
- Return ONLY valid Python code in a single markdown code block (```python ... ```)
- Use type hints on all function signatures
- Methods return dicts matching the specified response schema EXACTLY
- Stateful methods must mutate self state appropriately
- Handle edge cases: missing args, invalid values, not-found scenarios
- Never raise exceptions from methods - return error info in the response dict
- Follow the exact parameter names from the tool definitions (keep camelCase as-is)
- Import only: json, math, re, copy, datetime, typing (List, Dict, Any, Optional, Tuple)
- Do NOT include any explanatory text outside the code block
- Each method must have a docstring
- The class __init__ must accept initial_config: dict and set up internal state
- For classes with multiple initial_config variants, normalize to a canonical form in __init__
```

#### Class Generation Prompt

Structure of `build_class_prompt()`:

```markdown
## Task: Generate the {ClassName} class
File: tools/{class_key}.py

### Class State (initial_config)
```json
{canonical_config_json}
```

### Tools to implement ({N} methods)

#### Method: `{api_name}`
Description: {api_description}
Parameters:
 - {name}: {type} (required|optional, default=X) - {description}
Response schema (return dict must match this exactly):
```json
{response_schema_from_func_doc}
```
Invocation examples (3 shown):
 Example 1:
   call_string: {call_string}
   arguments: {arguments_json}
   state at call time: {initial_config_subset}
   user intent: {user_message}

### Special instructions:
{per-class guidance, e.g.:}
- gorilla_file_system: Track current_dir as list, root as nested dict, cd/ls/mkdir/touch behaviors
- math_api: Stateless, all methods are pure computation
- ticket_api: Normalize config variants (ticket_list→self.ticket_queue, etc.)
- vehicle_control: Flat key-value state, camelCase method names, displayCarStatus returns subsets

### Output format:
Return the complete Python class in a single ```python ... ``` code block.
The class must include __init__(self, initial_config: dict) and all listed methods.
```

#### Schemas Generation Prompt

Structure of `build_schemas_prompt()`:

```markdown
## Task: Generate Pydantic input schemas for {ClassName} tools

Generate one Pydantic BaseModel per tool method, named as {MethodName}Input.
For example, if the method is 'get_stock_info', the schema class is 'GetStockInfoInput'.
For camelCase methods like 'startEngine', use 'StartEngineInput'.

Rules:
- Use exact parameter names from the tool definitions (keep camelCase as-is)
- Required params have no default; optional params use their declared default or None
- Use proper Python types: str, int, float, bool, list, dict, Optional[], List[]
- For enum parameters, use Literal[] from typing
- Each model must have a docstring
- Do NOT import anything beyond: from pydantic import BaseModel, Field; from typing import ...

### Tools ({N} schemas needed):

#### {api_name}
Required: {required_list}
 {param_name}: type={type}, required={bool}, default={default}, desc={description}

Return all schema classes in a single ```python ... ``` code block.
```

#### Test Generation Prompt

Structure of `build_tests_prompt()`:

```markdown
## Task: Generate pytest unit tests for {ClassName}
File: tests/tools/test_{class_key}.py

### Requirements:
- Generate 2-3 tests per method ({N} methods)
- Use pytest fixtures for class instance setup
- Test normal operation, edge cases, and error handling
- Import the class from tools.{class_key}
- Import schemas from tools.schemas

### Test structure:
```python
import pytest
import json
from tools.{class_key} import {ClassName}
```

### initial_config for fixtures:
```json
{canonical_config_json}
```

### Example invocations (use as test case inspiration):
 {call_string}
 # user intent: {user_message}

### Generated class code (for reference):
```python
{class_code_or_summary}
```

### Generated schema code (for reference):
```python
{schemas_code}
```

Return the complete test file in a single ```python ... ``` code block.
```

### Validation

- **Syntax**: `compile(code, filename, "exec")` after every generation step
- **Method presence**: Verifies `def {api_name}` exists in generated class code; retries with missing method names appended to prompt
- **Error feedback**: On syntax errors, the error message is appended to the prompt for retry
- **Class code truncation**: If class code exceeds 4000 chars in the test prompt, only `def`/`class`/`@decorator` lines and docstrings are kept
- **Pytest execution**: After all classes, runs `python -m pytest tests/tools/ -v --tb=short -x` (300s timeout)

### Output Files

| File | Description |
|---|---|
| `tools/{class_key}.py` | 8 Python classes with `__init__(self, initial_config: dict)` and methods returning `Dict[str, Any]` |
| `tools/schemas.py` | Combined Pydantic `BaseModel` subclasses named `{MethodName}Input` |
| `tools/__init__.py` | `TOOL_CLASSES` dict + `create_tool_instance()` factory |
| `tests/tools/test_{class_key}.py` | 8 pytest test files with fixtures |
| `tests/tools/test_smoke.py` | Parametrized integration tests across all classes |
| `tests/tools/conftest.py` | Shared test fixtures |

### Example Commands

```bash
# Generate all 8 classes with tests
python scripts/generate_tool_implementations.py

# Generate only math_api and trading_bot
python scripts/generate_tool_implementations.py --classes math_api,trading_bot

# Skip existing, verbose output
python scripts/generate_tool_implementations.py --skip-existing --verbose

# Generate tests only for existing class files
python scripts/generate_tool_implementations.py --only-tests

# Class + schema only (no tests)
python scripts/generate_tool_implementations.py --skip-tests
```

### Estimated Runtime

~10-15 minutes for all 8 classes (3 LLM calls per class × 8 classes, with 1s delays).

---

## Stage 3: Generating Step-by-Step Datapoints

### Purpose

Generate verified training datapoints by simulating realistic multi-turn tool-calling conversations. Each datapoint contains a user query, a sequence of tool invocations with real outputs, and a final natural-language response. Tool calls are executed against actual Python implementations from Stage 2 (falling back to LLM-based virtual simulation when no implementation exists).

### Scripts

```bash
# Batch generation (CLI)
python src/generate_step_by_step.py [OPTIONS]

# Single datapoint (library)
python src/apigen_step_by_step.py
```

### Arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `--num-datapoints` | `-n` | 100 | Target number of verified datapoints |
| `--num-actions` | `-a` | 2 | Tool calls per datapoint |
| `--output` | `-o` | `step_by_step_datapoints.jsonl` | Output JSONL path |
| `--tool-pool` | | `magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl` | Tool pool file |
| `--invocation-examples` | | `magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl` | Invocation examples (for Python impl loading) |
| `--model` | `-m` | `z-ai/glm-5.1` | LLM model |

### Three-Stage Generation Workflow

Each datapoint is produced by a `StepByStepGenerator.generate_datapoint()` call that runs three stages. Failed datapoints (verification failures) are discarded and not counted toward the target.

```
┌─ Stage 1: Generate and Verify Query ───────────────────────────┐
│                                                                 │
│  1. Pick random focus category                                  │
│  2. LLM generates {query, intent, expected_tools}              │
│  3. Validate:                                                   │
│     - expected_tools count == num_actions                       │
│     - all tool names exist in ToolManager                       │
│     - LLM sequence validation (if num_actions <= 5)            │
│  4. On failure: append feedback, retry (up to 5 attempts)      │
│  5. On success: return QueryGenerationResult                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─ Stage 2: Generate Tool Invocations ────────────────────────────┐
│                                                                 │
│  For each tool in expected_tools (sequential):                  │
│  1. LLM generates arguments (with execution context)           │
│  2. Process {{placeholder}} patterns in arguments               │
│  3. Execute tool:                                               │
│     - If Python impl exists → call actual method                │
│     - Else → LLM-based virtual simulation                      │
│  4. Check for error outputs                                     │
│  5. Validate output type against declared output_type           │
│  6. Update execution context: {tool}_{field} = value            │
│  7. Append TrajectoryStep                                       │
│  On per-tool failure: retry with feedback (up to 3 attempts)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─ Stage 3: Finalize ─────────────────────────────────────────────┐
│                                                                 │
│  1. LLM generates final natural-language response               │
│  2. Run full verification:                                      │
│     a. Tool relevance (keyword overlap heuristic)               │
│     b. Invocation order (pattern-based)                         │
│     c. Output type consistency                                  │
│     d. Placeholder resolution                                   │
│  3. If verification fails → discard (return None)               │
│  4. If verification passes → assemble StepByStepDatapoint       │
└─────────────────────────────────────────────────────────────────┘
```

### Tool Execution: Python Implementations vs. Virtual Simulation

The `ToolManager` maintains two execution paths:

**Python implementations** (preferred when available):
- Loaded from `tools/` via `create_tool_instance()` factory
- Instantiated with canonical `initial_config` from invocation examples
- Methods return deterministic, real Python objects
- State is reset before each datapoint via `reset_python_tool_instances()`
- Parameter types are coerced (e.g., string `"42"` → int `42`)

**LLM-based virtual simulation** (fallback):
- Builds a simulation prompt with tool schema, output guidance, few-shot examples
- Calls LLM to generate the return value
- Validates output type and description via a second LLM call
- Up to 2 retries on validation failure

### Placeholder Resolution

When the LLM generates arguments containing `{{key}}` patterns, they are resolved from the execution context:

```python
# After get_user_id returns {"user_id": "USR005"}:
execution_context = {
    "get_user_id_user_id": "USR005",
    "get_user_id_output": {"user_id": "USR005"}
}

# An argument like {"receiver_id": "{{get_user_id_user_id}}"} resolves to:
{"receiver_id": "USR005"}   # type-preserved when entire value is a placeholder
```

### Prompts

#### Query Generation (`apigen_step_by_step.py:291-325`)

```
You are generating a realistic user query for testing a tool-calling system.

Generate a natural, realistic user query that would require using EXACTLY {num_actions} tools to fulfill.

=== REQUIREMENTS ===
1. The query should be specific and actionable
2. It should mention concrete entities (names, IDs, dates, locations, etc.)
3. It should require EXACTLY {num_actions} tool calls to complete - not more, not less
4. The expected_tools list must contain EXACTLY {num_actions} tool names
5. CRITICAL: Use ONLY the exact tool names from the AVAILABLE TOOLS section below
6. CRITICAL: Do NOT invent tool names - only use tools that exist in the list
7. The tools should logically fit together to accomplish the query

=== AVAILABLE TOOLS WITH DESCRIPTIONS ===
{tools_with_descriptions}

{example_queries}

=== FOCUS CATEGORY ===
Primary category: {focus_category} (select tools primarily from this category)

[=== PREVIOUS ATTEMPT FEEDBACK ===
{accumulated_feedback}
=== END FEEDBACK ===]

=== YOUR TASK ===
Generate a query for category: {focus_category} that requires EXACTLY {num_actions} tools
from the AVAILABLE TOOLS list above.

Respond ONLY with valid JSON in this exact format:
{
  "query": "the generated user query - be specific with names, dates, IDs",
  "intent": "brief description of what the user wants to accomplish",
  "expected_tools": ["tool_name_1", "tool_name_2", ...]
}
```

Few-shot examples are included (filtered by proximity to `num_actions`):

| Category | Tools | Example Query |
|---|---|---|
| Travel Booking | 3 | authenticate_travel → get_flight_cost → book_flight |
| Finance | 3 | get_symbol_by_name → place_order → add_to_watchlist |
| Events | 4 | create_ticket → get_ticket → edit_ticket → get_user_tickets |
| Storage | 3 | cd → du → cat |
| Communication | 2 | get_user_id → send_message |

#### Tool Sequence Validation (`apigen_step_by_step.py:189-205`)

Only run when `num_actions <= 5`:

```
You are validating a tool sequence plan for a user query.

User Query: {query}
Intent: {intent}
Planned Tool Sequence: {expected_tools}

Tool Schemas:
{tool_schemas}

Evaluate if the sequence logically fits the query intent.

Respond with JSON:
{
  "is_valid": true/false,
  "issues": ["list of issues if any"]
}
```

#### Tool Arguments Generation (`apigen_step_by_step.py:669-706`)

```
Generate arguments for the tool '{tool_name}' based on the user query and previous steps.

=== USER QUERY ===
{query}

=== PREVIOUS STEPS ===
{trajectory_summary}

=== EXECUTION CONTEXT ===
{execution_context_json_truncated_to_500_chars}

=== TOOL SCHEMA ===
{parameters_schema}

=== EXPECTED OUTPUT ===
Type: {output_type}
Description: {output_description}

[=== PREVIOUS ATTEMPT FEEDBACK ===
{feedback}]

=== YOUR TASK ===
Generate arguments for '{tool_name}' that:
1. Match the schema above
2. Fulfill the user query
3. Use values from Execution Context when available (e.g., user_id from previous step)
4. Are specific and realistic
5. Will produce an output that matches the Expected Output type and description above

Respond with JSON containing only the arguments:
{
  "arg1": "value1",
  "arg2": "value2"
}
```

#### Final Response Generation (`apigen_step_by_step.py:929-936`)

```
Based on the following conversation, generate a natural final response.

User Query: {query}

Actions taken:
{actions_summary_json}

Generate a concise, natural response that summarizes what was accomplished.
```

#### Virtual Tool Simulation Prompt (`tool_manager.py`)

Used when no Python implementation is available:

```
You are an expert function simulator. Simulate the execution of the following function call.

=== FUNCTION DETAILS ===
Function Name: {tool_name}
Function Description: {description}
Function Parameters Schema: {parameters}

{output_guidance}

=== ARGUMENTS PROVIDED ===
{params}
Current Date/Time: {datetime}

{few_shot_examples}

=== YOUR TASK ===
Generate the return value that '{tool_name}' would produce if executed with the given arguments.

{output_format_instructions}
```

Output guidance is built dynamically from `output_type` and `output_description`, with type-specific few-shot examples:

| Type | Few-Shot Example |
|---|---|
| dict | `get_user` → `{"user_id": "U123", "username": "john_doe", ...}` |
| list | `list_files` → `["document.txt", "image.png", ...]` |
| string | `generate_welcome_message` → `"Welcome, John! ..."` |
| integer | `count_lines` → `42` |
| float | `calculate_exchange_rate` → `0.9234` |

### Verification

After generation, four checks run. **All must pass** for the datapoint to be saved:

| Check | Method | Logic |
|---|---|---|
| Tool relevance | `verify_tool_relevance()` | Keyword overlap between tool description and query; `score > 0.1` or name word overlap |
| Invocation order | `verify_invocation_order()` | Flags tools with "create"/"update"/"send" in name appearing as first step without prior context |
| Output consistency | `verify_output_consistency()` | Python `type()` check against declared `output_type`; handles compound types and nested dict values |
| Placeholder resolution | `verify_placeholder_resolution()` | Scans for unresolved `{{...}}` patterns in arguments |

### Output Format

File: `data/generated/step_by_step_datapoints.jsonl` (one JSON object per line)

```json
{
  "trajectory": {
    "query": "Send a message to user john_doe saying 'Meeting at 3pm'",
    "steps": [
      {
        "step_number": 1,
        "tool_calls": [
          {
            "tool_name": "get_user_id",
            "arguments": {"user": "john_doe"},
            "output": {"user_id": "USR005"}
          }
        ],
        "reasoning": "Generated arguments for get_user_id based on query context"
      },
      {
        "step_number": 2,
        "tool_calls": [
          {
            "tool_name": "send_message",
            "arguments": {"receiver_id": "USR005", "message": "Meeting at 3pm"},
            "output": {"success": true, "message_id": "MSG-123"}
          }
        ],
        "reasoning": "Generated arguments for send_message based on query context"
      }
    ],
    "final_response": "I've sent your message 'Meeting at 3pm' to john_doe.",
    "tools_used": ["get_user_id", "send_message"],
    "categories_used": ["Communication"]
  },
  "generation_metadata": {
    "num_actions": 2,
    "focus_category": "Communication",
    "query_intent": "User wants to send a message to another user",
    "expected_tools": ["get_user_id", "send_message"]
  },
  "verification_result": {
    "query": "...",
    "tool_relevance_checks": [
      {"tool_name": "get_user_id", "is_relevant": true, "relevance_score": 0.15, "reasoning": "..."},
      {"tool_name": "send_message", "is_relevant": true, "relevance_score": 0.25, "reasoning": "..."}
    ],
    "order_is_correct": true,
    "order_verification_details": "No order issues detected.",
    "output_validations": [
      {"tool_name": "get_user_id", "step_number": 1, "output_type_matches": true, "issues": []},
      {"tool_name": "send_message", "step_number": 2, "output_type_matches": true, "issues": []}
    ],
    "placeholder_resolution": {
      "all_resolved": true,
      "total_placeholders": 0,
      "resolved_count": 0,
      "details": []
    },
    "overall_verification_passed": true,
    "verification_summary": "Verification PASSED"
  },
  "token_usage": {
    "prompt_tokens": 12345,
    "completion_tokens": 678,
    "total_tokens": 13023,
    "total_llm_calls": 8
  },
  "timestamp": "2026-04-11T14:30:00.000000",
  "generation_attempt": 5
}
```

### Example Commands

```bash
# Generate 100 datapoints with 2 tool calls each
python src/generate_step_by_step.py -n 100 -a 2

# 50 datapoints with 5 tool calls, custom output path
python src/generate_step_by_step.py -n 50 -a 5 -o data/generated/large_steps.jsonl

# Custom model
python src/generate_step_by_step.py -n 10 -a 3 -m nvidia/nemotron-3-super-120b-a12b
```

### Estimated Runtime

Varies significantly with `num_actions` and verification pass rate. For 100 datapoints with 2 actions: ~2-4 hours (each datapoint requires multiple LLM calls for query generation, argument generation per step, and verification; only verified datapoints count).

### Key Design Decisions

- **Only verified datapoints are saved** -- if verification fails, the datapoint is discarded and not counted
- **Datapoints are written immediately** (append mode) -- no batch write, so progress is preserved on interruption
- **Python tool state is reset per datapoint** -- `reset_python_tool_instances()` ensures state isolation
- **Focus category is randomly sampled** -- uniform distribution across all categories
- **Stage 2 iterates expected_tools directly** -- no LLM tool selection; the tool sequence is planned in Stage 1 and followed sequentially

---

## File Inventory

### Stage 1: Extraction

| File | Role |
|---|---|
| `magnet_tool_extraction/extract_bfcl_with_outputs.py` | Main extraction script |
| `magnet_tool_extraction/llm_output_predictor.py` | LLM output prediction with prompts |
| `magnet_tool_extraction/parse_bfcl.py` | BFCL func_doc parser |
| `magnet_tool_extraction/download_bfcl_v4.py` | BFCL data download helper |
| `magnet_tool_extraction/tool_definition.py` | ToolDefinition/ToolParameters dataclasses |
| `magnet_tool_extraction/top_100_frequent_tools.txt` | Frequent tools filter list |
| `magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl` | **Output** (105 tools) |
| `magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl` | Invocation examples (3641 entries) |

### Stage 2: Implementation

| File | Role |
|---|---|
| `scripts/generate_tool_implementations.py` | Master generation script (prompts + orchestration) |
| `tools/{class_key}.py` | **Output** (8 class modules) |
| `tools/schemas.py` | **Output** (combined Pydantic schemas) |
| `tools/__init__.py` | **Output** (class registry + factory) |
| `tests/tools/test_{class_key}.py` | **Output** (8 test files) |
| `tests/tools/test_smoke.py` | **Output** (integration tests) |
| `tests/tools/conftest.py` | **Output** (shared fixtures) |

### Stage 3: Datapoint Generation

| File | Role |
|---|---|
| `src/generate_step_by_step.py` | CLI batch generation script |
| `src/apigen_step_by_step.py` | Core StepByStepGenerator class + models + verification |
| `src/tool_manager.py` | ToolManager (Python impl + virtual simulation) |
| `src/tool_simulation.py` | Legacy LLM-based simulation (fallback) |
| `src/llm_client.py` | LocalOpenAILLMClient with token tracking |
| `src/prompts.py` | Extracted prompt templates (refactored, not yet integrated) |
| `src/function_schema.py` | JSON schema generation from Python signatures |
| `data/generated/step_by_step_*.jsonl` | **Output** (generated datapoints) |

---

## End-to-End Quick Start

```bash
# 0. Set up environment
echo 'OPENAI_API_KEY=nvapi-...' > .env
echo 'OPENAI_API_BASE=https://integrate.api.nvidia.com/v1' >> .env

# 1. Extract tools from BFCL (produces bfcl_v3_tools_with_outputs.jsonl)
python magnet_tool_extraction/extract_bfcl_with_outputs.py

# 2. Generate Python implementations + schemas + tests (produces tools/*.py)
python scripts/generate_tool_implementations.py

# 3. Run the generated tests to verify implementations
python -m pytest tests/tools/ -v --tb=short -x

# 4. Generate training datapoints (produces step_by_step_datapoints.jsonl)
python src/generate_step_by_step.py -n 100 -a 2 -o data/generated/my_dataset.jsonl
```
