# Datapoint Generation Algorithm

## Overview

This document describes the algorithm for generating training datapoints from tool categories in the APIGen-MT pipeline. The process involves selecting diverse tool subsets, generating blueprints through LLM prompting, validating executability, and quality-checking through iterative refinement.

---

## High-Level Workflow

```
1. Load Tool Pool (categorized)
   ↓
2. Select Random Tool Subset (uniform category coverage)
   ↓
3. Generate Query Template
   ↓
4. Generate Blueprint Candidate (LLM Prompt 1)
   ↓
5. Validate Format & Executability
   ↓
6. Quality Review (LLM Prompt 2)
   ↓
7. [If needed] Refine Blueprint (iterate steps 4-6)
   ↓
8. Simulate Execution Trace
   ↓
9. Save Datapoint
```

---

## Detailed Algorithm Steps

### Step 1: Load Tool Pool by Category

**Location:** `generate_datapoints_bfcl.py::load_tool_categories()`

**Purpose:** Load tools from BFCL JSONL file and group them by category.

**Process:**
1. Read JSONL file line by line
2. Extract category from each tool definition
3. For tools with 'Unknown' category, infer from source field:
   - `live` → "Live APIs"
   - `exec` → "Executable"
   - `simple` → "Simple Functions"
   - `multiple` → "Multiple Functions"
   - `parallel` → "Parallel Functions"
4. Group tools into dictionary: `{category: [tool1, tool2, ...]}`

**Example Categories:**
- Calendar
- Communication
- File System
- Travel
- Stock/Trading
- Vehicle Control
- etc.

---

### Step 2: Select Random Tool Subset

**Location:** `generate_datapoints_bfcl.py::select_tool_subset()`

**Purpose:** Select a diverse subset of tools with uniform category coverage.

**Algorithm:**
```python
tools_per_category = max(1, max_tools // len(categories))
for each category:
    randomly_select(tools_per_category tools from category)
```

**Parameters:**
- `max_tools`: Maximum tools in subset (default: 80)
- Ensures each category has equal representation

**Output:**
- List of 60-80 tools across all categories
- Written to temporary tool pool JSONL file

---

### Step 3: Generate Query Template

**Location:** `generate_datapoints_bfcl.py::main()` (lines 297-303)

**Purpose:** Create a user query that guides blueprint generation.

**Template:**
```
Using tools from {focus_category}, perform a {num_actions}-step multi-step operation 
that requires retrieving information and then creating or updating a record. 
(variation #{N})
```

**Parameters:**
- `focus_category`: Randomly selected from available categories
- `num_actions`: Specified by `--num-actions` argument (default: 2)
- `N`: Variation number to ensure uniqueness

**Example Query:**
```
"Using tools from Calendar, perform a 3-step multi-step operation that requires 
retrieving information and then creating or updating a record. (variation #42)"
```

---

### Step 4: Generate Blueprint Candidate

**Location:** `apigen-phase1.py::_generate_blueprint_candidate()`

**Purpose:** Use LLM to generate a structured blueprint with tool calls.

#### LLM Prompt Structure

**System Prompt:**
```
You are a 'Task Blueprint Generator' for Phase 1 of the APIGen-MT pipeline. 
Your goal is to create a detailed blueprint representing a realistic and verifiable 
multi-turn interaction scenario between a user and an AI agent.

Based on the given tool descriptions, you must generate a JSON object containing:

1. `q` (string): The user's initial question/request. It should be specific and natural, 
   preferably representing a scenario that requires multiple steps of interaction.

2. `a_gt_steps` (list): A Ground Truth tool call list that the agent must call to 
   completely and in the correct order resolve the user's request `q`. 
   Each element must be in the format:
   `{"tool_name": "tool_name", "arguments": {"arg_name": "value", ...}}`

3. `o_gt` (string): A natural description of the final summary or response message 
   that the agent should provide to the user.

Response Rules:
- Your final response must contain a valid JSON object inside a ```json ... ``` code block
- Use placeholders like {{tool_name.output.field_name}} to reference outputs from previous steps

[Previous LLM Reviews - if any]
[Previous Validation Errors - if any]

**Tool Descriptions:**
Available Tools and their Schemas:
{
  "name": "get_calendar_events",
  "parameters": {
    "properties": {
      "start_date": {"type": "string"},
      "end_date": {"type": "string"}
    },
    "required": ["start_date", "end_date"]
  }
}
...
```

**User Prompt:**
```
User Query (q): Using tools from Calendar, perform a 3-step multi-step operation...

[Below is feedback on a previous attempt... - if regenerating]

Using the guidelines and available tools above, generate the Blueprint JSON for this request.
```

#### Output Structure

**Blueprint JSON:**
```json
{
  "q": "Find my calendar events for next week and create a summary meeting.",
  "a_gt_steps": [
    {
      "tool_calls": [
        {
          "tool_name": "get_calendar_events",
          "arguments": {
            "start_date": "2025-01-20",
            "end_date": "2025-01-27"
          }
        }
      ]
    },
    {
      "tool_calls": [
        {
          "tool_name": "create_calendar_event",
          "arguments": {
            "summary": "Weekly Review",
            "start_time": "{{get_calendar_events.output.start_time}}",
            "end_time": "{{get_calendar_events.output.end_time}}"
          }
        }
      ]
    }
  ],
  "o_gt": "Successfully created a summary meeting based on next week's calendar events."
}
```

**Key Features:**
- Placeholder syntax: `{{tool_name.output.field}}` for inter-step dependencies
- Multiple tool calls per step (parallel execution)
- Sequential steps for dependent operations

---

### Step 5: Validate Format & Executability

**Location:** `apigen-phase1.py::_validate_blueprint_format_and_executability()`

**Purpose:** Verify that the blueprint can be executed without errors.

#### Validation Checks

**5.1 Format Validation:**
- Pydantic model validation
- Check `a_gt_steps` is not empty
- Each step contains at least one tool call
- All required fields present

**5.2 Executability Validation:**

For each tool call in sequence:

```python
for each step in blueprint.a_gt_steps:
    for each tool_call in step.tool_calls:
        # 1. Check tool exists
        if not tool_manager.tool_exists(tool_call.tool_name):
            return error
        
        # 2. Process placeholders
        processed_args = _process_placeholders(
            tool_call.arguments, 
            execution_context  # Results from previous steps
        )
        
        # 3. Validate required arguments
        for req_arg in tool_schema["parameters"]["required"]:
            if req_arg not in processed_args:
                return error
        
        # 4. Simulate tool execution
        simulated_output = tool_manager.invoke_tool(
            tool_call.tool_name,
            processed_args
        )
        
        # 5. Store output for subsequent steps
        execution_context[f"{tool_call.tool_name}.output"] = simulated_output
```

**Validation Result:**
```json
{
  "is_valid_format": true,
  "format_errors": null,
  "is_executable": true,
  "executability_checks": [
    {
      "step_index": 0,
      "tool_name": "get_calendar_events",
      "can_execute": true,
      "reason": "Successfully simulated invocation.",
      "simulated_output": {...}
    }
  ],
  "overall_validation_passed": true
}
```

---

### Step 6: Quality Review (LLM)

**Location:** `apigen-phase1.py::_get_llm_review_and_feedback()`

**Purpose:** Use LLM to assess blueprint quality and provide feedback.

#### LLM Quality Review Prompt

**System Prompt:**
```
You are an expert in data quality control for AI agent development. 
Please carefully review the provided task blueprint, the LLM's reasoning 
during generation, and the automatic validation results.

Your goal is to evaluate whether this blueprint is suitable for generating 
high-quality training data, and to provide specific, actionable feedback.

Note: The use of placeholders ({{tool_name.output.field_name}}) is essential 
for multi-step and dependency implementations. If placeholders are used 
correctly, do not penalize for this.
```

**User Prompt:**
```
Below is the task blueprint and related information to be reviewed:

1. **User's Initial Request (q)**:
   [query text]

2. **Generated Task Blueprint (a_gt_steps, o_gt)**:
   [blueprint JSON]

3. **LLM's Reasoning During Blueprint Generation**:
   [reasoning text]

4. **Automatic Validation Results**:
   [validation JSON]

**Review Items and Evaluation Criteria:**
* Clarity and Realism of Request (q)
* Logical Coherence and Accuracy (a_gt_steps) - order, parallelism, placeholder usage
* Appropriateness of Tool Usage - selected tools, argument values
* Appropriateness of Outcome (o_gt)
* Implications of Automatic Validation Results
* Overall Quality: (Excellent, Good, Fair, Poor)

**Output Format:**
{
  "quality_assessment": "Excellent|Good|Fair|Poor",
  "feedback_summary": "...",
  "suggested_corrections": "..."
}
```

#### Quality Assessment Output

```json
{
  "quality_assessment": "Good",
  "feedback_summary": "The blueprint is logically coherent and uses appropriate tools. The placeholder syntax is correct.",
  "suggested_corrections": "Consider adding error handling for the case when no events are found."
}
```

**Quality Levels:**
- **Excellent**: Ready to use, no issues
- **Good**: Minor improvements possible but acceptable
- **Fair**: Needs refinement before use
- **Poor**: Significant issues, regenerate

---

### Step 7: Iterative Refinement Loop

**Location:** `apigen-phase1.py::generate_verified_blueprint()`

**Purpose:** Regenerate blueprint if quality is insufficient.

**Algorithm:**
```python
max_attempts = 3

for attempt in range(1, max_attempts + 1):
    # Generate blueprint (includes previous feedback)
    blueprint = generate_blueprint_candidate(
        q=query,
        previous_feedback=last_feedback,
        previous_llm_reviews=review_history,
        previous_validation_result=validation_result
    )
    
    # Validate
    validation_result = validate_blueprint(blueprint)
    
    if not validation_result.overall_validation_passed:
        last_feedback = format_errors(validation_result)
        continue
    
    # Quality review
    llm_review = get_llm_review(blueprint, validation_result)
    
    if llm_review.quality in ["Excellent", "Good"]:
        return VerifiedBlueprint(
            blueprint=blueprint,
            validation_result=validation_result,
            llm_review_history=review_history
        )
    else:
        last_feedback = llm_review.feedback_summary + llm_review.suggested_corrections
        continue

return None  # Failed after max_attempts
```

**Feedback Integration:**

When regenerating, the system includes:
1. Previous LLM quality assessments
2. Previous validation errors
3. Specific correction suggestions

This creates a self-improving loop where each iteration addresses identified issues.

---

### Step 8: Simulate Execution Trace

**Location:** `tool_simulation.py::simulate_execution_trace()`

**Purpose:** Generate simulated tool execution outputs for training.

**Process:**
```python
execution_trace = []

for each tool_call in blueprint.a_gt_steps:
    simulated_return = simulate_tool_return(
        function_name=tool_call.tool_name,
        arguments=tool_call.arguments
    )
    
    execution_step = {
        "step_index": idx,
        "function_name": tool_call.tool_name,
        "arguments": tool_call.arguments,
        "simulated_return": simulated_return,
        "timestamp": generate_timestamp()
    }
    
    execution_trace.append(execution_step)
```

**Simulated Return Examples:**

For `get_calendar_events`:
```json
{
  "status": "success",
  "result": {
    "type": "list",
    "events": [
      {"title": "Team Meeting", "start": "2025-01-20T10:00:00Z"},
      {"title": "Project Review", "start": "2025-01-21T14:00:00Z"}
    ],
    "count": 2
  },
  "simulated": true
}
```

For `create_calendar_event`:
```json
{
  "status": "success",
  "result": {
    "type": "dict",
    "event_id": "EVT-12345",
    "message": "Event created successfully",
    "created": true
  },
  "simulated": true
}
```

---

### Step 9: Save Final Datapoint

**Location:** `generate_datapoints_bfcl.py::main()` (lines 344-365)

**Purpose:** Save verified datapoint to output file.

**Datapoint Structure:**
```json
{
  "query": "Using tools from Calendar, perform a 3-step multi-step operation...",
  "blueprint": {
    "q": "Find my calendar events for next week...",
    "a_gt_steps": [...],
    "o_gt": "Successfully created a summary meeting..."
  },
  "simulated_execution_trace": [
    {
      "step_index": 0,
      "function_name": "get_calendar_events",
      "arguments": {...},
      "simulated_return": {...},
      "timestamp": "2025-01-15T10:50:00Z"
    }
  ],
  "validation_result": {
    "is_valid_format": true,
    "is_executable": true,
    "overall_validation_passed": true
  },
  "llm_review_history": [
    {
      "quality_assessment": "Good",
      "feedback_summary": "The blueprint is logically coherent..."
    }
  ],
  "generation_attempts": 2,
  "tools_used": ["get_calendar_events", "create_calendar_event"],
  "categories_used": ["Calendar"],
  "timestamp": "2025-01-15T10:55:32Z"
}
```

---

## Key Design Principles

### 1. Placeholder System

Placeholders enable multi-step dependencies:
```
{{tool_name.output.field_name}}
```

**Examples:**
- `{{get_calendar_events.output.events.0.id}}`
- `{{search_users.output.user_id}}`
- `{{create_file.output.file_path}}`

**Processing:** Resolved during validation using execution context from previous steps.

---

### 2. Tool Pool Diversity

**Uniform Category Coverage:**
- Ensures all categories are represented equally
- Prevents bias toward common tool types
- Creates varied and realistic scenarios

**Benefits:**
- Training data covers diverse domains
- Models learn cross-category reasoning
- Reduces overfitting to specific tool patterns

---

### 3. Iterative Quality Improvement

**Multi-stage validation:**
1. Automatic format checking
2. Executability simulation
3. LLM quality assessment
4. Regeneration with feedback

**Benefits:**
- Catches errors early
- Ensures blueprint correctness
- Improves data quality iteratively

---

### 4. Realistic Execution Simulation

**Tool-specific simulators:**
- File system operations (`ls`, `cat`, `mkdir`)
- Math operations (`add`, `multiply`, `sqrt`)
- API operations (`get_stock_info`, `post_tweet`)
- Vehicle control (`start_engine`, `lock_doors`)

**Benefits:**
- Provides realistic training targets
- Tests blueprint executability
- Creates complete training examples

---

## Configuration Parameters

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-datapoints` | 100 | Number of datapoints to generate |
| `--num-actions` | 2 | Number of tool calls per blueprint |
| `--debug` | False | Enable detailed LLM logging |

### Generation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_tools` | 80 | Maximum tools in subset |
| `max_attempts` | 3 | Maximum regeneration attempts |
| `validation_attempts` | 2 | Attempts before giving up |

### Quality Thresholds

| Level | Action |
|-------|--------|
| Excellent | Accept immediately |
| Good | Accept immediately |
| Fair | Regenerate with feedback |
| Poor | Regenerate with feedback |

---

## Output Files

### Datapoints File
- **Path:** `data/bfcl_datapoints/bfcl_multiturn_datapoints_{timestamp}.jsonl`
- **Format:** JSONL (one JSON object per line)
- **Encoding:** UTF-8

### Temporary Files
- **Temp Tool Pool:** `data/temp_pools/temp_tool_pool.jsonl`
- **Debug Logs:** `logs/blueprint_generation_{timestamp}.log` (when `--debug` enabled)

---

## Example End-to-End Flow

### Input:
```
Tool Pool: BFCL_v3.jsonl (1000+ tools across 10+ categories)
Query: "Using tools from Calendar, perform a 2-step multi-step operation..."
```

### Step-by-Step:

1. **Load Tools:**
   - Calendar: 150 tools
   - Communication: 120 tools
   - Travel: 90 tools
   - ...

2. **Select Subset (80 tools):**
   - Calendar: 8 tools
   - Communication: 8 tools
   - Travel: 8 tools
   - ...

3. **Generate Blueprint (Attempt 1):**
   - LLM generates: `get_calendar_events` → `create_calendar_event`
   - Placeholders: `{{get_calendar_events.output.start_time}}`

4. **Validate:**
   - Format: ✓ Valid
   - Executability: ✓ All tools exist, arguments valid
   - Simulation: ✓ Successfully executed

5. **Quality Review:**
   - Assessment: "Good"
   - Feedback: "Blueprint is logically coherent"

6. **Accept & Save:**
   - Datapoint saved with execution trace
   - Ready for training

---

## Summary

The datapoint generation algorithm combines:

1. **Strategic Tool Selection** - Uniform category coverage ensures diversity
2. **LLM-Guided Blueprint Generation** - Natural language queries translated to structured tool calls
3. **Multi-Level Validation** - Format, executability, and quality checks
4. **Iterative Refinement** - Feedback-driven improvement loop
5. **Execution Simulation** - Realistic training targets

This pipeline produces high-quality, diverse training data for multi-turn agent systems, ensuring that generated blueprints are both realistic and executable.