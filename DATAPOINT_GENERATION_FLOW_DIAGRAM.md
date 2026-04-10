# Datapoint Generation Flow Diagram

## Visual Representation

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATAPOINT GENERATION PIPELINE                │
└─────────────────────────────────────────────────────────────────┘

PHASE 1: TOOL SELECTION
━━━━━━━━━━━━━━━━━━━━━━━━

┌──────────────────────┐
│   BFCL Tool Pool     │
│   (JSONL file)       │
│   - 1000+ tools      │
│   - 10+ categories   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐         ┌─────────────────────┐
│  Load & Categorize   │         │ Categories:         │
│  ─────────────────   │         │ • Calendar          │
│  Group by category   │────────►│ • Communication     │
│  Infer from source   │         │ • File System       │
└──────────┬───────────┘         │ • Travel            │
           │                     │ • Stock/Trading     │
           ▼                     │ • Vehicle Control   │
┌──────────────────────┐         └─────────────────────┘
│  Select Subset       │
│  ─────────────────   │         Strategy:
│  Uniform coverage    │         • max_tools = 80
│  across categories   │         • tools_per_category = 80/num_categories
└──────────┬───────────┘         • Random selection within category
           │
           ▼
┌──────────────────────┐
│  Temp Tool Pool      │
│  (60-80 tools)       │
└──────────┬───────────┘
           │
           │
PHASE 2: BLUEPRINT GENERATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
           │
           ▼
┌──────────────────────┐         Template:
│  Generate Query      │         "Using tools from {category}, perform
│  ─────────────────   │         a {num_actions}-step multi-step operation
│  Focus category      │         that requires retrieving information and
│  Variation number    │         then creating or updating a record."
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│                    LLM BLUEPRINT GENERATION                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    SYSTEM PROMPT                        │  │
│  │  ─────────────────────────────────────────────────────│  │
│  │  "You are a Task Blueprint Generator for APIGen-MT.    │  │
│  │   Generate a JSON object with:                         │  │
│  │   - q: User's request                                  │  │
│  │   - a_gt_steps: List of tool calls                     │  │
│  │   - o_gt: Expected outcome                             │  │
│  │                                                        │  │
│  │   Use placeholders: {{tool.output.field}}             │  │
│  │   Response: JSON in ```json ... ``` block             │  │
│  │                                                        │  │
│  │   [Tool Schemas - 60-80 tools]                        │  │
│  │   [Previous Reviews - if regenerating]                │  │
│  │   [Validation Errors - if regenerating]               │  │"
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                     USER PROMPT                         │  │
│  │  ─────────────────────────────────────────────────────│  │
│  │  "User Query: Using tools from Calendar, perform...   │  │
│  │                                                        │  │
│  │   [Feedback from previous attempt - if any]           │  │
│  │                                                        │  │
│  │   Generate the Blueprint JSON."                        │  │"
│  └────────────────────────────────────────────────────────┘  │
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐         Example Output:
│  Blueprint JSON      │         {
│  ─────────────────   │           "q": "Find calendar events...",
│  • q: query          │           "a_gt_steps": [
│  • a_gt_steps: [...] │             {
│  • o_gt: outcome     │               "tool_calls": [{
│                      │                 "tool_name": "get_events",
│  Placeholders used   │                 "arguments": {
│  for dependencies    │                   "start": "2025-01-20"
│                      │                 }
│                      │               }]
│                      │             },
│                      │             {
│                      │               "tool_calls": [{
│                      │                 "tool_name": "create_event",
│                      │                 "arguments": {
│                      │                   "summary": "Review",
│                      │                   "start": "{{get_events.output.start}}"
│                      │                 }
│                      │               }]
│                      │             }
│                      │           ],
│                      │           "o_gt": "Successfully created..."
│                      │         }
└──────────┬───────────┘
           │
           ▼
PHASE 3: VALIDATION
━━━━━━━━━━━━━━━━━━━━
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│              FORMAT VALIDATION                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │  ✓ Pydantic model validation                       │  │
│  │  ✓ a_gt_steps not empty                            │  │
│  │  ✓ Each step has ≥1 tool call                      │  │
│  │  ✓ All required fields present                     │  │
│  └────────────────────────────────────────────────────┘  │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│           EXECUTABILITY VALIDATION                       │
│  ┌────────────────────────────────────────────────────┐  │
│  │  FOR each step in blueprint:                       │  │
│  │    FOR each tool_call in step:                     │  │
│  │      ├─► Check tool exists in pool                 │  │
│  │      ├─► Process placeholders                      │  │
│  │      │    (resolve {{tool.output.field}})          │  │
│  │      ├─► Validate required arguments               │  │
│  │      ├─► Simulate tool execution                   │  │
│  │      └─► Store output in execution_context         │  │
│  │                                                    │  │
│  │  execution_context accumulates results:            │  │
│  │  {                                                 │  │
│  │    "get_events.output": {                          │  │
│  │      "start": "2025-01-20T10:00:00Z",              │  │
│  │      "events": [...]                               │  │
│  │    }                                               │  │
│  │  }                                                 │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Result:                                                 │
│  {                                                       │
│    "is_valid_format": true,                             │
│    "is_executable": true,                               │
│    "executability_checks": [...],                       │
│    "overall_validation_passed": true                    │
│  }                                                       │
└──────────┬───────────────────────────────────────────────┘
           │
           ├──────► [IF VALIDATION FAILS]
           │              │
           │              ▼
           │        ┌──────────────┐
           │        │ Format Error │
           │        │ Message      │
           │        └──────┬───────┘
           │               │
           │               ▼
           │        ┌──────────────────────┐
           │        │ Regenerate Blueprint │
           │        │ with Feedback        │
           │        └──────────┬───────────┘
           │                   │
           │                   └──────► [Back to Phase 2]
           │
           ▼
PHASE 4: QUALITY REVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│                 LLM QUALITY REVIEW                       │
│  ┌────────────────────────────────────────────────────┐  │
│  │                 SYSTEM PROMPT                       │  │
│  │  ─────────────────────────────────────────────────│  │
│  │  "You are an expert in data quality control.       │  │
│  │   Review the blueprint for:                        │  │
│  │   - Clarity and realism                            │  │
│  │   - Logical coherence                              │  │
│  │   - Tool usage appropriateness                     │  │
│  │   - Overall quality (Excellent/Good/Fair/Poor)     │  │
│  │                                                    │  │
│  │   Note: Placeholders are correct and expected."    │  │"
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │                  USER PROMPT                        │  │
│  │  ─────────────────────────────────────────────────│  │
│  │  Review this blueprint:                            │  │
│  │                                                    │  │
│  │  1. Query: [query text]                            │  │
│  │  2. Blueprint: [JSON]                              │  │
│  │  3. Reasoning: [LLM's generation reasoning]       │  │
│  │  4. Validation: [validation results]               │  │
│  │                                                    │  │
│  │  Assess quality and provide feedback."             │  │"
│  └────────────────────────────────────────────────────┘  │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│  Quality Assessment  │
│  ─────────────────   │         Decision Tree:
│  {                   │
│    "quality": "Good",│         Excellent/Good
│    "feedback": "...",│              │
│    "corrections": "" │              ▼
│  }                   │         ┌─────────────┐
└──────────┬───────────┘         │   ACCEPT    │
           │                     │   Blueprint │
           ├──────► [IF Fair/Poor]              │
           │              │              └─────────────┘
           │              ▼
           │        ┌──────────────┐
           │        │   Generate   │
           │        │   Feedback   │         Example Feedback:
           │        └──────┬───────┘         "The blueprint needs more
           │               │                 specific arguments. Consider
           │               ▼                 adding error handling."
           │        ┌──────────────────────┐
           │        │ Regenerate Blueprint │
           │        │ with Feedback        │
           │        │ (max_attempts = 3)   │
           │        └──────────┬───────────┘
           │                   │
           │                   └──────► [Back to Phase 2]
           │
           ▼
PHASE 5: EXECUTION SIMULATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│              SIMULATE EXECUTION TRACE                    │
│  ┌────────────────────────────────────────────────────┐  │
│  │  FOR each tool_call in blueprint.a_gt_steps:       │  │
│  │                                                    │  │
│  │    simulated_return = simulate_tool_return(        │  │
│  │      function_name,                                │  │
│  │      arguments                                     │  │
│  │    )                                               │  │
│  │                                                    │  │
│  │    execution_step = {                              │  │
│  │      "step_index": idx,                            │  │
│  │      "function_name": tool_name,                   │  │
│  │      "arguments": {...},                           │  │
│  │      "simulated_return": {...},                    │  │
│  │      "timestamp": "2025-01-15T10:50:00Z"           │  │
│  │    }                                               │  │
│  │                                                    │  │
│  │    execution_trace.append(execution_step)          │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Tool-Specific Simulators:                               │
│  ┌────────────────────────────────────────────────────┐  │
│  │  File System: ls, cat, mkdir, mv, rm              │  │
│  │  Math: add, subtract, multiply, divide, sqrt      │  │
│  │  Trading: get_stock_info, buy_stock               │  │
│  │  Calendar: get_events, create_event               │  │
│  │  Vehicle: start_engine, lock_doors                │  │
│  │  Communication: send_message, get_message         │  │
│  │  Travel: book_flight, book_hotel                  │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Example Simulated Output:                               │
│  {                                                       │
│    "status": "success",                                 │
│    "result": {                                          │
│      "type": "list",                                    │
│      "events": [                                        │
│        {"title": "Team Meeting",                        │
│         "start": "2025-01-20T10:00:00Z"},              │
│        {"title": "Project Review",                      │
│         "start": "2025-01-21T14:00:00Z"}               │
│      ],                                                  │
│      "count": 2                                         │
│    },                                                    │
│    "simulated": true                                     │
│  }                                                       │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼
PHASE 6: SAVE DATAPOINT
━━━━━━━━━━━━━━━━━━━━━━━━
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│                 FINAL DATAPOINT                          │
│  {                                                       │
│    "query": "Using tools from Calendar...",             │
│    "blueprint": {                                        │
│      "q": "Find my calendar events...",                 │
│      "a_gt_steps": [...],                                │
│      "o_gt": "Successfully created..."                  │
│    },                                                    │
│    "simulated_execution_trace": [                        │
│      {                                                   │
│        "step_index": 0,                                 │
│        "function_name": "get_calendar_events",          │
│        "arguments": {                                    │
│          "start_date": "2025-01-20",                    │
│          "end_date": "2025-01-27"                       │
│        },                                                │
│        "simulated_return": {                            │
│          "status": "success",                           │
│          "result": {...}                                │
│        },                                                │
│        "timestamp": "2025-01-15T10:50:00Z"              │
│      },                                                  │
│      ...                                                 │
│    ],                                                    │
│    "validation_result": {                                │
│      "is_valid_format": true,                           │
│      "is_executable": true,                             │
│      "overall_validation_passed": true                  │
│    },                                                    │
│    "llm_review_history": [                               │
│      {                                                   │
│        "quality_assessment": "Good",                    │
│        "feedback_summary": "Logically coherent..."      │
│      }                                                   │
│    ],                                                    │
│    "generation_attempts": 2,                             │
│    "tools_used": ["get_calendar_events", ...],          │
│    "categories_used": ["Calendar"],                     │
│    "timestamp": "2025-01-15T10:55:32Z"                  │
│  }                                                       │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│  Save to JSONL       │
│  ─────────────────   │
│  data/bfcl_datapoints│
│  /bfcl_multiturn_    │
│  datapoints_         │
│  {timestamp}.jsonl   │
└──────────────────────┘

═══════════════════════════════════════════════════════════════

KEY DESIGN PATTERNS
━━━━━━━━━━━━━━━━━━━━

1. PLACEHOLDER SYSTEM
   ┌─────────────────────────────────────────┐
   │  {{tool_name.output.field_name}}        │
   │                                         │
   │  Examples:                              │
   │  • {{get_events.output.events.0.id}}   │
   │  • {{search_users.output.user_id}}      │
   │  • {{create_file.output.file_path}}     │
   │                                         │
   │  Resolved during validation using       │
   │  execution_context from previous steps  │
   └─────────────────────────────────────────┘

2. ITERATIVE REFINEMENT
   ┌─────────────────────────────────────────┐
   │  Loop (max_attempts = 3):               │
   │                                         │
   │  Generate → Validate → Review           │
   │      ▲                   │              │
   │      │                   ▼              │
   │      └───── [IF NOT GOOD] ─────┘        │
   │                                         │
   │  Feedback includes:                     │
   │  • Validation errors                    │
   │  • LLM quality assessments              │
   │  • Specific correction suggestions     │
   └─────────────────────────────────────────┘

3. UNIFORM CATEGORY COVERAGE
   ┌─────────────────────────────────────────┐
   │  tools_per_category = 80 / num_cats    │
   │                                         │
   │  Ensures:                               │
   │  • Equal representation                 │
   │  • Diverse scenarios                    │
   │  • Cross-category reasoning             │
   │  • No bias toward common tools          │
   └─────────────────────────────────────────┘

4. MULTI-STAGE VALIDATION
   ┌─────────────────────────────────────────┐
   │  Stage 1: Format (Pydantic)            │
   │     ↓                                   │
   │  Stage 2: Executability (Simulation)    │
   │     ↓                                   │
   │  Stage 3: Quality (LLM Review)          │
   │     ↓                                   │
   │  Stage 4: Accept or Regenerate          │
   └─────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════

QUALITY THRESHOLDS
━━━━━━━━━━━━━━━━━━

  Excellent  ─────► ✓ ACCEPT immediately
      │
  Good       ─────► ✓ ACCEPT immediately
      │
  Fair       ─────► ✗ REGENERATE with feedback
      │
  Poor       ─────► ✗ REGENERATE with feedback

═══════════════════════════════════════════════════════════════

SUCCESS METRICS
━━━━━━━━━━━━━━━━

✓ Blueprint format validation passed
✓ All tools exist in pool
✓ Placeholders correctly resolved
✓ Required arguments provided
✓ Execution simulation successful
✓ LLM quality assessment: Excellent/Good
✓ Complete execution trace generated
✓ Datapoint saved to output file