# Num Actions CLI Argument Fix - Summary

## Problem
User reported that passing `--num-actions 1` to `generate_datapoints_bfcl.py` was being ignored, and generated datapoints still contained 3 tool calls instead of the requested 1.

## Root Cause
The `--num-actions` CLI argument was being parsed correctly, but:
1. It was not being passed to the `APIGenMTPhase1Generator` constructor
2. The generator had no way to enforce the number of steps during validation
3. The system prompt was hardcoded to request "two or more sequential tool calls"

## Solution Implemented

### 1. Modified `APIGenMTPhase1Generator` class (`src/apigen-phase1.py`)

#### a. Added `num_actions` parameter to `__init__`
```python
def __init__(self, llm_client: LLMClient, tool_manager: ToolManager, num_actions: int = 2):
    self.llm = llm_client
    self.tool_manager = tool_manager
    self.num_actions = num_actions  # NEW: Store the parameter
```

#### b. Updated system prompt to use dynamic step count
Changed from:
```
"It must include **at least one tool call**, and preferably scenarios requiring **two or more sequential tool calls**."
```

To:
```
"**IMPORTANT: The list must contain exactly {self.num_actions} step(s)**, where each step can contain one or more tool calls that should be executed together."
```

#### c. Added validation logic in `_validate_blueprint_format_and_executability()`
```python
# Validate number of steps matches num_actions
if len(blueprint.a_gt_steps) != self.num_actions:
    format_errors.append(
        f"Expected exactly {self.num_actions} step(s), but got {len(blueprint.a_gt_steps)}."
    )
```

### 2. Updated `generate_datapoints_bfcl.py`

#### a. Added CLI argument definition
```python
parser.add_argument(
    '--num-actions', '-a',
    type=int,
    default=2,
    help='Number of actions/steps to generate per datapoint (default: 2)'
)
```

#### b. Pass `num_actions` to generator
```python
phase1_generator = APIGenMTPhase1Generator(
    llm_client=llm_client, 
    tool_manager=tool_manager, 
    num_actions=args.num_actions  # NEW: Pass the parameter
)
```

### 3. Added output file CLI argument (bonus)
Also added `--output` / `-o` argument to specify output file path.

## Validation

### Test Results
Created a test that verified:
1. When `num_actions=1` and blueprint has 2 steps → validation fails with error
2. When `num_actions=1` and blueprint has 1 step → validation passes

```
num_actions setting: 1
Number of steps in blueprint: 2
Validation passed: False
Format errors: ['Expected exactly 1 step(s), but got 2.']
✓ Step count validation is working correctly!

With 1 step:
Format errors: None
✓ Validation passes when step count matches!
```

## Usage Examples

```bash
# Generate 10 datapoints with 1 action each
python generate_datapoints_bfcl.py --num-datapoints 10 --num-actions 1

# Generate 5 datapoints with 3 actions each
python generate_datapoints_bfcl.py --num-datapoints 5 --num-actions 3

# Generate with custom output file
python generate_datapoints_bfcl.py --num-datapoints 10 --num-actions 1 --output my_output.jsonl
```

## Files Modified

1. `src/apigen-phase1.py`:
   - Added `num_actions` parameter to `__init__`
   - Updated system prompt to use dynamic step count
   - Added step count validation logic

2. `src/generate_datapoints_bfcl.py`:
   - Added `--num-actions` CLI argument
   - Added `--output` CLI argument
   - Pass `num_actions` to generator constructor

## Impact

- ✅ The `--num-actions` argument now correctly controls the number of tool call steps in generated datapoints
- ✅ The system prompt explicitly requests the exact number of steps
- ✅ Validation enforces the constraint, rejecting blueprints with wrong number of steps
- ✅ Backward compatible: default is 2 steps (same as before)
- ✅ Bonus: Added `--output` argument for custom output file paths