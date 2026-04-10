# Implementation: --num-actions Command-Line Argument

## Summary
Added a `--num-actions` command-line argument to `generate_datapoints_bfcl.py` that allows users to specify the number of actions/steps to generate in each datapoint.

## Changes Made

### 1. Argument Parser (Lines 73-83)
Added new argument with the following configuration:
```python
parser.add_argument(
    '--num-actions', '-a',
    type=int,
    default=2,
    help='Number of actions/steps to generate per datapoint (default: 2)'
)
```

### 2. Main Function (Lines 215, 219)
- Added `num_actions = args.num_actions` to extract the parameter
- Added print statement to display the target actions per datapoint

### 3. Query Generation (Line 294)
Updated the query template to include the specified number of steps:
```python
query = f"Using tools from {focus_category}, perform a {num_actions}-step multi-step operation that requires retrieving information and then creating or updating a record. (variation #{attempt + len(datapoints)})"
```

### 4. Documentation (Lines 6-20)
Updated module docstring with:
- New option description
- Updated examples showing different num_actions values

## Usage Examples

```bash
# Generate 100 datapoints with 2 actions each (default)
python generate_datapoints_bfcl.py

# Generate 50 datapoints with 3 actions each
python generate_datapoints_bfcl.py --num-datapoints 50 --num-actions 3

# Generate 10 datapoints with 5 actions each in debug mode
python generate_datapoints_bfcl.py -n 10 -a 5 --debug
```

## Testing

Verified that:
- ✅ Argument parser correctly accepts `--num-actions` and `-a` flags
- ✅ Default value is 2 when not specified
- ✅ Custom values are properly passed through
- ✅ Query generation includes the correct number of steps in the prompt
- ✅ Help text displays correctly with examples

## Impact

This change allows users to control the complexity of generated datapoints by specifying how many tool calls/steps should be included in each blueprint. This is useful for:
- Generating datasets with varying complexity levels
- Testing LLM capabilities with different multi-step reasoning requirements
- Creating benchmarks for different action sequence lengths