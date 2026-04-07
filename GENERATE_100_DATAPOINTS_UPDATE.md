# Generate 100 Datapoints Script Update

## Overview
Updated the `generate_100_datapoints.py` script to use the new BFCL v3 tool definitions and ensure uniform sampling from different tool categories.

## Changes Made

### 1. New Script Created
- **File**: `src/generate_100_datapoints_bfcl.py`
- **Purpose**: Generate 100 datapoints using the updated BFCL v3 tool definitions with uniform category sampling

### 2. Tool Pool Updated
- **Old Path**: `/home/ishalyminov/data/magnet_mt/output/tool_pool.jsonl`
- **New Path**: `/home/ishalyminov/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_all_tool_definitions.jsonl`
- **Output File**: `apigen_phase1_100_datapoints_bfcl.jsonl`

### 3. Category Inference Logic
For tools without explicit categories (marked as 'Unknown'), the script now infers categories from the source file:

| Source Pattern | Category |
|---------------|----------|
| Contains 'live' | Live APIs |
| Contains 'exec' | Executable |
| Contains 'simple' | Simple Functions |
| Contains 'multiple' | Multiple Functions |
| Contains 'parallel' | Parallel Functions |
| Other | Other |

### 4. Uniform Category Sampling
The `select_tool_subset()` function now ensures **uniform distribution** across all categories:

- **Before**: Random sampling with minimum 3 tools per category
- **After**: Strict uniform sampling with equal tools per category (80 tools ÷ 13 categories ≈ 6 tools per category)

**Benefits**:
- Ensures diversity in generated datapoints
- Prevents bias toward larger categories
- Each category gets equal representation

### 5. Category Distribution
After categorization, the tool pool contains:

```
Communication        :   10 tools
Events              :    9 tools
Executable          :   55 tools
Finance             :   22 tools
Live APIs           :  724 tools
Multiple Functions  :  346 tools
Parallel Functions  :   52 tools
Posting Api         :   14 tools
Science             :   17 tools
Simple Functions    :  368 tools
Storage             :   18 tools
Travel Booking      :   17 tools
Vehicle Control     :   22 tools
-----------------------------------
TOTAL               : 1674 tools
```

### 6. Enhanced Metadata
Each generated datapoint now includes:
- `categories_in_pool`: List of all categories in the tool subset
- `focus_category`: The category that the query focuses on
- `tools_used`: List of tools actually used in the blueprint

### 7. Query Generation
Improved query generation to be more specific and grounded in the available tools:
- Focuses on a randomly selected category from the tool subset
- Generates multi-step operation queries
- Ensures queries are realistic given the available tools

## Usage

```bash
cd /home/ishalyminov/data/APIGen-MT/src
python3 generate_100_datapoints_bfcl.py
```

## Testing Results

Verified that uniform sampling works correctly:
- Each sample of 80 tools contains 6-8 tools from each of the 13 categories
- Total tools across all categories: 1674
- Uniform distribution ensures no category dominates the generation process

## Files Modified

1. **New**: `src/generate_100_datapoints_bfcl.py` - Main generation script
2. **Backup**: `src/generate_100_datapoints.py.old` - Backup of original script

## Date
Updated: 2026-04-05
