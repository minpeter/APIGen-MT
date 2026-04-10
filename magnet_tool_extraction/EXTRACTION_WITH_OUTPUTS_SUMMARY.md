# BFCL Tool Extraction with LLM Output Prediction

## Summary

Successfully implemented BFCL tool extraction with LLM-predicted output types and descriptions.

## Changes Made

### 1. Download Script (`download_bfcl_v4.py`)
- Automatically downloads BFCL_v4 from HuggingFace if it doesn't exist
- Falls back to existing BFCL_v3 if v4 download fails
- Supports HuggingFace token authentication via `HF_TOKEN` env var or `--token` argument
- Handles missing datasets gracefully with clear error messages

### 2. LLM Output Predictor (`llm_output_predictor.py`)
- Uses NVIDIA Nemotron model via OpenAI-compatible API
- Predicts `output_type` and `output_description` for each tool
- Analyzes tool schema (name, description, parameters) and invocation contexts
- Uses LLM-as-a-judge approach for accurate predictions
- Handles errors gracefully with fallback to "unknown" type

### 3. Extraction Script (`extract_bfcl_with_outputs.py`)
- Extracts tool definitions from BFCL multi_turn_func_doc
- Extracts invocation contexts from test files
- Integrates LLM predictor for output field generation
- Supports limiting tools for testing (`--limit` flag)
- Outputs JSONL format with enhanced tool definitions

## Usage

### Basic Usage
```bash
# Extract all tools from BFCL_v3 with output predictions
python extract_bfcl_with_outputs.py --version v3 --client nvidia

# Extract from BFCL_v4 (downloads if not available)
python extract_bfcl_with_outputs.py --version v4 --client nvidia

# Test with limited tools
python extract_bfcl_with_outputs.py --version v3 --limit 10 --client nvidia
```

### With Debug Mode
```bash
# Enable debug to see LLM prompts and responses
python extract_bfcl_with_outputs.py --version v3 --limit 3 --client nvidia --debug
```

### Force Download
```bash
# Force re-download of BFCL data
python extract_bfcl_with_outputs.py --version v4 --force-download --client nvidia
```

## Output Format

Each tool in the output JSONL file has the following structure:

```json
{
  "category": "Storage",
  "tool_name": "gorilla_file_system",
  "tool_description": "Functions provided by the gorilla file system toolkit.",
  "api_name": "cat",
  "api_description": "Display the contents of a file...",
  "parameters": {
    "type": "dict",
    "properties": {...},
    "required": [...],
    "optional": [...]
  },
  "output_type": "string",
  "output_description": "The contents of the specified file from the current directory, returned as a string."
}
```

## Test Results

Successfully tested with 10 tools from BFCL_v3:
- **Total tools processed**: 10
- **Successful predictions**: 8
- **Failed predictions**: 2 (JSON parsing errors, fallback to "unknown")
- **Output types predicted**: 
  - string (5 tools)
  - list of strings (2 tools)
  - list (1 tool)
  - unknown (2 tools - errors)

## LLM Configuration

- **Model**: nvidia/nemotron-3-super-120b-a12b
- **API Base**: https://integrate.api.nvidia.com/v1
- **API Key**: From `OPENAI_API_KEY` environment variable
- **Temperature**: 0.7
- **Max Tokens**: 500

## Files Created

1. `download_bfcl_v4.py` - Downloads BFCL data
2. `llm_output_predictor.py` - LLM-based output predictor
3. `extract_bfcl_with_outputs.py` - Main extraction script
4. `bfcl_v3_tools_with_outputs.jsonl` - Sample output (10 tools)

## Next Steps

To run full extraction on all 105 tools:

```bash
python extract_bfcl_with_outputs.py --version v3 --client nvidia
```

This will create `bfcl_v3_tools_with_outputs.jsonl` with all tools including predicted output fields.