# BFCL Tool Extraction - Quick Start

## Main Extraction Script

**Use this script**: `extract_bfcl_with_outputs.py`

This is the only extraction script you need. It:
- Extracts tool definitions from BFCL_v3
- Predicts output types and descriptions using LLM
- Downloads BFCL_v3 automatically if needed

## Usage

```bash
# Basic usage (uses BFCL_v3 by default)
python extract_bfcl_with_outputs.py

# With custom data directory
python extract_bfcl_with_outputs.py --data-dir /path/to/data

# Force re-download
python extract_bfcl_with_outputs.py --force-download

# Limit tools for testing
python extract_bfcl_with_outputs.py --limit 10

# Use different LLM client
python extract_bfcl_with_outputs.py --client friendli
```

## Output

The script generates:
- `bfcl_v3_tools_with_outputs.jsonl` - Enhanced tool definitions with predicted outputs

## Important Notes

1. **BFCL Version**: This script only works with BFCL_v3 (hardcoded, no --version option)
2. **LLM Client**: Requires NVIDIA API key by default (set in `.env` file)
3. **Automatic Download**: BFCL_v3 is downloaded automatically if not found

## Deprecated Scripts (Do Not Use)

The following scripts have been removed:
- ❌ `extract_all_bfcl_tools.py` / `extract_all_bfcl_tools_v2.py`
- ❌ `extract_bfcl_complete.py` / `extract_bfcl_complete_v2.py`
- ❌ `analyze_bfcl_all_categories.py` / `analyze_bfcl_all_categories_v2.py`
- ❌ `run_bfcl_extraction.sh`

All functionality is now consolidated in `extract_bfcl_with_outputs.py`.