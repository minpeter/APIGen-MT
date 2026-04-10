# BFCL Tool Extraction with Auto-Download

This directory contains scripts for extracting tool definitions and invocations from the Berkeley Function Calling Leaderboard (BFCL) dataset. The scripts now support automatic downloading of the dataset from HuggingFace if it doesn't exist locally.

## Overview

The extraction pipeline has been updated to:
1. **Automatically download BFCL data** from HuggingFace if not present
2. Support **multiple BFCL versions** (currently v3, with v4 support when available)
3. Provide a **complete extraction pipeline** with a single command

## Files

### New Files (v2 versions with auto-download)

- **`bfcl_downloader.py`** - Module for downloading BFCL datasets from HuggingFace
- **`extract_all_bfcl_tools_v2.py`** - Extract all tool definitions from BFCL (with auto-download)
- **`extract_bfcl_complete_v2.py`** - Extract tool definitions and invocations (with auto-download)
- **`analyze_bfcl_all_categories_v2.py`** - Analyze tool usage by category (with auto-download)
- **`run_bfcl_extraction.sh`** - Complete extraction pipeline script

### Original Files (for reference)

- `extract_all_bfcl_tools.py` - Original extraction script (hardcoded paths)
- `extract_bfcl_complete.py` - Original complete extraction script
- `analyze_bfcl_all_categories.py` - Original analysis script
- `parse_bfcl.py` - BFCL parser module (unchanged)
- `tool_definition.py` - Tool definition classes (unchanged)

## Quick Start

### Run the Complete Pipeline

```bash
# Using default settings (BFCL v3, existing data directory)
./run_bfcl_extraction.sh

# Specify version and data directory
./run_bfcl_extraction.sh --version v3 --data-dir /path/to/data

# Force re-download
./run_bfcl_extraction.sh --version v3 --force-download
```

### Run Individual Scripts

```bash
# Download BFCL data (if needed)
python3 bfcl_downloader.py /path/to/data v3

# Extract all tool definitions
python3 extract_all_bfcl_tools_v2.py --version v3 --data-dir /path/to/data

# Extract complete tool information (definitions + invocations)
python3 extract_bfcl_complete_v2.py --version v3 --data-dir /path/to/data

# Analyze tool usage by category
python3 analyze_bfcl_all_categories_v2.py --version v3 --data-dir /path/to/data
```

## Command Line Options

### Common Options

All v2 scripts support:

- `--version VERSION` - BFCL version to use (default: v3)
- `--data-dir DATA_DIR` - Base data directory (default: /home/ishalyminov/data/magnet_mt/data)
- `--force-download` - Force re-download even if dataset exists

### Examples

```bash
# Use BFCL v3 (default)
python3 extract_all_bfcl_tools_v2.py

# Use a specific version
python3 extract_all_bfcl_tools_v2.py --version v3

# Download to a custom directory
python3 extract_all_bfcl_tools_v2.py --data-dir /tmp/bfcl_data

# Force fresh download
python3 extract_all_bfcl_tools_v2.py --force-download
```

## Output Files

The extraction pipeline generates the following output files:

1. **`bfcl_{version}_all_tool_definitions.jsonl`** - All tool definitions from BFCL (multi-turn + test files)
2. **`bfcl_{version}_all_tools_summary.md`** - Summary report of all tools
3. **`bfcl_{version}_tool_definitions.jsonl`** - Multi-turn tool definitions only
4. **`bfcl_{version}_invocation_examples.jsonl`** - Tool invocations with arguments and context
5. **`bfcl_{version}_samples_human_readable.md`** - Human-readable samples for manual inspection
6. **`bfcl_{version}_all_categories_analysis.json`** - Detailed category-wise analysis

## BFCL Dataset Information

- **HuggingFace Dataset**: `gorilla-llm/Berkeley-Function-Calling-Leaderboard`
- **Current Version**: v3 (as of 2025-07-17)
- **GitHub**: https://github.com/ShishirPatil/gorilla

### Dataset Structure

```
BFCL_v3/
├── BFCL_v3_simple.json
├── BFCL_v3_multiple.json
├── BFCL_v3_parallel.json
├── BFCL_v3_live_*.json
├── BFCL_v3_exec_*.json
├── BFCL_v3_multi_turn_*.json
├── multi_turn_func_doc/
│   ├── gorilla_file_system.json
│   ├── trading_bot.json
│   ├── ticket_api.json
│   └── ...
└── possible_answer/
    └── *.json
```

## Features

### Automatic Download

- Checks if BFCL dataset exists locally
- Downloads from HuggingFace if not present
- Supports resuming partial downloads
- Handles both test files and function documentation

### Version Flexibility

- Support for multiple BFCL versions
- Easy to upgrade when v4 becomes available
- Version-specific output file naming

### Comprehensive Extraction

- Extracts from all BFCL test file categories
- Parses multi-turn function documentation
- Merges definitions from multiple sources
- Generates detailed statistics and analysis

## Statistics

From BFCL v3 extraction:

- **Total unique tools**: 1,674
- **Multi-turn tools**: 129 (from `multi_turn_func_doc/`)
- **Test file tools**: 1,552 (embedded in test files)
- **Categories**: 8 main categories
- **Invocation examples**: 3,641 (from multi-turn test files)

### Top Tool Categories

| Category | Tools | Description |
|----------|-------|-------------|
| Storage | 18 | File system operations (gorilla_file_system) |
| Finance | 22 | Trading and financial operations (trading_bot) |
| Vehicle Control | 22 | Vehicle operations (vehicle_control) |
| Science | 17 | Math and science operations (math_api) |
| Travel Booking | 17 | Travel and booking operations (travel_booking) |
| Posting Api | 14 | Social media operations (posting_api) |
| Communication | 10 | Messaging operations (message_api) |
| Events | 9 | Event and ticket operations (ticket_api) |

## Troubleshooting

### Common Issues

1. **HuggingFace Rate Limits**
   ```
   Warning: You are sending unauthenticated requests to the HF Hub.
   ```
   Solution: Set `HF_TOKEN` environment variable for higher rate limits

2. **Missing Dependencies**
   ```bash
   pip install huggingface_hub datasets
   ```

3. **Permission Errors**
   ```bash
   chmod +x run_bfcl_extraction.sh
   ```

### Verifying Downloads

```bash
# Check downloaded files
ls -lh /path/to/data/BFCL_v3/

# Verify function docs
ls -lh /path/to/data/BFCL_v3/multi_turn_func_doc/

# Check output files
ls -lh bfcl_v3_*.jsonl
```

## Future Updates

When BFCL v4 becomes available:

1. Update `BFCL_DEFAULT_VERSION` in `bfcl_downloader.py`
2. Run extraction with `--version v4`
3. Compare outputs with v3

## Related Documentation

- [BFCL Official Documentation](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [HuggingFace Dataset](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
- [Magnet Paper](https://arxiv.org/abs/2404.16002) - Section 3.4 discusses BFCL usage

## License

BFCL dataset is licensed under Apache 2.0. See [BFCL License](https://github.com/ShishirPatil/gorilla/blob/main/LICENSE).