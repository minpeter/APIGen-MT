# BFCL Tool Extraction - Quick Reference Card

## 🚀 Quick Start

```bash
# Run complete extraction pipeline (downloads data if needed)
./run_bfcl_extraction.sh
```

## 📥 Download Only

```bash
# Download BFCL v3 data
python3 bfcl_downloader.py /path/to/data v3

# Or use Python directly
python3 -c "from bfcl_downloader import ensure_bfcl_data; ensure_bfcl_data('/path/to/data', 'v3')"
```

## 🔧 Individual Scripts

### Extract All Tools
```bash
python3 extract_all_bfcl_tools_v2.py --version v3 --data-dir /path/to/data
```

### Extract Complete Information
```bash
python3 extract_bfcl_complete_v2.py --version v3 --data-dir /path/to/data
```

### Analyze by Category
```bash
python3 analyze_bfcl_all_categories_v2.py --version v3 --data-dir /path/to/data
```

## 🎯 Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--version` | BFCL version | v3 |
| `--data-dir` | Data directory | /home/ishalyminov/data/magnet_mt/data |
| `--force-download` | Force re-download | False |

## 📊 Output Files

| File | Description | Size |
|------|-------------|------|
| `bfcl_v3_all_tool_definitions.jsonl` | All tool definitions | ~2.1M |
| `bfcl_v3_tool_definitions.jsonl` | Multi-turn tools only | ~81K |
| `bfcl_v3_invocation_examples.jsonl` | Tool invocations | ~5.0M |
| `bfcl_v3_samples_human_readable.md` | Human-readable samples | ~36K |
| `bfcl_v3_all_tools_summary.md` | Summary report | ~1.6K |
| `bfcl_v3_all_categories_analysis.json` | Category analysis | ~17K |

## 📈 Statistics (BFCL v3)

- **Total Tools**: 1,674
- **Multi-turn Tools**: 129
- **Test File Tools**: 1,552
- **Categories**: 8
- **Invocation Examples**: 3,641

## 🔄 Force Fresh Download

```bash
# Force re-download and re-extract
./run_bfcl_extraction.sh --force-download

# Or with specific directory
./run_bfcl_extraction.sh --data-dir /tmp/fresh_bfcl --force-download
```

## 🔍 Verify Installation

```bash
# Check if dependencies are installed
python3 -c "import huggingface_hub; print('✅ huggingface_hub installed')"
python3 -c "import datasets; print('✅ datasets installed')"

# Verify scripts
ls -l *.py *.sh | grep -E "(v2|downloader|run_bfcl)"

# Test downloader
python3 bfcl_downloader.py /tmp/test v3
```

## 🐛 Troubleshooting

### HuggingFace Rate Limits
```bash
# Set HF_TOKEN for higher rate limits
export HF_TOKEN=your_token_here
./run_bfcl_extraction.sh
```

### Missing Dependencies
```bash
pip install huggingface_hub datasets
```

### Permission Denied
```bash
chmod +x run_bfcl_extraction.sh
```

### Check Logs
```bash
# Run with verbose output
python3 extract_all_bfcl_tools_v2.py --version v3 2>&1 | tee extraction.log
```

## 📚 Dataset Information

- **HuggingFace**: `gorilla-llm/Berkeley-Function-Calling-Leaderboard`
- **GitHub**: https://github.com/ShishirPatil/gorilla
- **Current Version**: v3
- **License**: Apache 2.0

## 🔮 Future Versions

When BFCL v4 becomes available:

```bash
# Update default version in bfcl_downloader.py
# Then run:
./run_bfcl_extraction.sh --version v4
```

## 📖 Documentation

- Full documentation: `README_AUTO_DOWNLOAD.md`
- Implementation details: `IMPLEMENTATION_SUMMARY.md`
- Original scripts: `extract_all_bfcl_tools.py`, `extract_bfcl_complete.py`

## 💡 Tips

1. **First time?** Just run `./run_bfcl_extraction.sh`
2. **Need fresh data?** Use `--force-download`
3. **Custom location?** Use `--data-dir /your/path`
4. **Compare versions?** Run with different `--version` flags
5. **Check outputs?** Use `ls -lh bfcl_v3_*.jsonl`

## 🎓 Example Workflow

```bash
# 1. Run extraction
./run_bfcl_extraction.sh

# 2. Check outputs
ls -lh bfcl_v3_*.jsonl

# 3. View summary
cat bfcl_v3_all_tools_summary.md

# 4. Explore invocations
head -n 5 bfcl_v3_invocation_examples.jsonl | jq .

# 5. Analyze categories
cat bfcl_v3_all_categories_analysis.json | jq '.metadata'
```

---

**Need help?** Check `README_AUTO_DOWNLOAD.md` or `IMPLEMENTATION_SUMMARY.md`