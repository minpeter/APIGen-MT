# APIGen-MT Documentation Index

## Datapoint Generation Documentation

This directory contains comprehensive documentation for understanding and using the APIGen-MT datapoint generation pipeline.

---

## 📚 Documentation Files

### 1. 📖 [DATAPOINT_GENERATION_README.md](./DATAPOINT_GENERATION_README.md)
**Start Here** - Quick reference and overview of the entire pipeline.

**Contents:**
- Quick start guide with command examples
- Pipeline summary (input → process → output)
- Key features and design patterns
- Example output structure
- Troubleshooting guide
- Performance characteristics

**When to read:** First time setup, quick reference, or troubleshooting.

---

### 2. 🔧 [DATAPOINT_GENERATION_ALGORITHM.md](./DATAPOINT_GENERATION_ALGORITHM.md)
**Deep Dive** - Detailed algorithmic description with code references.

**Contents:**
- 9-step workflow breakdown
- Detailed algorithm for each phase:
  - Tool pool loading and categorization
  - Uniform category coverage selection
  - Query template generation
  - Blueprint generation (LLM Prompt 1)
  - Format & executability validation
  - Quality review (LLM Prompt 2)
  - Iterative refinement loop
  - Execution trace simulation
  - Datapoint saving
- Complete LLM prompt structures
- Placeholder system explanation
- Configuration parameters
- Example end-to-end flow

**When to read:** When you need to understand the implementation details or modify the pipeline.

---

### 3. 📊 [DATAPOINT_GENERATION_FLOW_DIAGRAM.md](./DATAPOINT_GENERATION_FLOW_DIAGRAM.md)
**Visual Guide** - ASCII art diagrams of the entire pipeline.

**Contents:**
- Phase-by-phase visual flow
- Decision trees for validation
- Data flow diagrams
- LLM prompt visualization
- Key design patterns (placeholder system, iterative refinement)
- Quality threshold decision tree
- Success metrics checklist

**When to read:** When you need a visual reference or want to understand the pipeline flow quickly.

---

## 🎯 Quick Navigation

### I want to...

#### Generate datapoints
→ See [DATAPOINT_GENERATION_README.md](./DATAPOINT_GENERATION_README.md) - Quick Start section

#### Understand the algorithm
→ See [DATAPOINT_GENERATION_ALGORITHM.md](./DATAPOINT_GENERATION_ALGORITHM.md) - Detailed Steps

#### See the pipeline visually
→ See [DATAPOINT_GENERATION_FLOW_DIAGRAM.md](./DATAPOINT_GENERATION_FLOW_DIAGRAM.md) - Visual Flow

#### Understand LLM prompts
→ See [DATAPOINT_GENERATION_ALGORITHM.md](./DATAPOINT_GENERATION_ALGORITHM.md) - Step 4 & Step 6

#### Fix a validation error
→ See [DATAPOINT_GENERATION_README.md](./DATAPOINT_GENERATION_README.md) - Troubleshooting

#### Modify the generation logic
→ See [DATAPOINT_GENERATION_ALGORITHM.md](./DATAPOINT_GENERATION_ALGORITHM.md) - All steps

---

## 🗂️ Related Implementation Files

### Core Scripts
- **`src/generate_datapoints_bfcl.py`** - Main generation script
- **`src/apigen-phase1.py`** - Blueprint generation and validation
- **`src/tool_manager.py`** - Tool pool management
- **`src/tool_simulation.py`** - Execution simulation
- **`src/llm_client.py`** - LLM API interaction

### Configuration
- **`src/.env`** - API credentials (OPENAI_API_KEY, OPENAI_API_BASE)
- **`data/bfcl_tool_pool/BFCL_v3.jsonl`** - Source tool definitions

### Output
- **`data/bfcl_datapoints/`** - Generated datapoints directory

---

## 📝 Document Summaries

### Algorithm Overview

The pipeline follows this high-level flow:

```
Load Tool Pool (categorized)
    ↓
Select Random Subset (uniform coverage)
    ↓
Generate Query Template
    ↓
Generate Blueprint (LLM Prompt 1)
    ↓
Validate Format & Executability
    ↓
Quality Review (LLM Prompt 2)
    ↓
[If needed] Refine (iterate)
    ↓
Simulate Execution Trace
    ↓
Save Datapoint
```

### Key Design Patterns

1. **Placeholder System** - `{{tool.output.field}}` for dependencies
2. **Uniform Category Coverage** - Equal representation across tool categories
3. **Multi-Stage Validation** - Format, executability, and quality checks
4. **Iterative Refinement** - Feedback-driven improvement (max 3 attempts)
5. **Execution Simulation** - Tool-specific simulated outputs

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API access (or compatible endpoint)
- BFCL tool pool file

### Quick Start

```bash
# 1. Configure API credentials
echo "OPENAI_API_KEY=your-key" > src/.env
echo "OPENAI_API_BASE=https://api.example.com/v1" >> src/.env

# 2. Generate datapoints
cd src
python generate_datapoints_bfcl.py --num-datapoints 10 --num-actions 2

# 3. Check output
ls -l ../data/bfcl_datapoints/
```

### Next Steps

1. Read the [README](./DATAPOINT_GENERATION_README.md) for detailed usage
2. Explore the [algorithm](./DATAPOINT_GENERATION_ALGORITHM.md) to understand the process
3. Reference the [flow diagram](./DATAPOINT_GENERATION_FLOW_DIAGRAM.md) for visual understanding

---

## 📖 Reading Order Recommendation

### For New Users
1. **Start:** DATAPOINT_GENERATION_README.md (overview)
2. **Then:** DATAPOINT_GENERATION_FLOW_DIAGRAM.md (visual understanding)
3. **Deep dive:** DATAPOINT_GENERATION_ALGORITHM.md (implementation details)

### For Developers
1. **Start:** DATAPOINT_GENERATION_FLOW_DIAGRAM.md (quick visual)
2. **Then:** DATAPOINT_GENERATION_ALGORITHM.md (implementation)
3. **Reference:** DATAPOINT_GENERATION_README.md (troubleshooting)

### For Researchers
1. **Start:** DATAPOINT_GENERATION_README.md (features & metrics)
2. **Then:** DATAPOINT_GENERATION_ALGORITHM.md (algorithm details)
3. **Reference:** DATAPOINT_GENERATION_FLOW_DIAGRAM.md (design patterns)

---

## 📊 Documentation Statistics

- **Total Lines:** 1,424 lines of documentation
- **Files:** 3 comprehensive documents
- **Coverage:** End-to-end pipeline documentation
- **Detail Level:** From quick reference to deep implementation details

---

## 🔗 External References

- **BFCL Dataset:** Berkeley Function Calling Leaderboard
- **APIGen-MT Paper:** Multi-turn API interaction generation methodology
- **Pydantic Documentation:** Data validation framework

---

## 💡 Tips

- Use `--debug` flag to see detailed LLM interactions
- Start with small `--num-datapoints` (10-20) for testing
- Check validation results in output to understand failures
- Review LLM quality assessments to improve generation
- Uniform category coverage ensures diverse training data

---

**Questions?** Start with the README, then dive into the algorithm documentation for detailed answers.

**Last Updated:** 2025-01-15  
**Version:** 1.0