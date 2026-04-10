# APIGen-MT Datapoint Generation Documentation

## Overview

This directory contains comprehensive documentation for the APIGen-MT datapoint generation pipeline, which creates high-quality training data for multi-turn AI agent systems.

## Documentation Files

### 1. [DATAPOINT_GENERATION_ALGORITHM.md](./DATAPOINT_GENERATION_ALGORITHM.md)

**Comprehensive Algorithm Documentation**

This document provides a detailed description of the datapoint generation algorithm, including:

- **High-level workflow** - 9-step pipeline from tool loading to datapoint saving
- **Detailed step-by-step breakdown** - Each phase explained in depth
- **LLM prompt structures** - Exact prompts used for blueprint generation and quality review
- **Placeholder system** - How inter-step dependencies are managed
- **Validation pipeline** - Format, executability, and quality checks
- **Configuration parameters** - All tunable settings and their defaults
- **Example end-to-end flow** - Complete walkthrough of a single datapoint generation

**Key Topics:**
- Tool pool loading and categorization
- Uniform category coverage strategy
- Blueprint generation with placeholders
- Multi-stage validation (format, executability, LLM review)
- Iterative refinement loop
- Execution trace simulation
- Output format and structure

**Target Audience:** Developers who need to understand, modify, or extend the generation pipeline.

---

### 2. [DATAPOINT_GENERATION_FLOW_DIAGRAM.md](./DATAPOINT_GENERATION_FLOW_DIAGRAM.md)

**Visual Flow Representation**

This document provides a visual diagram of the entire generation pipeline using ASCII art, including:

- **Phase-by-phase breakdown** - Visual separation of each major stage
- **Decision trees** - Validation and quality gates
- **Data flow** - How information moves through the pipeline
- **Feedback loops** - Iterative refinement paths
- **Example data** - Sample inputs and outputs at each stage
- **Key design patterns** - Placeholder system, uniform coverage, multi-stage validation

**Key Visualizations:**
- Tool selection strategy (uniform category coverage)
- LLM blueprint generation prompts
- Validation decision tree
- Quality assessment flow
- Execution simulation process
- Complete datapoint structure

**Target Audience:** Anyone who needs a quick visual reference for understanding the pipeline flow.

---

## Quick Start

### Generate Datapoints

```bash
# Generate 100 datapoints with default settings (2 actions each)
python src/generate_datapoints_bfcl.py

# Generate 50 datapoints with 3 actions each
python src/generate_datapoints_bfcl.py --num-datapoints 50 --num-actions 3

# Generate 10 datapoints with 5 actions each in debug mode
python src/generate_datapoints_bfcl.py -n 10 -a 5 --debug
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-datapoints` | 100 | Total datapoints to generate |
| `--num-actions` | 2 | Tool calls per blueprint |
| `--debug` | False | Enable detailed LLM logging |

---

## Pipeline Summary

### Input
- **Tool Pool:** BFCL_v3.jsonl (1000+ tools across 10+ categories)
- **Query Template:** "Using tools from {category}, perform {N}-step operation..."

### Process
1. **Tool Selection:** Uniform sampling across categories (60-80 tools)
2. **Blueprint Generation:** LLM generates structured tool call sequence
3. **Validation:** Format check, executability simulation, placeholder resolution
4. **Quality Review:** LLM assesses quality (Excellent/Good/Fair/Poor)
5. **Refinement:** Iterative improvement with feedback (max 3 attempts)
6. **Simulation:** Generate realistic execution traces

### Output
- **Datapoints:** JSONL file with complete training examples
- **Structure:** Query, blueprint, execution trace, validation results, quality history

---

## Key Features

### 1. Placeholder System
```
{{tool_name.output.field_name}}
```
Enables multi-step dependencies between tool calls.

### 2. Uniform Category Coverage
Equal representation across all tool categories ensures diverse training data.

### 3. Multi-Stage Validation
- **Format:** Pydantic model validation
- **Executability:** Tool existence, argument validation, simulation
- **Quality:** LLM review for realism and coherence

### 4. Iterative Refinement
Automatic regeneration with feedback when quality is insufficient.

### 5. Execution Simulation
Tool-specific simulators generate realistic outputs for training targets.

---

## Example Output

```json
{
  "query": "Using tools from Calendar, perform a 2-step operation...",
  "blueprint": {
    "q": "Find my calendar events for next week...",
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
              "start_time": "{{get_calendar_events.output.start_time}}"
            }
          }
        ]
      }
    ],
    "o_gt": "Successfully created a summary meeting..."
  },
  "simulated_execution_trace": [...],
  "validation_result": {
    "is_valid_format": true,
    "is_executable": true,
    "overall_validation_passed": true
  },
  "llm_review_history": [
    {
      "quality_assessment": "Good",
      "feedback_summary": "Logically coherent and realistic."
    }
  ],
  "generation_attempts": 2
}
```

---

## Related Files

### Implementation
- `src/generate_datapoints_bfcl.py` - Main generation script
- `src/apigen-phase1.py` - Blueprint generation and validation logic
- `src/tool_manager.py` - Tool pool management
- `src/tool_simulation.py` - Execution simulation
- `src/llm_client.py` - LLM API interaction

### Configuration
- `src/.env` - API credentials and configuration
- `data/bfcl_tool_pool/` - Source tool definitions
- `data/bfcl_datapoints/` - Generated output directory

---

## Understanding the Prompts

### Blueprint Generation Prompt (Simplified)

**System Message:**
```
You are a Task Blueprint Generator.
Generate JSON with:
- q: User request
- a_gt_steps: Tool call sequence
- o_gt: Expected outcome

Use placeholders: {{tool.output.field}}
Available tools: [60-80 schemas]
```

**User Message:**
```
Query: Using tools from Calendar, perform 2-step operation...
Generate the Blueprint JSON.
```

### Quality Review Prompt (Simplified)

**System Message:**
```
You are a data quality expert.
Review blueprint for:
- Clarity and realism
- Logical coherence
- Tool usage appropriateness
Assess: Excellent/Good/Fair/Poor
```

**User Message:**
```
Blueprint: [JSON]
Validation: [results]
Assess quality and provide feedback.
```

---

## Validation Criteria

### Format Validation
- ✓ Pydantic model structure
- ✓ Non-empty `a_gt_steps`
- ✓ Each step has tool calls
- ✓ Required fields present

### Executability Validation
- ✓ Tool exists in pool
- ✓ Required arguments provided
- ✓ Placeholders resolved correctly
- ✓ Tool execution simulation succeeds

### Quality Assessment
- **Excellent:** Ready for training, no issues
- **Good:** Minor improvements possible, acceptable
- **Fair:** Needs refinement before use
- **Poor:** Significant issues, regenerate

---

## Performance Characteristics

- **Generation Time:** ~30-60 seconds per datapoint (depends on LLM)
- **Success Rate:** ~80-90% after refinement
- **Average Attempts:** 1.5-2.5 generations per accepted datapoint
- **Tool Coverage:** Uniform across 10+ categories
- **Blueprint Complexity:** Configurable (2-10+ steps)

---

## Troubleshooting

### Common Issues

1. **"LLM failed to generate valid JSON"**
   - Check API connectivity
   - Verify model supports structured output
   - Try simpler query or fewer tools

2. **"Blueprint validation failed"**
   - Tool doesn't exist in pool
   - Missing required arguments
   - Placeholder reference error

3. **"Quality assessment: Poor"**
   - Unrealistic tool sequence
   - Missing context or dependencies
   - Inappropriate tool selection

### Debug Mode

Enable detailed logging:
```bash
python src/generate_datapoints_bfcl.py --debug
```

This shows:
- LLM prompts and responses
- Blueprint generation attempts
- Validation details
- Execution traces

---

## Future Enhancements

1. **Parallel Generation** - Generate multiple datapoints concurrently
2. **Caching** - Cache tool schemas and LLM responses
3. **Quality Metrics** - Automated quality scoring beyond LLM review
4. **Domain Constraints** - Category-specific validation rules
5. **Error Recovery** - More sophisticated failure handling

---

## References

- **BFCL Tool Pool:** Berkeley Function Calling Leaderboard dataset
- **APIGen-MT Paper:** Multi-turn API interaction generation
- **Pydantic Models:** Data validation and serialization

---

## Contact

For questions or issues with the datapoint generation pipeline:
1. Check the detailed documentation files
2. Review the source code comments
3. Enable debug mode for detailed logs

---

**Last Updated:** 2025-01-15  
**Version:** 1.0  
**Authors:** APIGen-MT Team