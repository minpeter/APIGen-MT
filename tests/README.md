# Unit Test Suite for generate_step_by_step

This directory contains a comprehensive unit-test suite for the core logic of `generate_step_by_step.py` and the `StepByStepGenerator` class.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── mocks/                   # Mock implementations
│   ├── mock_llm_client.py   # Fine-grained LLM mocking
│   ├── mock_llm_responses.py # Pre-defined valid/invalid responses
│   └── mock_tool_manager.py  # Tool manager mocking
├── unit/                    # Unit tests
│   ├── test_models.py       # Pydantic model validation tests (31 tests)
│   ├── test_placeholder_processing.py # Placeholder resolution tests (25 tests)
│   ├── test_verification.py # Verification logic tests (23 tests)
│   ├── test_step_generation.py # Step generation tests (16 tests)
│   ├── test_query_generation.py # Query generation tests (14 tests)
│   ├── test_datapoint_generation.py # Datapoint generation tests (18 tests)
│   └── test_generate_step_by_step.py # CLI/main tests (24 tests)
└── integration/             # Integration tests
    └── test_integration.py  # End-to-end tests (12 tests)

Total: 163 passing tests + 9 skipped complex tests
```

## Running Tests

### Run all tests
```bash
cd /home/ishalyminov/data/APIGen-MT
PYTHONPATH=src:tests python -m pytest tests/ -v
```

### Run only unit tests
```bash
PYTHONPATH=src:tests python -m pytest tests/unit -v
```

### Run specific test file
```bash
PYTHONPATH=src:tests python -m pytest tests/unit/test_placeholder_processing.py -v
```

### Run with coverage
```bash
PYTHONPATH=src:tests python -m pytest tests/ --cov=src --cov-report=term-missing
```

## Key Features

### 1. Fine-Grained Mock LLM
The `MockLLMClient` provides:
- Sequence-based responses for controlled testing
- Pattern-based responses for conditional logic
- Exception injection for error handling tests
- Usage tracking for verification

### 2. Comprehensive Mock Responses
`mock_llm_responses.py` includes:
- Valid JSON responses for all prompt types
- Invalid/malformed JSON responses
- Missing field responses
- Empty responses
- Edge cases (unicode, special characters)

### 3. Mock Tool Manager
The `MockToolManager` provides:
- 8 default tool schemas (Travel, Food, Communication, etc.)
- Canned outputs for each tool
- Configurable failure modes
- Invocation tracking

## Test Categories

### Model Tests (31 tests)
- ToolCallWithOutput creation and defaults
- TrajectoryStep with multiple tool calls
- ConversationTrajectory with all fields
- StepByStepDatapoint with metadata
- VerificationResult for pass/fail cases
- StepSelectionResult and QueryGenerationResult

### Placeholder Tests (25 tests)
- Simple placeholder resolution: `{{key}}`
- Nested resolution: `{{tool.output.key}}`
- Deeply nested: `{{a.b.c.d}}`
- Partial string replacement
- Unresolvable placeholder handling
- Non-string argument preservation
- Special characters and unicode

### Verification Tests (23 tests)
- Tool relevance checking with keyword overlap
- Invocation order validation
- Output type consistency (string, dict, list, number, boolean)
- Placeholder resolution verification
- Full verification orchestration

### Step Generation Tests (16 tests)
- Successful step selection
- Invalid JSON handling
- Empty response handling
- Trajectory context passing
- Tool execution simulation
- Final response generation

### Query Generation Tests (14 tests)
- Successful query generation
- Wrong tool count handling
- Invalid tool validation
- JSON decode error recovery
- Max retries exhaustion
- Focus category filtering

### Datapoint Generation Tests (18 tests)
- Successful datapoint generation
- Metadata population
- 3-step datapoint support
- Tools used tracking
- Categories tracking
- Verification result attachment

### CLI Tests (24 tests)
- Argument parsing with defaults
- Custom argument values
- Tool category loading
- Environment variable checking
- Discard statistics tracking

## Pre-Commit Hook

To ensure tests pass before committing, add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
cd /home/ishalyminov/data/APIGen-MT
PYTHONPATH=src:tests python -m pytest tests/ -v --tb=line -q
exit $?
```

## Coverage Targets

The test suite aims for:
- **≥90%** line coverage for core logic
- **100%** coverage for placeholder processing
- **100%** coverage for model validation

## Mock Response Examples

### Valid Query Response
```json
{
  "query": "Find flights from NYC to LA",
  "intent": "Travel planning",
  "expected_tools": ["search_flights", "book_hotel"]
}
```

### Invalid Response (Wrong Count)
```json
{
  "query": "Find flights",
  "intent": "Search",
  "expected_tools": ["search_flights"]  // Only 1, expected 2
}
```

### Malformed Response
```json
{"query": "test", "expected_tools": ["t1"  // Unclosed
```

## Continuous Integration

For CI/CD pipelines, use:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    cd /home/ishalyminov/data/APIGen-MT
    PYTHONPATH=src:tests python -m pytest tests/ -v --tb=line -q
```

## Notes

- All tests use mocked LLM responses to avoid actual API calls
- The mock implementations are fine-grained for detailed control
- Complex integration tests are skipped due to mock setup requirements
- The suite covers all major code paths including error handling
