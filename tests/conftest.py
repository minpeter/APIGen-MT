"""Shared fixtures for the test suite.

This module provides pytest fixtures that can be used across all test modules.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from tests.mocks.mock_llm_client import MockLLMClient, MockLLMClientBuilder
from tests.mocks.mock_llm_responses import (
    VALID_QUERY_RESPONSE_2_TOOLS,
    VALID_SEQUENCE_RESPONSE,
    VALID_STEP_RESPONSE,
    VALID_FINAL_RESPONSE,
    TOOL_SIMULATION_VALID_DICT,
)
from tests.mocks.mock_tool_manager import MockToolManager, MockToolManagerBuilder


# =============================================================================
# Mock Client Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Fixture to create a fresh mock LLM client.

    Returns:
        MockLLMClient instance with no preconfigured responses
    """
    return MockLLMClient()


@pytest.fixture
def mock_llm_with_responses():
    """Fixture to create a mock LLM client with typical responses.

    Returns:
        MockLLMClient with responses for successful 2-step datapoint generation
    """
    client = MockLLMClient()
    client.set_responses([
        VALID_QUERY_RESPONSE_2_TOOLS,  # generate_user_query
        VALID_SEQUENCE_RESPONSE,          # validate_expected_tools
        VALID_STEP_RESPONSE,              # step 1
        VALID_STEP_RESPONSE,              # step 2
        VALID_FINAL_RESPONSE,             # generate final response
    ])
    return client


@pytest.fixture
def mock_llm_builder():
    """Fixture to provide a mock LLM client builder.

    Returns:
        MockLLMClientBuilder instance
    """
    return MockLLMClientBuilder()


@pytest.fixture
def mock_tools():
    """Fixture to create a fresh mock tool manager.

    Returns:
        MockToolManager instance with default tools
    """
    return MockToolManager()


@pytest.fixture
def mock_tools_minimal():
    """Fixture to create a mock tool manager with minimal tools.

    Returns:
        MockToolManager with only search_flights and book_hotel
    """
    tools = [
        t for t in MockToolManager.DEFAULT_TOOLS
        if t["name"] in ["search_flights", "book_hotel"]
    ]
    outputs = {
        k: v for k, v in MockToolManager.DEFAULT_OUTPUTS.items()
        if k in ["search_flights", "book_hotel"]
    }
    return MockToolManager(tools=tools, outputs=outputs)


@pytest.fixture
def mock_tools_travel_only():
    """Fixture to create a mock tool manager with only travel tools.

    Returns:
        MockToolManager with Travel category tools only
    """
    return MockToolManagerBuilder().with_travel_tools().build()


# =============================================================================
# Generator Fixtures
# =============================================================================


@pytest.fixture
def generator_2_steps(mock_llm_with_responses, mock_tools):
    """Fixture to create a StepByStepGenerator configured for 2 steps.

    Args:
        mock_llm_with_responses: Mock LLM client fixture
        mock_tools: Mock tool manager fixture

    Returns:
        StepByStepGenerator instance
    """
    from apigen_step_by_step import StepByStepGenerator
    return StepByStepGenerator(
        llm_client=mock_llm_with_responses,
        tool_manager=mock_tools,
        num_actions=2,
    )


@pytest.fixture
def generator_3_steps(mock_llm, mock_tools):
    """Fixture to create a StepByStepGenerator configured for 3 steps.

    Args:
        mock_llm: Mock LLM client fixture
        mock_tools: Mock tool manager fixture

    Returns:
        StepByStepGenerator instance
    """
    from apigen_step_by_step import StepByStepGenerator
    return StepByStepGenerator(
        llm_client=mock_llm,
        tool_manager=mock_tools,
        num_actions=3,
    )


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_tool_schemas():
    """Fixture providing sample tool schemas.

    Returns:
        List of tool schema dictionaries
    """
    return [
        {
            "name": "search_flights",
            "description": "Search for flights",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                },
                "required": ["origin", "destination"],
            },
            "output_type": "list",
            "output_description": "List of flights",
            "category": "Travel",
        },
        {
            "name": "book_hotel",
            "description": "Book a hotel",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "check_in": {"type": "string"},
                },
                "required": ["location"],
            },
            "output_type": "dict",
            "output_description": "Booking confirmation",
            "category": "Travel",
        },
    ]


@pytest.fixture
def sample_execution_context():
    """Fixture providing a sample execution context.

    Returns:
        Dictionary simulating accumulated execution context
    """
    return {
        "search_flights_output": {
            "flight_id": "FL001",
            "price": 299,
            "airline": "TestAir",
        },
        "book_hotel_output": {
            "confirmation_id": "HT123",
            "status": "confirmed",
        },
    }


@pytest.fixture
def sample_trajectory():
    """Fixture providing a sample trajectory with one step.

    Returns:
        List of TrajectoryStep objects
    """
    from apigen_step_by_step import TrajectoryStep, ToolCallWithOutput
    return [
        TrajectoryStep(
            step_number=1,
            tool_calls=[
                ToolCallWithOutput(
                    tool_name="search_flights",
                    arguments={"origin": "NYC", "destination": "LA"},
                    output={"flight_id": "FL001", "price": 299},
                )
            ],
            reasoning="First step: search for flights",
        )
    ]


# =============================================================================
# Markers and Configuration
# =============================================================================


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast)")
    config.addinivalue_line("markers", "integration: Integration tests (slower)")
    config.addinivalue_line("markers", "slow: Slow tests that may be skipped")


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture providing the path to test data directory.

    Returns:
        Path object pointing to tests/data
    """
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir
