"""Shared layer-safe fixtures for the test suite.

The tools-only stack does not ship pipeline mocks, so those fixtures are
registered only when their implementations are importable.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register markers shared across test layers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast)")
    config.addinivalue_line("markers", "integration: Integration tests (slower)")
    config.addinivalue_line("markers", "slow: Slow tests that may be skipped")


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the directory used for generated test data."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


try:
    from tests.mocks.mock_llm_client import (
        MockLLMClient,
        MockLLMClientBuilder,
    )
    from tests.mocks.mock_llm_responses import (
        VALID_FINAL_RESPONSE,
        VALID_QUERY_RESPONSE_2_TOOLS,
        VALID_SEQUENCE_RESPONSE,
        VALID_STEP_RESPONSE,
    )
    from tests.mocks.mock_tool_manager import (
        MockToolManager,
        MockToolManagerBuilder,
    )

    @pytest.fixture
    def mock_llm() -> MockLLMClient:
        """Return an unconfigured mock LLM client."""
        return MockLLMClient()

    @pytest.fixture
    def mock_llm_with_responses() -> MockLLMClient:
        """Return a mock LLM configured for a successful two-step flow."""
        client = MockLLMClient()
        client.set_responses(
            [
                VALID_QUERY_RESPONSE_2_TOOLS,
                VALID_SEQUENCE_RESPONSE,
                VALID_STEP_RESPONSE,
                VALID_STEP_RESPONSE,
                VALID_FINAL_RESPONSE,
            ]
        )
        return client

    @pytest.fixture
    def mock_llm_builder() -> MockLLMClientBuilder:
        """Return the mock LLM builder."""
        return MockLLMClientBuilder()

    @pytest.fixture
    def mock_tools() -> MockToolManager:
        """Return the complete mock tool catalog."""
        return MockToolManager()

    @pytest.fixture
    def mock_tools_travel_only() -> MockToolManager:
        """Return a mock tool catalog restricted to travel tools."""
        return MockToolManagerBuilder().with_travel_tools().build()
except ImportError:
    pass
