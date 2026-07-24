"""Shared fixtures for the test suite (layer-safe).

Tools-only stack does not ship tests/mocks or generators. Mocks are
imported only when available so `tests/tools` collects cleanly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Unit tests (fast)")
    config.addinivalue_line("markers", "integration: Integration tests (slower)")
    config.addinivalue_line("markers", "slow: Slow tests that may be skipped")


@pytest.fixture(scope="session")
def test_data_dir():
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


# Optional mocks (pipeline layer)
try:
    from tests.mocks.mock_llm_client import MockLLMClient, MockLLMClientBuilder
    from tests.mocks.mock_llm_responses import (
        VALID_QUERY_RESPONSE_2_TOOLS,
        VALID_SEQUENCE_RESPONSE,
        VALID_STEP_RESPONSE,
        VALID_FINAL_RESPONSE,
    )
    from tests.mocks.mock_tool_manager import MockToolManager, MockToolManagerBuilder
    _HAS_MOCKS = True
except ImportError:  # pragma: no cover - tools-only layer
    _HAS_MOCKS = False


if _HAS_MOCKS:

    @pytest.fixture
    def mock_llm():
        return MockLLMClient()

    @pytest.fixture
    def mock_llm_with_responses():
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
    def mock_llm_builder():
        return MockLLMClientBuilder()

    @pytest.fixture
    def mock_tools():
        return MockToolManager()
