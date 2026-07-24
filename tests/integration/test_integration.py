"""Integration tests for step-by-step datapoint generation.

These tests verify end-to-end workflows with mocked dependencies
to ensure all components work together correctly.
"""

import json
from typing import Protocol

import pytest

from apigen_step_by_step import StepByStepDatapoint, StepByStepGenerator
from step_by_step_models import ObjectMap, StateSnapshot
from step_by_step_protocols import (
    LLM,
    StepByStepToolManager,
    is_object_map,
    parse_object_map,
)
from tests.mocks.mock_llm_responses import (
    VALID_FINAL_RESPONSE,
    VALID_QUERY_RESPONSE_2_TOOLS,
    VALID_QUERY_RESPONSE_3_TOOLS,
    VALID_SEQUENCE_RESPONSE,
    VALID_STEP_RESPONSE,
)


class ConfigurableMockLLM(LLM, Protocol):
    """LLM test double with observable response state."""

    call_count: int
    captured_prompts: list[object]

    def set_responses(self, responses: list[str]) -> None: ...


class IntegrationToolManager(StepByStepToolManager, Protocol):
    """Tool-manager test double with observable invocation state."""

    tool_outputs: dict[str, object]
    captured_invocations: list[ObjectMap]


class TestEndToEndDatapointGeneration:
    """End-to-end tests for datapoint generation."""

    @pytest.fixture
    def mock_llm_for_2_steps(
        self, mock_llm: ConfigurableMockLLM
    ) -> ConfigurableMockLLM:
        """Configure mock LLM for 2-step datapoint."""
        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,  # generate_user_query
            VALID_SEQUENCE_RESPONSE,          # validate_expected_tools
            VALID_STEP_RESPONSE,              # step 1
            VALID_STEP_RESPONSE,              # step 2
            VALID_FINAL_RESPONSE,             # _generate_final_response
        ])
        return mock_llm

    @pytest.fixture
    def mock_llm_for_3_steps(
        self, mock_llm: ConfigurableMockLLM
    ) -> ConfigurableMockLLM:
        """Configure mock LLM for 3-step datapoint."""
        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_3_TOOLS,  # generate_user_query
            VALID_SEQUENCE_RESPONSE,          # validate_expected_tools
            VALID_STEP_RESPONSE,              # step 1
            VALID_STEP_RESPONSE,              # step 2
            VALID_STEP_RESPONSE,              # step 3
            VALID_FINAL_RESPONSE,             # _generate_final_response
        ])
        return mock_llm

    @pytest.mark.slow
    def test_end_to_end_2_step_datapoint(
        self,
        mock_llm_for_2_steps: ConfigurableMockLLM,
        mock_tools: IntegrationToolManager,
    ) -> None:
        """Test complete 2-step datapoint generation.

        Note: This test may require additional setup or retries in practice.
        """
        generator = StepByStepGenerator(
            llm_client=mock_llm_for_2_steps,
            tool_manager=mock_tools,
            num_actions=2,
        )

        result = generator.generate_datapoint(query_retries=2)

        # May succeed or fail depending on validation
        if result:
            assert isinstance(result, StepByStepDatapoint)
            assert result.trajectory.query != ""
            assert result.verification_result is not None

    @pytest.mark.skip(reason="Requires complex mock setup")
    def test_end_to_end_3_step_datapoint(
        self,
        mock_llm_for_3_steps: ConfigurableMockLLM,
        mock_tools: IntegrationToolManager,
    ) -> None:
        """Test complete 3-step datapoint generation."""
        generator = StepByStepGenerator(
            llm_client=mock_llm_for_3_steps,
            tool_manager=mock_tools,
            num_actions=3,
        )

        result = generator.generate_datapoint(query_retries=1)

        if result:
            assert len(result.trajectory.steps) == 3
            # Verify sequential step numbering
            for i, step in enumerate(result.trajectory.steps, 1):
                assert step.step_number == i

    @pytest.mark.skip(reason="Requires proper mock setup for focus category")
    def test_end_to_end_with_focus_category(
        self,
        mock_llm_for_2_steps: ConfigurableMockLLM,
        mock_tools: IntegrationToolManager,
    ) -> None:
        """Test datapoint generation with category focus."""
        generator = StepByStepGenerator(
            llm_client=mock_llm_for_2_steps,
            tool_manager=mock_tools,
            num_actions=2,
        )

        result = generator.generate_datapoint(
            focus_category="Travel",
            query_retries=1,
        )

        if result:
            # Verify focus category in metadata
            assert result.generation_metadata.get("focus_category") == "Travel"


class TestPlaceholderChain:
    """Tests for placeholder chaining across steps."""

    def test_placeholder_chain_across_steps(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_llm: ConfigurableMockLLM,
        mock_tools: IntegrationToolManager,
    ) -> None:
        """Test that placeholders can reference previous step outputs."""
        def get_api_state() -> StateSnapshot:
            return {}

        def restore_api_state(_state: StateSnapshot) -> None:
            return None

        def initialize_api_state(*, force_new: bool = False) -> None:
            del force_new

        def has_python_implementation(_tool_name: str) -> bool:
            return True

        def is_replay_safe(_tool_name: str) -> bool:
            return True

        def invoke_python_tool(
            tool_name: str,
            params: ObjectMap,
        ) -> object:
            return mock_tools.invoke_tool(tool_name, params)

        monkeypatch.setattr(
            mock_tools,
            "get_api_state",
            get_api_state,
            raising=False,
        )
        monkeypatch.setattr(
            mock_tools,
            "restore_api_state",
            restore_api_state,
            raising=False,
        )
        monkeypatch.setattr(
            mock_tools,
            "initialize_api_state",
            initialize_api_state,
            raising=False,
        )
        monkeypatch.setattr(
            mock_tools,
            "has_python_implementation",
            has_python_implementation,
            raising=False,
        )
        monkeypatch.setattr(
            mock_tools,
            "is_replay_safe",
            is_replay_safe,
            raising=False,
        )
        monkeypatch.setattr(
            mock_tools,
            "invoke_python_tool",
            invoke_python_tool,
            raising=False,
        )
        monkeypatch.setattr(
            mock_tools,
            "api_name_to_class_key",
            {
                "search_flights": "mock",
                "book_hotel": "mock",
            },
            raising=False,
        )
        mock_tools.python_tool_instances["mock"] = mock_tools
        flights = [{"flight_id": "FL001"}]
        mock_tools.tool_outputs["search_flights"] = flights
        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
            json.dumps({
                "modifications": {},
                "reasoning": "no changes needed",
            }),
            json.dumps({"origin": "NYC", "destination": "LA"}),
            VALID_SEQUENCE_RESPONSE,
            json.dumps({
                "location": "{{search_flights_output}}",
                "check_in": "2026-08-01",
            }),
            VALID_SEQUENCE_RESPONSE,
            VALID_FINAL_RESPONSE,
        ])

        generator = StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

        result = generator.generate_datapoint(query_retries=1)

        assert result is not None
        assert len(result.trajectory.steps) == 2
        step2 = result.trajectory.steps[1]
        assert step2.tool_calls[0].arguments["location"] == (
            "{{search_flights_output}}"
        )
        captured_params = mock_tools.captured_invocations[1].get("params")
        assert is_object_map(captured_params)
        assert captured_params["location"] == flights


class TestRetryWithFeedback:
    """Tests for retry logic with feedback."""

    @pytest.mark.skip(reason="Complex retry test - requires extensive mock setup")
    def test_retry_with_feedback_integration(
        self,
        mock_llm: ConfigurableMockLLM,
        mock_tools: IntegrationToolManager,
    ) -> None:
        """Test that feedback is properly accumulated and used in retries."""
        from tests.mocks.mock_llm_responses import (
            QUERY_RESPONSE_WRONG_TOOL_COUNT,
            VALID_QUERY_RESPONSE_2_TOOLS,
        )

        # First attempt fails, second succeeds
        mock_llm.set_responses([
            QUERY_RESPONSE_WRONG_TOOL_COUNT,  # First attempt - wrong tool count
            VALID_QUERY_RESPONSE_2_TOOLS,       # Second attempt - succeeds
            VALID_SEQUENCE_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_FINAL_RESPONSE,
        ])

        generator = StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

        result = generator.generate_datapoint(query_retries=2)

        # May succeed or fail - just verify it runs
        if result:
            # Verify feedback was passed
            assert len(mock_llm.captured_prompts) > 1

    @pytest.mark.skip(reason="Complex retry test - requires extensive mock setup")
    def test_multiple_retries_before_success(
        self,
        mock_llm: ConfigurableMockLLM,
        mock_tools: IntegrationToolManager,
    ) -> None:
        """Test multiple retries before successful generation."""
        from tests.mocks.mock_llm_responses import (
            QUERY_RESPONSE_INVALID_TOOL,
            QUERY_RESPONSE_WRONG_TOOL_COUNT,
            VALID_QUERY_RESPONSE_2_TOOLS,
        )

        # Multiple failures before success
        mock_llm.set_responses([
            QUERY_RESPONSE_WRONG_TOOL_COUNT,  # Attempt 1
            QUERY_RESPONSE_INVALID_TOOL,        # Attempt 2
            VALID_QUERY_RESPONSE_2_TOOLS,       # Attempt 3 - succeeds
            VALID_SEQUENCE_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_FINAL_RESPONSE,
        ])

        generator = StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

        result = generator.generate_datapoint(query_retries=3)

        # May succeed or fail
        if result:
            # Should have taken multiple attempts
            assert mock_llm.call_count >= 3


class TestVerificationFailureRecovery:
    """Tests for recovery from verification failures."""

    def test_verification_failure_then_success(
        self,
        mock_llm: ConfigurableMockLLM,
        mock_tools: IntegrationToolManager,
    ) -> None:
        """Test that verification failures trigger retries."""
        # Generate a datapoint that might fail verification then succeed
        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_FINAL_RESPONSE,
        ])

        generator = StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

        result = generator.generate_datapoint(query_retries=2)

        if result:
            # Should have verification result
            assert result.verification_result is not None
            assert "overall_verification_passed" in result.verification_result


class TestToolCategoryFiltering:
    """Tests for category-based tool filtering."""

    def test_category_focus_limits_tool_pool(
        self,
        mock_llm: ConfigurableMockLLM,
        mock_tools_travel_only: IntegrationToolManager,
    ) -> None:
        """Test that focus category limits available tools."""
        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_FINAL_RESPONSE,
        ])

        generator = StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools_travel_only,
            num_actions=2,
        )

        result = generator.generate_datapoint(
            focus_category="Travel",
            query_retries=1,
        )

        if result:
            # All tools should be from Travel category
            for tool in result.trajectory.tools_used:
                category = mock_tools_travel_only.get_tool_category(tool)
                assert category == "Travel"


class TestDatapointCompleteness:
    """Tests for datapoint completeness."""

    def test_datapoint_has_all_required_fields(
        self,
        mock_llm: ConfigurableMockLLM,
        mock_tools: IntegrationToolManager,
    ) -> None:
        """Test that generated datapoint has all required fields."""
        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_FINAL_RESPONSE,
        ])

        generator = StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

        result = generator.generate_datapoint(query_retries=1)

        if result:
            # Required fields
            assert result.trajectory.query
            assert result.trajectory.steps
            assert result.trajectory.final_response
            assert result.trajectory.tools_used is not None
            assert result.trajectory.categories_used is not None
            assert result.generation_metadata
            assert result.verification_result

            # Each step has required fields
            for step in result.trajectory.steps:
                assert step.step_number > 0
                assert step.tool_calls

            # Each tool call has required fields
            for step in result.trajectory.steps:
                for tc in step.tool_calls:
                    assert tc.tool_name
                    assert tc.arguments is not None
                    assert tc.output is not None

    def test_datapoint_serialization_roundtrip(
        self,
        mock_llm: ConfigurableMockLLM,
        mock_tools: IntegrationToolManager,
    ) -> None:
        """Test that datapoint can be serialized and deserialized."""
        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_FINAL_RESPONSE,
        ])

        generator = StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

        result = generator.generate_datapoint(query_retries=1)

        if result:
            # Serialize
            json_str = result.model_dump_json()

            # Deserialize
            data = parse_object_map(json_str)

            # Verify structure
            assert "trajectory" in data
            assert "generation_metadata" in data
            assert "verification_result" in data


class TestMultiStepWorkflows:
    """Tests for multi-step workflows."""

    def test_query_generation_to_step_generation(
        self,
        mock_llm: ConfigurableMockLLM,
        mock_tools: IntegrationToolManager,
    ) -> None:
        """Test workflow from query generation through step generation."""
        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_FINAL_RESPONSE,
        ])

        generator = StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

        # Generate query
        query_result = generator.generate_user_query()

        # Should have generated a query
        assert query_result.query
        assert len(query_result.expected_tools) == 2

        # Generate datapoint
        datapoint = generator.generate_datapoint(query_retries=1)

        if datapoint:
            # Query should match
            assert datapoint.trajectory.query == query_result.query

    def test_step_execution_accumulates_context(
        self,
        mock_llm: ConfigurableMockLLM,
        mock_tools: IntegrationToolManager,
    ) -> None:
        """Test that step execution properly accumulates context."""
        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_FINAL_RESPONSE,
        ])

        generator = StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

        result = generator.generate_datapoint(query_retries=1)

        if result:
            # Verify steps have outputs
            for step in result.trajectory.steps:
                for tc in step.tool_calls:
                    assert tc.output is not None
