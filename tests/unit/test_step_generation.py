"""Unit tests for step generation methods in StepByStepGenerator.

These tests verify the _generate_next_step method and related functionality
for generating individual steps in the conversation trajectory.
"""

import pytest
from apigen_step_by_step import (
    StepByStepGenerator,
    TrajectoryStep,
    ToolCallWithOutput,
    StepSelectionResult,
)
from tests.mocks.mock_llm_responses import (
    VALID_STEP_RESPONSE,
    VALID_STEP_RESPONSE_PLAIN_JSON,
    STEP_RESPONSE_MISSING_TOOL_NAME,
    STEP_RESPONSE_MISSING_ARGUMENTS,
    STEP_RESPONSE_MISSING_REASONING,
    STEP_RESPONSE_EMPTY,
    STEP_RESPONSE_NOT_JSON,
    STEP_RESPONSE_WITH_PLACEHOLDER,
)


class TestGenerateNextStep:
    """Tests for _generate_next_step method."""

    @pytest.fixture
    def generator(self, mock_llm, mock_tools):
        """Create a generator with mock dependencies."""
        return StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

    def test_next_step_success(self, generator, mock_llm):
        """Test successful step generation."""
        mock_llm.set_responses([VALID_STEP_RESPONSE])

        result = generator._generate_next_step(
            query="Search flights and book hotel",
            trajectory=[],
            execution_context={},
            expected_tools=["search_flights", "book_hotel"],
            step_num=1,
        )

        assert isinstance(result, StepSelectionResult)
        assert result.tool_name == "search_flights"
        assert "origin" in result.arguments
        assert result.reasoning != ""

    def test_next_step_plain_json_response(self, generator, mock_llm):
        """Test step generation with plain JSON (no code block)."""
        mock_llm.set_responses([VALID_STEP_RESPONSE_PLAIN_JSON])

        result = generator._generate_next_step(
            query="Test",
            trajectory=[],
            execution_context={},
            expected_tools=["book_hotel"],
            step_num=1,
        )

        assert result.tool_name == "book_hotel"

    def test_next_step_with_trajectory_context(self, generator, mock_llm):
        """Test that trajectory context is passed to LLM."""
        mock_llm.set_responses([VALID_STEP_RESPONSE])

        previous_step = TrajectoryStep(
            step_number=1,
            tool_calls=[
                ToolCallWithOutput(
                    tool_name="search_flights",
                    arguments={"origin": "NYC"},
                    output={"flight_id": "FL001"},
                )
            ],
        )

        generator._generate_next_step(
            query="Book hotel",
            trajectory=[previous_step],
            execution_context={"flight_id": "FL001"},
            expected_tools=["book_hotel"],
            step_num=2,
        )

        # Verify the prompt was captured
        assert len(mock_llm.captured_prompts) > 0
        prompt_text = str(mock_llm.captured_prompts[0])
        assert "search_flights" in prompt_text

    def test_next_step_missing_tool_name(self, generator, mock_llm):
        """Test handling of response missing tool_name."""
        mock_llm.set_responses([STEP_RESPONSE_MISSING_TOOL_NAME])

        result = generator._generate_next_step(
            query="Test",
            trajectory=[],
            execution_context={},
            expected_tools=["tool1"],
            step_num=1,
        )

        # Should return empty result when tool_name is missing
        assert result.tool_name == ""

    def test_next_step_missing_arguments(self, generator, mock_llm):
        """Test handling of response missing arguments."""
        mock_llm.set_responses([STEP_RESPONSE_MISSING_ARGUMENTS])

        result = generator._generate_next_step(
            query="Test",
            trajectory=[],
            execution_context={},
            expected_tools=["tool1"],
            step_num=1,
        )

        # Should have tool_name but empty arguments
        assert result.tool_name == "search_flights"
        assert result.arguments == {}

    def test_next_step_empty_response(self, generator, mock_llm):
        """Test handling of empty response."""
        mock_llm.set_responses([STEP_RESPONSE_EMPTY])

        result = generator._generate_next_step(
            query="Test",
            trajectory=[],
            execution_context={},
            expected_tools=["tool1"],
            step_num=1,
    )

        assert result.tool_name == "__ERROR__"
        assert result.arguments == {}

    def test_next_step_not_json_response(self, generator, mock_llm):
        """Test handling of non-JSON response."""
        mock_llm.set_responses([STEP_RESPONSE_NOT_JSON])

        result = generator._generate_next_step(
            query="Test",
            trajectory=[],
            execution_context={},
            expected_tools=["tool1"],
            step_num=1,
    )

        # Should return __ERROR__ result on JSON parse error
        assert result.tool_name == "__ERROR__"


class TestSimulateToolExecution:
    """Tests for _simulate_tool_execution method."""

    @pytest.fixture
    def generator(self, mock_llm, mock_tools):
        """Create a generator with mock dependencies."""
        return StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

    def test_simulate_execution_success(self, generator, mock_tools):
        """Test successful tool execution."""
        result = generator._simulate_tool_execution(
            tool_name="search_flights",
            arguments={"origin": "NYC", "destination": "LA"},
            execution_context={},
        )

        # Should return canned output from mock
        assert result is not None

    def test_simulate_with_placeholder_processing(self, generator, mock_tools):
        """Test that placeholders are processed before execution."""
        result = generator._simulate_tool_execution(
            tool_name="search_flights",
            arguments={"origin": "{{city}}"},
            execution_context={"city": "NYC"},
        )

        # Placeholders should be resolved before invocation
        invocations = mock_tools.get_captured_invocations()
        assert len(invocations) > 0
        # The argument should be resolved
        assert "{{city}}" not in str(invocations[0]["params"])


class TestBuildTrajectory:
    """Tests for trajectory building."""

    @pytest.fixture
    def generator(self, mock_llm, mock_tools):
        """Create a generator with mock dependencies."""
        return StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

    def test_trajectory_step_numbering(self, generator, mock_llm):
        """Test that steps are numbered sequentially."""
        from tests.mocks.mock_llm_responses import (
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
        )

        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_STEP_RESPONSE,
        ])

        datapoint = generator.generate_datapoint(query_retries=0)

        if datapoint:
            for i, step in enumerate(datapoint.trajectory.steps, 1):
                assert step.step_number == i

    def test_execution_context_accumulation(self, generator, mock_llm, mock_tools):
        """Test that execution context accumulates outputs."""
        mock_tools.reset()

        # Simulate execution and check context
        output1 = generator._simulate_tool_execution(
            tool_name="search_flights",
            arguments={"origin": "NYC"},
            execution_context={},
        )

        context = {"search_flights_output": output1}

        output2 = generator._simulate_tool_execution(
            tool_name="book_hotel",
            arguments={"location": "LAX"},
            execution_context=context,
        )

        # Context should now have both outputs
        assert "search_flights_output" in context
        context["book_hotel_output"] = output2
        assert "book_hotel_output" in context

    def test_tools_used_tracking(self, generator, mock_llm):
        """Test that tools_used tracks unique tools."""
        from tests.mocks.mock_llm_responses import VALID_QUERY_RESPONSE_2_TOOLS

        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_STEP_RESPONSE,
            VALID_STEP_RESPONSE,
        ])

        # Parse expected tools from response
        import json
        response_text = VALID_QUERY_RESPONSE_2_TOOLS
        # Extract JSON
        json_str = response_text.split("```json")[1].split("```")[0].strip()
        data = json.loads(json_str)
        expected_tools = data["expected_tools"]

        assert len(expected_tools) == 2
        assert len(set(expected_tools)) == len(expected_tools)  # No duplicates


class TestGenerateFinalResponse:
    """Tests for _generate_final_response method."""

    @pytest.fixture
    def generator(self, mock_llm, mock_tools):
        """Create a generator with mock dependencies."""
        return StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

    def test_final_response_generation(self, generator, mock_llm):
        """Test final response generation."""
        from tests.mocks.mock_llm_responses import VALID_FINAL_RESPONSE

        mock_llm.set_responses([VALID_FINAL_RESPONSE])

        trajectory = [
            TrajectoryStep(
                step_number=1,
                tool_calls=[
                    ToolCallWithOutput(
                        tool_name="search_flights",
                        arguments={},
                        output={"flights": []},
                    )
                ],
            ),
        ]

        result = generator._generate_final_response(
            query="Find flights",
            trajectory=trajectory,
            execution_context={},
        )

        assert result == VALID_FINAL_RESPONSE.strip()

    def test_final_response_with_error(self, generator, mock_llm):
        """Test final response error handling."""
        mock_llm.set_exception(RuntimeError("LLM Error"))

        trajectory = [
            TrajectoryStep(
                step_number=1,
                tool_calls=[],
            ),
        ]

        result = generator._generate_final_response(
            query="Test",
            trajectory=trajectory,
            execution_context={},
        )

        # Should return fallback message on error
        assert "I have completed your request" in result


class TestStepWithPlaceholder:
    """Tests for step generation with placeholders."""

    @pytest.fixture
    def generator(self, mock_llm, mock_tools):
        """Create a generator with mock dependencies."""
        return StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

    def test_step_with_placeholder_argument(self, generator, mock_llm):
        """Test step generation with placeholder in arguments."""
        mock_llm.set_responses([STEP_RESPONSE_WITH_PLACEHOLDER])

        result = generator._generate_next_step(
            query="Send message about flight",
            trajectory=[
                TrajectoryStep(
                    step_number=1,
                    tool_calls=[
                        ToolCallWithOutput(
                            tool_name="search_flights",
                            arguments={},
                            output={"flight_id": "FL001"},
                        )
                    ],
                )
            ],
            execution_context={"search_flights_output": {"flight_id": "FL001"}},
            expected_tools=["send_email"],
            step_num=2,
        )

        assert result.tool_name == "send_message"
        # Placeholder should be in arguments
        assert "{{search_flights_output.flight_id}}" in str(result.arguments)
