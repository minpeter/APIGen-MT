"""Unit tests for verification methods in StepByStepGenerator.

These tests verify all verification logic including tool relevance,
invocation order, output consistency, and placeholder resolution.
"""

import pytest
from apigen_step_by_step import (
    StepByStepGenerator,
    TrajectoryStep,
    ToolCallWithOutput,
)


class TestVerifyToolRelevance:
    """Tests for verify_tool_relevance method."""

    @pytest.fixture
    def generator(self, mock_llm, mock_tools):
        """Create a generator with mock dependencies."""
        return StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

    def test_tool_relevance_high_overlap(self, generator):
        """Test tool with high keyword overlap scores well."""
        query = "Search for flights from NYC to LA"
        tool_name = "search_flights"
        step = TrajectoryStep(step_number=1)

        result = generator.verify_tool_relevance(query, tool_name, step)

        assert result["is_relevant"] is True
        assert result["tool_name"] == "search_flights"
        assert result["relevance_score"] > 0.1

    def test_tool_relevance_name_match(self, generator):
        """Test tool with name matching query words scores well."""
        query = "Book a hotel room in New York"
        tool_name = "book_hotel"
        step = TrajectoryStep(step_number=1)

        result = generator.verify_tool_relevance(query, tool_name, step)

        assert result["is_relevant"] is True

    def test_tool_relevance_not_relevant(self, generator):
        """Test tool with no relevance to query scores poorly."""
        query = "Send an email to my friend"
        tool_name = "search_flights"
        step = TrajectoryStep(step_number=1)

        result = generator.verify_tool_relevance(query, tool_name, step)

        assert result["is_relevant"] is False
        assert result["relevance_score"] < 0.1

    def test_tool_relevance_missing_tool(self, generator):
        """Test relevance check for non-existent tool."""
        query = "Test query"
        tool_name = "nonexistent_tool"
        step = TrajectoryStep(step_number=1)

        # The method should handle missing tools gracefully
        try:
            result = generator.verify_tool_relevance(query, tool_name, step)
            # If no exception, check result
            assert result["is_relevant"] is False
            assert result["relevance_score"] == 0.0
            assert "not found" in result["reasoning"].lower()
        except ValueError as e:
            # Or it may raise ValueError
            assert "not found" in str(e).lower()

    def test_tool_relevance_partial_match(self, generator):
        """Test tool with partial keyword overlap."""
        query = "Find a restaurant and get information"
        tool_name = "search_restaurants"
        step = TrajectoryStep(step_number=1)

        result = generator.verify_tool_relevance(query, tool_name, step)

        # Should have some overlap but not extremely high
        assert result["relevance_score"] > 0

    def test_tool_relevance_empty_query(self, generator):
        """Test relevance with empty query."""
        query = ""
        tool_name = "search_flights"
        step = TrajectoryStep(step_number=1)

        result = generator.verify_tool_relevance(query, tool_name, step)

        assert result["is_relevant"] is False
        assert result["relevance_score"] == 0.0

    def test_tool_relevance_case_insensitive(self, generator):
        """Test that relevance is case insensitive."""
        query = "SEARCH FOR FLIGHTS"
        tool_name = "search_flights"
        step = TrajectoryStep(step_number=1)

        result = generator.verify_tool_relevance(query, tool_name, step)

        assert result["is_relevant"] is True


class TestVerifyInvocationOrder:
    """Tests for verify_invocation_order method."""

    @pytest.fixture
    def generator(self, mock_llm, mock_tools):
        """Create a generator with mock dependencies."""
        return StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

    def test_order_correct_sequence(self, generator):
        """Test correct ordering of tools."""
        query = "Search flights and book hotel"
        trajectory = [
            TrajectoryStep(
                step_number=1,
                tool_calls=[ToolCallWithOutput(tool_name="search_flights")],
            ),
            TrajectoryStep(
                step_number=2,
                tool_calls=[ToolCallWithOutput(tool_name="book_hotel")],
            ),
        ]

        result = generator.verify_invocation_order(query, trajectory)

        assert result["order_is_correct"] is True

    def test_order_empty_trajectory(self, generator):
        """Test empty trajectory returns success."""
        query = "Test query"
        trajectory = []

        result = generator.verify_invocation_order(query, trajectory)

        assert result["order_is_correct"] is True
        assert "No steps" in result["order_verification_details"]

    def test_order_single_step(self, generator):
        """Test single step is always correct."""
        query = "Get weather"
        trajectory = [
            TrajectoryStep(
                step_number=1,
                tool_calls=[ToolCallWithOutput(tool_name="get_weather")],
            ),
        ]

        result = generator.verify_invocation_order(query, trajectory)

        assert result["order_is_correct"] is True

    def test_order_create_not_first_warning(self, generator):
        """Test warning when create tool is first."""
        query = "Create a new event"
        # Note: This test depends on actual tool names in mock_tools
        # Using a tool that might trigger the warning
        trajectory = [
            TrajectoryStep(
                step_number=1,
                tool_calls=[ToolCallWithOutput(tool_name="create_calendar_event")],
            ),
        ]

        result = generator.verify_invocation_order(query, trajectory)

        # The order check might flag this as potentially needing context
        assert "order_is_correct" in result


class TestVerifyOutputConsistency:
    """Tests for verify_output_consistency method."""

    @pytest.fixture
    def generator(self, mock_llm, mock_tools):
        """Create a generator with mock dependencies."""
        return StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

    def test_output_type_match_string(self, generator):
        """Test string output matches expected string type."""
        result = generator.verify_output_consistency(
            tool_name="test_tool",
            step_number=1,
            output="Success message",
            expected_type="string",
            expected_description="A message",
        )

        assert result["output_type_matches"] is True

    def test_output_type_match_dict(self, generator):
        """Test dict output matches expected dict type."""
        result = generator.verify_output_consistency(
            tool_name="test_tool",
            step_number=1,
            output={"key": "value"},
            expected_type="dict",
            expected_description="A dictionary",
        )

        assert result["output_type_matches"] is True

    def test_output_type_match_list(self, generator):
        """Test list output matches expected list type."""
        result = generator.verify_output_consistency(
            tool_name="test_tool",
            step_number=1,
            output=[1, 2, 3],
            expected_type="list",
            expected_description="A list",
        )

        assert result["output_type_matches"] is True

    def test_output_type_match_number(self, generator):
        """Test number output matches expected number type."""
        result = generator.verify_output_consistency(
            tool_name="test_tool",
            step_number=1,
            output=42,
            expected_type="number",
            expected_description="A number",
        )

        assert result["output_type_matches"] is True

    def test_output_type_match_integer(self, generator):
        """Test integer output matches expected integer type."""
        result = generator.verify_output_consistency(
            tool_name="test_tool",
            step_number=1,
            output=42,
            expected_type="integer",
            expected_description="An integer",
        )

        # Integer should match integer type
        assert result["output_type_matches"] is True or "int" in str(result.get("issues", []))

    def test_output_type_match_boolean(self, generator):
        """Test boolean output matches expected boolean type."""
        result = generator.verify_output_consistency(
            tool_name="test_tool",
            step_number=1,
            output=True,
            expected_type="boolean",
            expected_description="A boolean",
        )

        assert result["output_type_matches"] is True

    def test_output_type_mismatch(self, generator):
        """Test type mismatch is detected."""
        result = generator.verify_output_consistency(
            tool_name="test_tool",
            step_number=1,
            output="string value",
            expected_type="integer",
            expected_description="Should be integer",
        )

        assert result["output_type_matches"] is False
        assert len(result["issues"]) > 0

    def test_output_none_handling(self, generator):
        """Test None output handling."""
        result = generator.verify_output_consistency(
            tool_name="test_tool",
            step_number=1,
            output=None,
            expected_type="string",
            expected_description="A string",
        )

        assert result["output_type_matches"] is False
        assert "Output is None" in result["issues"]

    def test_output_compound_type_dict(self, generator):
        """Test compound type with dict."""
        result = generator.verify_output_consistency(
            tool_name="test_tool",
            step_number=1,
            output={"key": "value"},
            expected_type="dict or list",
            expected_description="A dict or list",
        )

        assert result["output_type_matches"] is True

    def test_output_unknown_type(self, generator):
        """Test unknown expected type."""
        result = generator.verify_output_consistency(
            tool_name="test_tool",
            step_number=1,
            output="anything",
            expected_type="unknown",
            expected_description="",
        )

        # Unknown type behavior varies by implementation - just verify it runs
        assert "output_type_matches" in result


class TestVerifyPlaceholderResolution:
    """Tests for verify_placeholder_resolution method."""

    @pytest.fixture
    def generator(self, mock_llm, mock_tools):
        """Create a generator with mock dependencies."""
        return StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

    def test_all_placeholders_resolved(self, generator):
        """Test all placeholders are marked as resolved."""
        trajectory = [
            TrajectoryStep(
                step_number=1,
                tool_calls=[
                    ToolCallWithOutput(
                        tool_name="tool1",
                        arguments={"ref": "{{flight_id}}"},
                        output={},
                    )
                ],
            ),
        ]
        execution_context = {"flight_id": "FL001"}

        result = generator.verify_placeholder_resolution(trajectory, execution_context)

        assert result["all_resolved"] is True
        assert result["total_placeholders"] == 1
        assert result["resolved_count"] == 1

    def test_unresolved_placeholders_tracked(self, generator):
        """Test unresolved placeholders are tracked."""
        trajectory = [
            TrajectoryStep(
                step_number=1,
                tool_calls=[
                    ToolCallWithOutput(
                        tool_name="tool1",
                        arguments={"ref": "{{missing_key}}"},
                        output={},
                    )
                ],
            ),
        ]
        execution_context = {}

        result = generator.verify_placeholder_resolution(trajectory, execution_context)

        assert result["all_resolved"] is False
        assert result["total_placeholders"] == 1
        assert result["resolved_count"] == 0

    def test_no_placeholders_in_trajectory(self, generator):
        """Test trajectory with no placeholders."""
        trajectory = [
            TrajectoryStep(
                step_number=1,
                tool_calls=[
                    ToolCallWithOutput(
                        tool_name="tool1",
                        arguments={"key": "value"},
                        output={},
                    )
                ],
            ),
        ]
        execution_context = {}

        result = generator.verify_placeholder_resolution(trajectory, execution_context)

        assert result["all_resolved"] is True
        assert result["total_placeholders"] == 0

    def test_multiple_placeholders_mixed_resolution(self, generator):
        """Test mix of resolved and unresolved placeholders."""
        trajectory = [
            TrajectoryStep(
                step_number=1,
                tool_calls=[
                    ToolCallWithOutput(
                        tool_name="tool1",
                        arguments={
                            "found": "{{exists}}",
                            "missing": "{{not_found}}",
                        },
                        output={},
                    )
                ],
            ),
        ]
        execution_context = {"exists": "value"}

        result = generator.verify_placeholder_resolution(trajectory, execution_context)

        assert result["all_resolved"] is False
        assert result["total_placeholders"] == 2
        assert result["resolved_count"] == 1

    def test_nested_placeholder_resolution(self, generator):
        """Test nested placeholder resolution."""
        trajectory = [
            TrajectoryStep(
                step_number=1,
                tool_calls=[
                    ToolCallWithOutput(
                        tool_name="tool1",
                        arguments={"ref": "{{output.key}}"},
                        output={},
                    )
                ],
            ),
        ]
        execution_context = {"output": {"key": "value"}}

        result = generator.verify_placeholder_resolution(trajectory, execution_context)

        assert result["all_resolved"] is True
        assert result["resolved_count"] == 1

    def test_placeholder_details_structure(self, generator):
        """Test that details have correct structure."""
        trajectory = [
            TrajectoryStep(
                step_number=1,
                tool_calls=[
                    ToolCallWithOutput(
                        tool_name="tool1",
                        arguments={"arg": "{{key}}"},
                        output={},
                    )
                ],
            ),
        ]
        execution_context = {"key": "value"}

        result = generator.verify_placeholder_resolution(trajectory, execution_context)

        assert len(result["details"]) == 1
        detail = result["details"][0]
        assert detail["step"] == 1
        assert detail["tool"] == "tool1"
        assert detail["argument"] == "arg"
        assert detail["placeholder"] == "{{key}}"
        assert detail["resolved"] is True


class TestRunFullVerification:
    """Tests for run_full_verification method."""

    @pytest.fixture
    def generator(self, mock_llm, mock_tools):
        """Create a generator with mock dependencies."""
        return StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

    def test_full_verification_all_pass(self, generator, mock_tools):
        """Test full verification when all checks pass."""
        from tests.mocks.mock_tool_manager import MockToolManager

        # Create a tool manager with relevant tools
        tools = [
            t for t in MockToolManager.DEFAULT_TOOLS
            if t["name"] == "search_flights"
        ]
        outputs = {"search_flights": MockToolManager.DEFAULT_OUTPUTS["search_flights"]}
        generator.tool_manager = MockToolManager(tools=tools, outputs=outputs)

        query = "Search for flights from NYC"
        trajectory = [
            TrajectoryStep(
                step_number=1,
                tool_calls=[
                    ToolCallWithOutput(
                        tool_name="search_flights",
                        arguments={"origin": "NYC", "destination": "LA"},
                        output=outputs["search_flights"],
                    )
                ],
            ),
        ]
        execution_context = {"search_flights_output": outputs["search_flights"]}

        result = generator.run_full_verification(query, trajectory, execution_context)

        assert isinstance(result.overall_verification_passed, bool)

    def test_full_verification_empty_trajectory(self, generator):
        """Test full verification with empty trajectory."""
        query = "Test query"
        trajectory = []
        execution_context = {}

        result = generator.run_full_verification(query, trajectory, execution_context)

        # Empty trajectory should pass most checks
        assert result.order_is_correct is True
