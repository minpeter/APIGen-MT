"""Unit tests for Pydantic models in apigen_step_by_step.

These tests verify the creation, validation, and serialization of
all Pydantic models used in the step-by-step datapoint generation.
"""

import pytest
from apigen_step_by_step import (
    ToolCallWithOutput,
    TrajectoryStep,
    ConversationTrajectory,
    StepByStepDatapoint,
    VerificationResult,
    StepSelectionResult,
    QueryGenerationResult,
)


class TestToolCallWithOutput:
    """Tests for ToolCallWithOutput model."""

    def test_creation_with_all_fields(self):
        """Test creating ToolCallWithOutput with all fields."""
        tool_call = ToolCallWithOutput(
            tool_name="search_flights",
            arguments={"origin": "NYC", "destination": "LA"},
            output={"flight_id": "FL001", "price": 299},
        )
        assert tool_call.tool_name == "search_flights"
        assert tool_call.arguments == {"origin": "NYC", "destination": "LA"}
        assert tool_call.output == {"flight_id": "FL001", "price": 299}

    def test_creation_with_defaults(self):
        """Test creating ToolCallWithOutput with defaults."""
        tool_call = ToolCallWithOutput(tool_name="simple_tool")
        assert tool_call.tool_name == "simple_tool"
        assert tool_call.arguments == {}
        assert tool_call.output is None

    def test_creation_with_none_output(self):
        """Test creating ToolCallWithOutput with explicit None output."""
        tool_call = ToolCallWithOutput(
            tool_name="test_tool",
            arguments={"param": "value"},
            output=None,
        )
        assert tool_call.output is None

    def test_creation_with_list_output(self):
        """Test creating ToolCallWithOutput with list output."""
        tool_call = ToolCallWithOutput(
            tool_name="list_tool",
            arguments={},
            output=[{"id": 1}, {"id": 2}],
        )
        assert isinstance(tool_call.output, list)
        assert len(tool_call.output) == 2

    def test_creation_with_string_output(self):
        """Test creating ToolCallWithOutput with string output."""
        tool_call = ToolCallWithOutput(
            tool_name="string_tool",
            arguments={},
            output="Operation completed successfully",
        )
        assert tool_call.output == "Operation completed successfully"

    def test_model_dump(self):
        """Test model serialization."""
        tool_call = ToolCallWithOutput(
            tool_name="search_flights",
            arguments={"origin": "NYC"},
            output={"results": []},
        )
        dumped = tool_call.model_dump()
        assert dumped["tool_name"] == "search_flights"
        assert dumped["arguments"] == {"origin": "NYC"}
        assert dumped["output"] == {"results": []}


class TestTrajectoryStep:
    """Tests for TrajectoryStep model."""

    def test_creation_with_all_fields(self):
        """Test creating TrajectoryStep with all fields."""
        tool_call = ToolCallWithOutput(
            tool_name="search_flights",
            arguments={},
            output={},
        )
        step = TrajectoryStep(
            step_number=1,
            tool_calls=[tool_call],
            reasoning="Search for available flights",
        )
        assert step.step_number == 1
        assert len(step.tool_calls) == 1
        assert step.reasoning == "Search for available flights"

    def test_creation_with_multiple_tool_calls(self):
        """Test creating TrajectoryStep with multiple tool calls."""
        tool_calls = [
            ToolCallWithOutput(tool_name="tool1", arguments={}, output={}),
            ToolCallWithOutput(tool_name="tool2", arguments={}, output={}),
        ]
        step = TrajectoryStep(
            step_number=2,
            tool_calls=tool_calls,
            reasoning="Parallel tool execution",
        )
        assert len(step.tool_calls) == 2

    def test_creation_with_defaults(self):
        """Test creating TrajectoryStep with default values."""
        step = TrajectoryStep(step_number=1)
        assert step.step_number == 1
        assert step.tool_calls == []
        assert step.reasoning is None

    def test_creation_without_reasoning(self):
        """Test creating TrajectoryStep without reasoning."""
        step = TrajectoryStep(
            step_number=1,
            tool_calls=[ToolCallWithOutput(tool_name="test")],
        )
        assert step.reasoning is None


class TestConversationTrajectory:
    """Tests for ConversationTrajectory model."""

    def test_creation_with_all_fields(self):
        """Test creating ConversationTrajectory with all fields."""
        step = TrajectoryStep(
            step_number=1,
            tool_calls=[ToolCallWithOutput(tool_name="test")],
        )
        trajectory = ConversationTrajectory(
            query="Find flights to LA",
            steps=[step],
            final_response="Found 3 flights",
            tools_used=["search_flights"],
            categories_used=["Travel"],
        )
        assert trajectory.query == "Find flights to LA"
        assert len(trajectory.steps) == 1
        assert trajectory.final_response == "Found 3 flights"

    def test_creation_with_defaults(self):
        """Test creating ConversationTrajectory with defaults."""
        trajectory = ConversationTrajectory(query="Test query", final_response="")
        assert trajectory.query == "Test query"
        assert trajectory.steps == []
        assert trajectory.final_response == ""
        assert trajectory.tools_used == []
        assert trajectory.categories_used == []

    def test_creation_with_empty_final_response(self):
        """Test creating with empty final response."""
        trajectory = ConversationTrajectory(
            query="Test",
            final_response="",
        )
        assert trajectory.final_response == ""

    def test_model_dump_json(self):
        """Test JSON serialization."""
        trajectory = ConversationTrajectory(
            query="Test query",
            steps=[TrajectoryStep(step_number=1)],
            final_response="Done",
        )
        json_str = trajectory.model_dump_json()
        assert "Test query" in json_str
        assert '"steps"' in json_str


class TestStepByStepDatapoint:
    """Tests for StepByStepDatapoint model."""

    def test_creation_with_trajectory(self):
        """Test creating datapoint with trajectory."""
        trajectory = ConversationTrajectory(
            query="Test query",
            final_response="Done",
        )
        datapoint = StepByStepDatapoint(trajectory=trajectory)
        assert datapoint.trajectory.query == "Test query"
        assert datapoint.trajectory.final_response == "Done"

    def test_creation_with_metadata(self):
        """Test creating datapoint with generation metadata."""
        trajectory = ConversationTrajectory(query="Test", final_response="Done")
        datapoint = StepByStepDatapoint(
            trajectory=trajectory,
            generation_metadata={"num_actions": 2, "focus_category": "Travel"},
        )
        assert datapoint.generation_metadata["num_actions"] == 2

    def test_creation_with_verification(self):
        """Test creating datapoint with verification result."""
        trajectory = ConversationTrajectory(query="Test", final_response="Done")
        datapoint = StepByStepDatapoint(
            trajectory=trajectory,
            verification_result={"passed": True, "issues": []},
        )
        assert datapoint.verification_result["passed"] is True

    def test_model_dump_complete(self):
        """Test complete model dump."""
        trajectory = ConversationTrajectory(query="Test", final_response="Done")
        datapoint = StepByStepDatapoint(
            trajectory=trajectory,
            generation_metadata={"version": "1.0"},
            verification_result={"passed": True},
        )
        dumped = datapoint.model_dump()
        assert dumped["trajectory"]["query"] == "Test"
        assert dumped["generation_metadata"]["version"] == "1.0"


class TestVerificationResult:
    """Tests for VerificationResult model."""

    def test_creation_with_all_fields(self):
        """Test creating VerificationResult with all fields."""
        result = VerificationResult(
            query="Test query",
            tool_relevance_checks=[{"tool_name": "t1", "is_relevant": True}],
            order_is_correct=True,
            order_verification_details="Order is logical",
            output_validations=[{"tool_name": "t1", "valid": True}],
            placeholder_resolution={"all_resolved": True},
            overall_verification_passed=True,
            verification_summary="All checks passed",
        )
        assert result.query == "Test query"
        assert result.overall_verification_passed is True

    def test_creation_with_failure(self):
        """Test creating VerificationResult with failure."""
        result = VerificationResult(
            query="Test",
            order_is_correct=False,
            overall_verification_passed=False,
            verification_summary="Some checks failed",
        )
        assert result.overall_verification_passed is False

    def test_creation_with_defaults(self):
        """Test creating VerificationResult with defaults."""
        result = VerificationResult(
            query="Test",
            order_is_correct=True,
            overall_verification_passed=True,
        )
        assert result.tool_relevance_checks == []
        assert result.output_validations == []


class TestStepSelectionResult:
    """Tests for StepSelectionResult model."""

    def test_creation_with_all_fields(self):
        """Test creating StepSelectionResult with all fields."""
        result = StepSelectionResult(
            tool_name="search_flights",
            arguments={"origin": "NYC"},
            reasoning="User wants to search flights",
        )
        assert result.tool_name == "search_flights"
        assert result.arguments == {"origin": "NYC"}
        assert result.reasoning == "User wants to search flights"

    def test_creation_with_defaults(self):
        """Test creating StepSelectionResult with defaults."""
        result = StepSelectionResult(tool_name="test_tool", reasoning="")
        assert result.tool_name == "test_tool"
        assert result.arguments == {}
        assert result.reasoning == ""

    def test_creation_with_empty_reasoning(self):
        """Test creating with empty reasoning string."""
        result = StepSelectionResult(
            tool_name="tool",
            arguments={},
            reasoning="",
        )
        assert result.reasoning == ""


class TestQueryGenerationResult:
    """Tests for QueryGenerationResult model."""

    def test_creation_with_all_fields(self):
        """Test creating QueryGenerationResult with all fields."""
        result = QueryGenerationResult(
            query="Find flights to LA",
            intent="Travel planning",
            expected_tools=["search_flights", "book_hotel"],
        )
        assert result.query == "Find flights to LA"
        assert result.intent == "Travel planning"
        assert result.expected_tools == ["search_flights", "book_hotel"]

    def test_creation_with_defaults(self):
        """Test creating QueryGenerationResult with defaults."""
        result = QueryGenerationResult(
            query="Simple query",
            intent="Testing",
        )
        assert result.query == "Simple query"
        assert result.expected_tools == []

    def test_creation_with_empty_tools(self):
        """Test creating with explicitly empty tools list."""
        result = QueryGenerationResult(
            query="Test",
            intent="Test",
            expected_tools=[],
        )
        assert result.expected_tools == []

    def test_creation_with_single_tool(self):
        """Test creating with single tool in expected_tools."""
        result = QueryGenerationResult(
            query="Test",
            intent="Test",
            expected_tools=["single_tool"],
        )
        assert len(result.expected_tools) == 1


class TestModelRelationships:
    """Tests for relationships between models."""

    def test_trajectory_contains_steps(self):
        """Test that trajectory can contain multiple steps."""
        steps = [
            TrajectoryStep(step_number=1, tool_calls=[ToolCallWithOutput(tool_name="t1")]),
            TrajectoryStep(step_number=2, tool_calls=[ToolCallWithOutput(tool_name="t2")]),
        ]
        trajectory = ConversationTrajectory(
            query="Test",
            steps=steps,
            final_response="Done",
        )
        assert len(trajectory.steps) == 2
        assert trajectory.steps[0].step_number == 1
        assert trajectory.steps[1].step_number == 2

    def test_step_contains_tool_calls(self):
        """Test that steps contain tool calls."""
        step = TrajectoryStep(
            step_number=1,
            tool_calls=[
                ToolCallWithOutput(tool_name="tool1", output={"result": 1}),
                ToolCallWithOutput(tool_name="tool2", output={"result": 2}),
            ],
        )
        assert len(step.tool_calls) == 2
        assert step.tool_calls[0].tool_name == "tool1"
        assert step.tool_calls[1].tool_name == "tool2"

    def test_datapoint_contains_trajectory(self):
        """Test that datapoint wraps trajectory."""
        trajectory = ConversationTrajectory(
            query="Test query",
            steps=[TrajectoryStep(step_number=1)],
            final_response="Done",
        )
        datapoint = StepByStepDatapoint(trajectory=trajectory)
        assert datapoint.trajectory.steps[0].step_number == 1
