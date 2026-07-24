"""Unit tests for query generation methods in StepByStepGenerator.

These tests verify the generate_user_query and validate_expected_tools
methods for generating user queries.
"""

import json

import pytest

from apigen_step_by_step import (
    QueryGenerationResult,
    StepByStepGenerator,
)
from step_by_step_models import ObjectMap, StateSnapshot
from tests.mocks.mock_llm_client import MockLLMClient
from tests.mocks.mock_llm_responses import (
    MALFORMED_JSON_MISSING_QUOTE,
    MALFORMED_JSON_UNCLOSED_ARRAY,
    MALFORMED_JSON_UNCLOSED_BRACE,
    QUERY_RESPONSE_EMPTY_TOOLS,
    QUERY_RESPONSE_INVALID_TOOL,
    QUERY_RESPONSE_MISSING_EXPECTED_TOOLS,
    QUERY_RESPONSE_TOO_MANY_TOOLS,
    QUERY_RESPONSE_WRONG_TOOL_COUNT,
    VALID_QUERY_RESPONSE_2_TOOLS,
    VALID_QUERY_RESPONSE_3_TOOLS,
    VALID_SEQUENCE_RESPONSE,
)


class QueryTestToolManager:
    """Complete structural manager contract for query-generation tests."""

    def __init__(self) -> None:
        names_and_categories = (
            ("search_flights", "Travel"),
            ("book_hotel", "Travel"),
            ("get_weather", "Information"),
            ("send_email", "Communication"),
            ("search_restaurants", "Food"),
            ("make_reservation", "Food"),
            ("create_calendar_event", "Productivity"),
            ("get_reviews", "Information"),
        )
        self.tool_schemas: list[ObjectMap] = [
            {
                "name": name,
                "description": name.replace("_", " "),
                "category": category,
                "output_type": "dict",
                "output_description": "Test output",
            }
            for name, category in names_and_categories
        ]
        self.tool_outputs: ObjectMap = {}
        self.captured_invocations: list[ObjectMap] = []
        self.should_fail: bool = False
        self.fail_tool: str | None = None
        self.python_tool_instances: dict[str, object] = {}
        self.api_name_to_class_key: dict[str, str] = {}

    def get_tools_json_schema(self) -> list[ObjectMap]:
        return self.tool_schemas

    def get_tool_schema(self, tool_name: str) -> ObjectMap:
        return next(
            tool for tool in self.tool_schemas if tool["name"] == tool_name
        )

    def get_categories(self) -> list[str]:
        return sorted(
            {
                category
                for tool in self.tool_schemas
                if isinstance(category := tool.get("category"), str)
            }
        )

    def get_tools_by_category(self, category: str) -> list[ObjectMap]:
        return [
            tool for tool in self.tool_schemas if tool["category"] == category
        ]

    def get_tool_category(self, tool_name: str) -> str | None:
        category = self.get_tool_schema(tool_name).get("category")
        return category if isinstance(category, str) else None

    def tool_exists(self, tool_name: str) -> bool:
        return any(tool["name"] == tool_name for tool in self.tool_schemas)

    def invoke_tool(self, tool_name: str, params: ObjectMap) -> object:
        del params
        return self.tool_outputs.get(tool_name, {})

    def get_api_state(self) -> StateSnapshot:
        return {}

    def restore_api_state(self, state: StateSnapshot) -> None:
        del state

    def initialize_api_state(self, force_new: bool = False) -> None:
        del force_new

    def has_python_implementation(self, tool_name: str) -> bool:
        del tool_name
        return False

    def invoke_python_tool(
        self,
        tool_name: str,
        params: ObjectMap,
    ) -> object:
        raise NotImplementedError(f"No Python implementation for {tool_name}: {params}")


@pytest.fixture
def mock_tools() -> QueryTestToolManager:
    return QueryTestToolManager()


class TestGenerateUserQuery:
    """Tests for generate_user_query method."""

    @pytest.fixture
    def generator(self, mock_llm: MockLLMClient, mock_tools: QueryTestToolManager):
        """Create a generator for 2 actions."""
        return StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

    def test_generate_query_success(self, generator: StepByStepGenerator, mock_llm: MockLLMClient):
        """Test successful query generation."""
        mock_llm.set_responses([VALID_QUERY_RESPONSE_2_TOOLS, VALID_SEQUENCE_RESPONSE])

        result = generator.generate_user_query()

        assert isinstance(result, QueryGenerationResult)
        assert result.query != ""
        assert result.intent != ""
        assert len(result.expected_tools) == 2

    def test_generate_query_plain_json(self, generator: StepByStepGenerator, mock_llm: MockLLMClient):
        """Test query generation with plain JSON response."""
        # Use valid tools from mock
        import json
        response = json.dumps({
            "query": "Find flights and book hotel",
            "intent": "Travel planning",
            "expected_tools": ["search_flights", "book_hotel"],
        })
        mock_llm.set_responses([
            response,
            VALID_SEQUENCE_RESPONSE,
        ])

        result = generator.generate_user_query()

        assert result.query == "Find flights and book hotel"
        assert result.expected_tools == ["search_flights", "book_hotel"]

    def test_generate_query_3_tools(self, mock_llm: MockLLMClient, mock_tools: QueryTestToolManager):
        """Test query generation for 3 tools."""
        generator = StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=3,
        )
        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_3_TOOLS,
            VALID_SEQUENCE_RESPONSE,
        ])

        result = generator.generate_user_query()

        assert len(result.expected_tools) == 3

    def test_generate_query_with_focus_category(self, generator: StepByStepGenerator, mock_llm: MockLLMClient):
        """Test query generation with focus category."""
        mock_llm.set_responses([VALID_QUERY_RESPONSE_2_TOOLS, VALID_SEQUENCE_RESPONSE])

        _ = generator.generate_user_query(focus_category="Travel")

        # Verify the category was mentioned in prompt
        assert len(mock_llm.captured_prompts) > 0
        prompt_text = str(mock_llm.captured_prompts[0])
        assert "Travel" in prompt_text

    def test_generate_query_wrong_tool_count_retries(self, generator: StepByStepGenerator, mock_llm: MockLLMClient):
        """Test retry when expected_tools count is wrong."""
        # First attempt has wrong count, second succeeds
        mock_llm.set_responses([
            QUERY_RESPONSE_WRONG_TOOL_COUNT,  # 1 tool instead of 2
            VALID_QUERY_RESPONSE_2_TOOLS,      # Correct 2 tools
            VALID_SEQUENCE_RESPONSE,
        ])

        result = generator.generate_user_query()

        # Should eventually succeed after retry
        assert len(result.expected_tools) == 2
        # Should have made multiple calls
        assert mock_llm.call_count >= 2

    def test_generate_query_too_many_tools(self, generator: StepByStepGenerator, mock_llm: MockLLMClient):
        """Test handling when too many tools are requested."""
        mock_llm.set_responses([
            QUERY_RESPONSE_TOO_MANY_TOOLS,
            VALID_QUERY_RESPONSE_2_TOOLS,  # Retry succeeds
            VALID_SEQUENCE_RESPONSE,
        ])

        result = generator.generate_user_query()

        assert len(result.expected_tools) == 2

    def test_generate_query_invalid_tools_retries(self, generator: StepByStepGenerator, mock_llm: MockLLMClient):
        """Test retry when tools don't exist."""
        mock_llm.set_responses([
            QUERY_RESPONSE_INVALID_TOOL,     # Invalid tools
            VALID_QUERY_RESPONSE_2_TOOLS,    # Valid tools
            VALID_SEQUENCE_RESPONSE,
        ])

        result = generator.generate_user_query()

        assert all(generator.tool_manager.tool_exists(t) for t in result.expected_tools)

    def test_generate_query_json_decode_error_retries(self, generator: StepByStepGenerator, mock_llm: MockLLMClient):
        """Test retry on JSON decode error."""
        mock_llm.set_responses([
            MALFORMED_JSON_UNCLOSED_BRACE,
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
        ])

        result = generator.generate_user_query()

        assert result.query != ""
        assert mock_llm.call_count >= 2

    def test_generate_query_max_retries_exceeded(self, generator: StepByStepGenerator, mock_llm: MockLLMClient):
        """Test graceful failure after max retries."""
        # All responses fail
        mock_llm.set_responses([
            MALFORMED_JSON_UNCLOSED_BRACE,
            MALFORMED_JSON_UNCLOSED_ARRAY,
            MALFORMED_JSON_MISSING_QUOTE,
        ])

        result = generator.generate_user_query(max_retries=2)

        # Should return empty result
        assert result.query == ""
        assert result.expected_tools == []

    def test_generate_query_empty_expected_tools(self, generator: StepByStepGenerator, mock_llm: MockLLMClient):
        """Test handling of empty expected_tools."""
        mock_llm.set_responses([
            QUERY_RESPONSE_EMPTY_TOOLS,
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
        ])

        result = generator.generate_user_query()

        # Should retry and get valid tools
        assert len(result.expected_tools) == 2

    def test_generate_query_missing_expected_tools(self, generator: StepByStepGenerator, mock_llm: MockLLMClient):
        """Test handling when expected_tools field is missing."""
        mock_llm.set_responses([
            QUERY_RESPONSE_MISSING_EXPECTED_TOOLS,
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
        ])

        result = generator.generate_user_query()

        # Should retry and get valid tools
        assert len(result.expected_tools) == 2


class TestValidateExpectedTools:
    """Tests for validate_expected_tools method."""

    @pytest.fixture
    def generator(self, mock_llm: MockLLMClient, mock_tools: QueryTestToolManager):
        """Create a generator for 2 actions."""
        return StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

    def test_validate_expected_tools_success(self, generator: StepByStepGenerator, mock_llm: MockLLMClient):
        """Test successful tool sequence validation."""
        mock_llm.set_responses([VALID_SEQUENCE_RESPONSE])

        is_valid, message = generator.validate_expected_tools(
            query="Search flights and book hotel",
            expected_tools=["search_flights", "book_hotel"],
            intent="Travel planning",
        )

        assert is_valid is True
        assert message == ""

    def test_validate_expected_tools_failure(self, generator: StepByStepGenerator, mock_llm: MockLLMClient):
        """Test tool sequence validation failure."""
        from tests.mocks.mock_llm_responses import INVALID_SEQUENCE_RESPONSE

        mock_llm.set_responses([INVALID_SEQUENCE_RESPONSE])

        is_valid, message = generator.validate_expected_tools(
            query="Send email and check weather",
            expected_tools=["get_weather", "send_email"],
            intent="Mixed tasks",
        )

        assert is_valid is False
        assert "validation failed" in message.lower() or "issues" in message.lower()

    def test_validate_expected_tools_judge_error_fails_closed(
        self,
        generator: StepByStepGenerator,
        mock_llm: MockLLMClient,
    ):
        mock_llm.set_responses(["not-json"])

        is_valid, message = generator.validate_expected_tools(
            query="Search flights and book hotel",
            expected_tools=["search_flights", "book_hotel"],
            intent="Travel planning",
        )

        assert is_valid is False
        assert "validation error" in message.lower()

    def test_validate_expected_tools_count_mismatch(self, generator: StepByStepGenerator):
        """Test validation when tool count doesn't match num_actions."""
        is_valid, message = generator.validate_expected_tools(
            query="Test",
            expected_tools=["search_flights"],  # Only 1, but num_actions=2
            intent="Test",
        )

        # Should fail immediately without calling LLM
        assert is_valid is False
        assert "count" in message.lower()

    def test_validate_expected_tools_empty_tools(self, generator: StepByStepGenerator):
        """Test validation with empty tools list."""
        is_valid, message = generator.validate_expected_tools(
            query="Test",
            expected_tools=[],
            intent="Test",
        )

        assert is_valid is False
        assert "2" in message  # Should mention expected count


class TestGenerateUserQueryEdgeCases:
    """Tests for edge cases in query generation."""

    @pytest.fixture
    def generator(self, mock_llm: MockLLMClient, mock_tools: QueryTestToolManager):
        """Create a generator for 2 actions."""
        return StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

    def test_generate_query_with_context_hint(self, generator: StepByStepGenerator, mock_llm: MockLLMClient):
        """Test query generation with context hint (passed via focus_category)."""
        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
        ])

        hint = "Travel"
        result = generator.generate_user_query(focus_category=hint)

        # Verify result was generated
        assert result is not None
        # Verify the category was mentioned in prompt
        assert len(mock_llm.captured_prompts) > 0
        prompt_text = str(mock_llm.captured_prompts[0])
        assert "Travel" in prompt_text

    def test_generate_query_with_validation_feedback(self, generator: StepByStepGenerator, mock_llm: MockLLMClient):
        """Test query generation with validation feedback."""
        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
        ])

        feedback = "Tools were not relevant to query"
        _ = generator.generate_user_query(validation_feedback=feedback)

        # Verify feedback was included
        assert len(mock_llm.captured_prompts) > 0
        prompt_text = str(mock_llm.captured_prompts[0])
        assert feedback in prompt_text

    def test_generate_query_parses_code_block(self, generator: StepByStepGenerator, mock_llm: MockLLMClient):
        """Test parsing JSON from code blocks."""
        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
        ])

        result = generator.generate_user_query()

        assert "search_flights" in result.expected_tools or "book_hotel" in result.expected_tools

    def test_generate_query_extra_text_around(self, generator: StepByStepGenerator, mock_llm: MockLLMClient):
        """Test parsing JSON with extra text around it."""
        response = '''Here's the query:
```json
{
  "query": "Find restaurants",
  "intent": "Dining",
  "expected_tools": ["search_restaurants", "get_reviews"]
}
```
Hope this helps!'''
        mock_llm.set_responses([response, VALID_SEQUENCE_RESPONSE])

        result = generator.generate_user_query()

        assert result.query == "Find restaurants"

    def test_generate_query_unicode_in_query(self, generator: StepByStepGenerator, mock_llm: MockLLMClient):
        """Test handling of unicode characters in query."""
        response = json.dumps({
            "query": "Book a table at José's Café 🍽️",
            "intent": "Restaurant booking",
            "expected_tools": ["search_restaurants", "make_reservation"],
        })
        mock_llm.set_responses([response, VALID_SEQUENCE_RESPONSE])

        result = generator.generate_user_query()

        assert "José" in result.query
        assert "🍽️" in result.query


class TestQueryResultStructure:
    """Tests for query result structure."""

    def test_query_result_has_required_fields(self, mock_llm: MockLLMClient, mock_tools: QueryTestToolManager):
        """Test that query result has all required fields."""
        generator = StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )
        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
        ])

        result = generator.generate_user_query()

        assert hasattr(result, "query")
        assert hasattr(result, "intent")
        assert hasattr(result, "expected_tools")
        assert isinstance(result.expected_tools, list)

    def test_query_result_defaults(self):
        """Test query result default values."""
        result = QueryGenerationResult(
            query="Test",
            intent="Test",
        )

        assert result.expected_tools == []
