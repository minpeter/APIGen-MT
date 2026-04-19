"""Unit tests for query generation methods in StepByStepGenerator.

These tests verify the generate_user_query and validate_expected_tools
methods for generating user queries.
"""

import json
import pytest
from apigen_step_by_step import (
    StepByStepGenerator,
    QueryGenerationResult,
)
from tests.mocks.mock_llm_responses import (
    VALID_QUERY_RESPONSE_2_TOOLS,
    VALID_QUERY_RESPONSE_3_TOOLS,
    VALID_QUERY_RESPONSE_PLAIN_JSON,
    VALID_SEQUENCE_RESPONSE,
    QUERY_RESPONSE_WRONG_TOOL_COUNT,
    QUERY_RESPONSE_TOO_MANY_TOOLS,
    QUERY_RESPONSE_INVALID_TOOL,
    QUERY_RESPONSE_MISSING_QUERY_FIELD,
    QUERY_RESPONSE_MISSING_EXPECTED_TOOLS,
    QUERY_RESPONSE_EMPTY_TOOLS,
    MALFORMED_JSON_UNCLOSED_BRACE,
    MALFORMED_JSON_UNCLOSED_ARRAY,
    MALFORMED_JSON_MISSING_QUOTE,
)


class TestGenerateUserQuery:
    """Tests for generate_user_query method."""

    @pytest.fixture
    def generator(self, mock_llm, mock_tools):
        """Create a generator for 2 actions."""
        return StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

    def test_generate_query_success(self, generator, mock_llm):
        """Test successful query generation."""
        mock_llm.set_responses([VALID_QUERY_RESPONSE_2_TOOLS, VALID_SEQUENCE_RESPONSE])

        result = generator.generate_user_query()

        assert isinstance(result, QueryGenerationResult)
        assert result.query != ""
        assert result.intent != ""
        assert len(result.expected_tools) == 2

    def test_generate_query_plain_json(self, generator, mock_llm):
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

    def test_generate_query_3_tools(self, mock_llm, mock_tools):
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

    def test_generate_query_with_focus_category(self, generator, mock_llm):
        """Test query generation with focus category."""
        mock_llm.set_responses([VALID_QUERY_RESPONSE_2_TOOLS, VALID_SEQUENCE_RESPONSE])

        result = generator.generate_user_query(focus_category="Travel")

        # Verify the category was mentioned in prompt
        assert len(mock_llm.captured_prompts) > 0
        prompt_text = str(mock_llm.captured_prompts[0])
        assert "Travel" in prompt_text

    def test_generate_query_wrong_tool_count_retries(self, generator, mock_llm):
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

    def test_generate_query_too_many_tools(self, generator, mock_llm):
        """Test handling when too many tools are requested."""
        mock_llm.set_responses([
            QUERY_RESPONSE_TOO_MANY_TOOLS,
            VALID_QUERY_RESPONSE_2_TOOLS,  # Retry succeeds
            VALID_SEQUENCE_RESPONSE,
        ])

        result = generator.generate_user_query()

        assert len(result.expected_tools) == 2

    def test_generate_query_invalid_tools_retries(self, generator, mock_llm):
        """Test retry when tools don't exist."""
        mock_llm.set_responses([
            QUERY_RESPONSE_INVALID_TOOL,     # Invalid tools
            VALID_QUERY_RESPONSE_2_TOOLS,    # Valid tools
            VALID_SEQUENCE_RESPONSE,
        ])

        result = generator.generate_user_query()

        assert all(generator.tool_manager.tool_exists(t) for t in result.expected_tools)

    def test_generate_query_json_decode_error_retries(self, generator, mock_llm):
        """Test retry on JSON decode error."""
        mock_llm.set_responses([
            MALFORMED_JSON_UNCLOSED_BRACE,
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
        ])

        result = generator.generate_user_query()

        assert result.query != ""
        assert mock_llm.call_count >= 2

    def test_generate_query_max_retries_exceeded(self, generator, mock_llm):
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

    def test_generate_query_empty_expected_tools(self, generator, mock_llm):
        """Test handling of empty expected_tools."""
        mock_llm.set_responses([
            QUERY_RESPONSE_EMPTY_TOOLS,
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
        ])

        result = generator.generate_user_query()

        # Should retry and get valid tools
        assert len(result.expected_tools) == 2

    def test_generate_query_missing_expected_tools(self, generator, mock_llm):
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
    def generator(self, mock_llm, mock_tools):
        """Create a generator for 2 actions."""
        return StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

    def test_validate_expected_tools_success(self, generator, mock_llm):
        """Test successful tool sequence validation."""
        mock_llm.set_responses([VALID_SEQUENCE_RESPONSE])

        is_valid, message = generator.validate_expected_tools(
            query="Search flights and book hotel",
            expected_tools=["search_flights", "book_hotel"],
            intent="Travel planning",
        )

        assert is_valid is True
        assert message == ""

    def test_validate_expected_tools_failure(self, generator, mock_llm):
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

    def test_validate_expected_tools_count_mismatch(self, generator):
        """Test validation when tool count doesn't match num_actions."""
        is_valid, message = generator.validate_expected_tools(
            query="Test",
            expected_tools=["search_flights"],  # Only 1, but num_actions=2
            intent="Test",
        )

        # Should fail immediately without calling LLM
        assert is_valid is False
        assert "count" in message.lower()

    def test_validate_expected_tools_empty_tools(self, generator):
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
    def generator(self, mock_llm, mock_tools):
        """Create a generator for 2 actions."""
        return StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

    def test_generate_query_with_context_hint(self, generator, mock_llm):
        """Test query generation with context hint."""
        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
        ])

        hint = "Focus on travel planning"
        result = generator.generate_user_query(context_hint=hint)

        # Verify result was generated
        assert result is not None
        # Hint should be passed - verify the method accepts it without error

    def test_generate_query_with_validation_feedback(self, generator, mock_llm):
        """Test query generation with validation feedback."""
        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
        ])

        feedback = "Tools were not relevant to query"
        result = generator.generate_user_query(validation_feedback=feedback)

        # Verify feedback was included
        assert len(mock_llm.captured_prompts) > 0
        prompt_text = str(mock_llm.captured_prompts[0])
        assert feedback in prompt_text

    def test_generate_query_parses_code_block(self, generator, mock_llm):
        """Test parsing JSON from code blocks."""
        mock_llm.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
        ])

        result = generator.generate_user_query()

        assert "search_flights" in result.expected_tools or "book_hotel" in result.expected_tools

    def test_generate_query_extra_text_around(self, generator, mock_llm):
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

    def test_generate_query_unicode_in_query(self, generator, mock_llm):
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

    def test_query_result_has_required_fields(self, mock_llm, mock_tools):
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
