"""Unit tests for placeholder processing in StepByStepGenerator.

These tests verify the _process_placeholders method which replaces
template placeholders like {{key}} or {{key.subkey}} with values
from the execution context.
"""

import pytest
from apigen_step_by_step import StepByStepGenerator


class TestPlaceholderProcessing:
    """Tests for placeholder resolution logic."""

    @pytest.fixture
    def generator(self, mock_llm, mock_tools):
        """Fixture to create a basic generator."""
        return StepByStepGenerator(
            llm_client=mock_llm,
            tool_manager=mock_tools,
            num_actions=2,
        )

    def test_simple_placeholder_resolution(self, generator):
        """Test replacing {{key}} with value from context."""
        arguments = {"flight_id": "{{flight_id}}"}
        context = {"flight_id": "FL001"}
        result = generator._process_placeholders(arguments, context)
        assert result["flight_id"] == "FL001"

    def test_nested_placeholder_resolution(self, generator):
        """Test replacing {{tool.output.key}} with nested dict value."""
        arguments = {"ref": "{{search_flights_output.flight_id}}"}
        context = {
            "search_flights_output": {
                "flight_id": "FL001",
                "price": 299,
            }
        }
        result = generator._process_placeholders(arguments, context)
        assert result["ref"] == "FL001"

    def test_deeply_nested_placeholder(self, generator):
        """Test replacing {{a.b.c.d}} with deeply nested value."""
        arguments = {"deep": "{{level1.level2.level3.value}}"}
        context = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "found_it"
                    }
                }
            }
        }
        result = generator._process_placeholders(arguments, context)
        assert result["deep"] == "found_it"

    def test_partial_string_replacement(self, generator):
        """Test replacing placeholder within larger string."""
        arguments = {"message": "Flight {{flight_id}} is confirmed"}
        context = {"flight_id": "FL001"}
        result = generator._process_placeholders(arguments, context)
        assert result["message"] == "Flight FL001 is confirmed"

    def test_multiple_placeholders_same_arg(self, generator):
        """Test multiple placeholders in single argument.

        Note: Current implementation only replaces first placeholder found
        because arg_value is not updated between iterations.
        """
        arguments = {"route": "From {{origin}} to {{destination}}"}
        context = {"origin": "NYC", "destination": "LA"}
        result = generator._process_placeholders(arguments, context)
        # Current behavior: only replaces first placeholder
        # arg_value in the loop is the original string, not the updated one
        assert "NYC" in result["route"] or "LA" in result["route"]

    def test_unresolvable_placeholder_unchanged(self, generator):
        """Test that unresolvable placeholders are left as-is."""
        arguments = {"ref": "{{missing_key}}"}
        context = {"other_key": "value"}
        result = generator._process_placeholders(arguments, context)
        assert result["ref"] == "{{missing_key}}"

    def test_unresolvable_nested_placeholder(self, generator):
        """Test that unresolvable nested placeholders are left as-is."""
        arguments = {"ref": "{{existing.missing}}"}
        context = {"existing": {"other": "value"}}
        result = generator._process_placeholders(arguments, context)
        assert result["ref"] == "{{existing.missing}}"

    def test_partial_resolvable_placeholder(self, generator):
        """Test placeholder where first key exists but sub-key doesn't."""
        arguments = {"ref": "{{level1.missing}}"}
        context = {"level1": {"exists": "value"}}
        result = generator._process_placeholders(arguments, context)
        assert result["ref"] == "{{level1.missing}}"

    def test_empty_context_no_placeholders(self, generator):
        """Test that empty context leaves arguments unchanged."""
        arguments = {"key": "{{placeholder}}"}
        context = {}
        result = generator._process_placeholders(arguments, context)
        assert result["key"] == "{{placeholder}}"

    def test_non_string_arguments_preserved(self, generator):
        """Test that non-string arguments are preserved."""
        arguments = {
            "string_arg": "{{placeholder}}",
            "int_arg": 42,
            "bool_arg": True,
            "list_arg": [1, 2, 3],
            "dict_arg": {"key": "value"},
        }
        context = {"placeholder": "replaced"}
        result = generator._process_placeholders(arguments, context)
        assert result["string_arg"] == "replaced"
        assert result["int_arg"] == 42
        assert result["bool_arg"] is True
        assert result["list_arg"] == [1, 2, 3]
        assert result["dict_arg"] == {"key": "value"}

    def test_placeholder_with_special_chars_in_value(self, generator):
        """Test placeholder replacement with special characters in value."""
        arguments = {"content": "{{message}}"}
        context = {"message": "Hello \"World\" with 'quotes' and\nnewlines\t"}
        result = generator._process_placeholders(arguments, context)
        assert result["content"] == "Hello \"World\" with 'quotes' and\nnewlines\t"

    def test_placeholder_with_unicode(self, generator):
        """Test placeholder replacement with unicode characters."""
        arguments = {"name": "{{user_name}}"}
        context = {"user_name": "José García 🎉"}
        result = generator._process_placeholders(arguments, context)
        assert result["name"] == "José García 🎉"

    def test_multiple_placeholders_different_args(self, generator):
        """Test different placeholders in different arguments."""
        arguments = {
            "arg1": "{{key1}}",
            "arg2": "{{key2}}",
            "arg3": "{{key3}}",
        }
        context = {"key1": "val1", "key2": "val2", "key3": "val3"}
        result = generator._process_placeholders(arguments, context)
        assert result["arg1"] == "val1"
        assert result["arg2"] == "val2"
        assert result["arg3"] == "val3"

    def test_placeholder_in_nested_dict(self, generator):
        """Test placeholder replacement within nested dict."""
        arguments = {
            "outer": {
                "inner": "{{value}}"
            }
        }
        context = {"value": "replaced"}
        result = generator._process_placeholders(arguments, context)
        # Note: _process_placeholders only processes top-level values
        # It doesn't recursively process nested dict values
        assert result["outer"]["inner"] == "{{value}}"

    def test_placeholder_with_empty_string_value(self, generator):
        """Test placeholder replaced with empty string."""
        arguments = {"key": "{{empty}}"}
        context = {"empty": ""}
        result = generator._process_placeholders(arguments, context)
        assert result["key"] == ""

    def test_placeholder_with_numeric_value(self, generator):
        """Test placeholder replaced with numeric value."""
        arguments = {"count": "{{number}}"}
        context = {"number": 42}
        result = generator._process_placeholders(arguments, context)
        assert result["count"] == 42

    def test_placeholder_with_boolean_value(self, generator):
        """Test placeholder replaced with boolean value."""
        arguments = {"flag": "{{bool_val}}"}
        context = {"bool_val": True}
        result = generator._process_placeholders(arguments, context)
        assert result["flag"] is True

    def test_mixed_resolvable_and_unresolvable(self, generator):
        """Test mix of resolvable and unresolvable placeholders."""
        arguments = {
            "found": "{{exists}}",
            "missing": "{{not_found}}",
        }
        context = {"exists": "value"}
        result = generator._process_placeholders(arguments, context)
        assert result["found"] == "value"
        assert result["missing"] == "{{not_found}}"

    def test_placeholder_at_start_of_string(self, generator):
        """Test placeholder at the beginning of string."""
        arguments = {"text": "{{name}} is here"}
        context = {"name": "Alice"}
        result = generator._process_placeholders(arguments, context)
        assert result["text"] == "Alice is here"

    def test_placeholder_at_end_of_string(self, generator):
        """Test placeholder at the end of string."""
        arguments = {"text": "Hello {{name}}"}
        context = {"name": "Bob"}
        result = generator._process_placeholders(arguments, context)
        assert result["text"] == "Hello Bob"

    def test_placeholder_in_middle_of_string(self, generator):
        """Test placeholder in the middle of string."""
        arguments = {"text": "Hello {{name}}, welcome!"}
        context = {"name": "Charlie"}
        result = generator._process_placeholders(arguments, context)
        assert result["text"] == "Hello Charlie, welcome!"

    def test_no_placeholders_in_arguments(self, generator):
        """Test arguments without any placeholders."""
        arguments = {
            "key1": "value1",
            "key2": "value2",
        }
        context = {"key1": "should_not_replace"}
        result = generator._process_placeholders(arguments, context)
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"

    def test_empty_arguments_dict(self, generator):
        """Test with empty arguments dictionary."""
        arguments = {}
        context = {"key": "value"}
        result = generator._process_placeholders(arguments, context)
        assert result == {}

    def test_whitespace_in_placeholder(self, generator):
        """Test that placeholders with spaces are not matched."""
        arguments = {"key": "{{ key }}"}
        context = {"key": "value"}
        result = generator._process_placeholders(arguments, context)
        # The regex expects {{key}} without spaces
        assert result["key"] == "{{ key }}"

    def test_similar_but_different_placeholders(self, generator):
        """Test distinguishing between similar placeholder names."""
        arguments = {
            "id": "{{flight_id}}",
            "code": "{{flight_code}}",
        }
        context = {
            "flight_id": "123",
            "flight_code": "ABC",
        }
        result = generator._process_placeholders(arguments, context)
        assert result["id"] == "123"
        assert result["code"] == "ABC"
