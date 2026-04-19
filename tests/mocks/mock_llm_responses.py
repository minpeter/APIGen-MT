"""Pre-defined mock LLM responses for different prompt types.

This module contains various valid and invalid mock responses for testing
different scenarios in the step-by-step datapoint generation system.
"""

# =============================================================================
# VALID QUERY GENERATION RESPONSES
# =============================================================================

VALID_QUERY_RESPONSE_2_TOOLS = """```json
{
  "query": "Find flights from NYC to LA and book a hotel near the airport",
  "intent": "Plan travel with flight search and hotel booking",
  "expected_tools": ["search_flights", "book_hotel"]
}
```"""

VALID_QUERY_RESPONSE_3_TOOLS = """```json
{
  "query": "Search for restaurants, get reviews, and make a reservation",
  "intent": "Find dining options and make reservation",
  "expected_tools": ["search_restaurants", "get_reviews", "make_reservation"]
}
```"""

VALID_QUERY_RESPONSE_PLAIN_JSON = """{
  "query": "Check weather and send an alert",
  "intent": "Weather monitoring and alerting",
  "expected_tools": ["check_weather", "send_alert"]
}"""

# =============================================================================
# INVALID QUERY GENERATION RESPONSES
# =============================================================================

QUERY_RESPONSE_WRONG_TOOL_COUNT = """```json
{
  "query": "Find flights from NYC to LA",
  "intent": "Search for flights",
  "expected_tools": ["search_flights"]
}
```"""

QUERY_RESPONSE_TOO_MANY_TOOLS = """```json
{
  "query": "Do many things",
  "intent": "Multi-step task",
  "expected_tools": ["tool1", "tool2", "tool3", "tool4", "tool5"]
}
```"""

QUERY_RESPONSE_INVALID_TOOL = """```json
{
  "query": "Do something with tools",
  "intent": "Unknown intent",
  "expected_tools": ["nonexistent_tool_1", "nonexistent_tool_2"]
}
```"""

QUERY_RESPONSE_MISSING_QUERY_FIELD = """```json
{
  "intent": "Missing query field",
  "expected_tools": ["tool1", "tool2"]
}
```"""

QUERY_RESPONSE_MISSING_INTENT_FIELD = """```json
{
  "query": "Missing intent field",
  "expected_tools": ["tool1", "tool2"]
}
```"""

QUERY_RESPONSE_MISSING_EXPECTED_TOOLS = """```json
{
  "query": "Missing expected_tools",
  "intent": "Test query"
}
```"""

QUERY_RESPONSE_EMPTY_TOOLS = """```json
{
  "query": "A query",
  "intent": "Testing",
  "expected_tools": []
}
```"""

QUERY_RESPONSE_EMPTY_QUERY = """```json
{
  "query": "",
  "intent": "Empty query",
  "expected_tools": ["tool1", "tool2"]
}
```"""

# =============================================================================
# MALFORMED JSON RESPONSES
# =============================================================================

MALFORMED_JSON_UNCLOSED_BRACE = """{"query": "test", "expected_tools": ["t1"]"""

MALFORMED_JSON_UNCLOSED_ARRAY = """{"expected_tools": ["t1" """

MALFORMED_JSON_MISSING_QUOTE = """{query: "test"}"""

MALFORMED_JSON_TRAILING_COMMA = """{"query": "test",}"""

MALFORMED_JSON_SINGLE_QUOTES = """{'query': 'test'}"""

MALFORMED_JSON_EXTRA_TEXT_OUTSIDE = """Some text {"query": "test"} more text"""

MALFORMED_JSON_XML_INSTEAD = """<response><query>test</query></response>"""

MALFORMED_JSON_IN_CODE_BLOCK = """```json
{"query": "broken",
```"""

# =============================================================================
# TOOL SEQUENCE VALIDATION RESPONSES
# =============================================================================

VALID_SEQUENCE_RESPONSE = """```json
{
  "is_valid": true,
  "issues": []
}
```"""

INVALID_SEQUENCE_RESPONSE = """```json
{
  "is_valid": false,
  "issues": ["Tool sequence doesn't match query intent", "Second tool requires output from first"]
}
```"""

SEQUENCE_VALIDATION_MISSING_FIELD = """{"is_valid": true}"""

SEQUENCE_VALIDATION_NO_JSON_BLOCK = """{"is_valid": true, "issues": []}"""

SEQUENCE_VALIDATION_MALFORMED = """```json
{"is_valid": false, "issues":
```"""

# =============================================================================
# STEP SELECTION RESPONSES
# =============================================================================

VALID_STEP_RESPONSE = """```json
{
  "tool_name": "search_flights",
  "arguments": {"origin": "NYC", "destination": "LA"},
  "reasoning": "User wants to find flights first before booking hotel"
}
```"""

VALID_STEP_RESPONSE_PLAIN_JSON = """{
  "tool_name": "book_hotel",
  "arguments": {"location": "LAX", "check_in": "2024-01-01"},
  "reasoning": "Need to book hotel after finding flights"
}"""

STEP_RESPONSE_WITH_PLACEHOLDER = """```json
{
  "tool_name": "send_message",
  "arguments": {
    "recipient": "user",
    "content": "Your flight {{search_flights_output.flight_id}} is confirmed"
  },
  "reasoning": "Using placeholder from previous step"
}
```"""

STEP_RESPONSE_MISSING_TOOL_NAME = """```json
{
  "arguments": {"param": "value"},
  "reasoning": "Some reasoning"
}
```"""

STEP_RESPONSE_MISSING_ARGUMENTS = """```json
{
  "tool_name": "search_flights",
  "reasoning": "Need to search first"
}
```"""

STEP_RESPONSE_MISSING_REASONING = """```json
{
  "tool_name": "search_flights",
  "arguments": {"origin": "NYC"}
}
```"""

STEP_RESPONSE_INVALID_TOOL = """```json
{
  "tool_name": "invalid_tool_that_doesnt_exist",
  "arguments": {},
  "reasoning": "Trying to use unavailable tool"
}
```"""

STEP_RESPONSE_EMPTY = ""

STEP_RESPONSE_NOT_JSON = "This is just plain text, not JSON at all"

STEP_RESPONSE_ONLY_WHITESPACE = "   \n\t   "

STEP_RESPONSE_EXTRA_TEXT_AROUND = """Here's the tool call:
```json
{
  "tool_name": "search_restaurants",
  "arguments": {"location": "NYC"},
  "reasoning": "User is hungry"
}
```
Hope that helps!"""

# =============================================================================
# FINAL RESPONSE GENERATION
# =============================================================================

VALID_FINAL_RESPONSE = """Based on your request, I found flights from NYC to LA and booked a hotel near LAX airport for you. Your confirmation numbers are FL123 and HT456."""

FINAL_RESPONSE_SHORT = "Done!"

FINAL_RESPONSE_EMPTY = ""

FINAL_RESPONSE_WITH_FORMATTING = """I've completed your request:

1. Found flights NYC to LA
2. Booked hotel near LAX

Have a great trip!"""

# =============================================================================
# TOOL SIMULATION RESPONSES
# =============================================================================

TOOL_SIMULATION_VALID_DICT = '{"flights": [{"id": "FL001", "price": 299, "airline": "TestAir"}]}'

TOOL_SIMULATION_VALID_LIST = '[{"id": 1, "name": "Hotel A"}, {"id": 2, "name": "Hotel B"}]'

TOOL_SIMULATION_VALID_STRING = '"Operation completed successfully"'

TOOL_SIMULATION_VALID_BOOL = "true"

TOOL_SIMULATION_VALID_NUMBER = "42"

TOOL_SIMULATION_EMPTY_DICT = "{}"

TOOL_SIMULATION_EMPTY_LIST = "[]"

TOOL_SIMULATION_ERROR = '{"error": "Invalid parameters provided"}'

TOOL_SIMULATION_INVALID_JSON = 'not valid json {'

TOOL_SIMULATION_EMPTY = ""

# =============================================================================
# DICTIONARIES FOR EASY ACCESS
# =============================================================================

VALID_RESPONSES = {
    "query_2_tools": VALID_QUERY_RESPONSE_2_TOOLS,
    "query_3_tools": VALID_QUERY_RESPONSE_3_TOOLS,
    "query_plain": VALID_QUERY_RESPONSE_PLAIN_JSON,
    "sequence_valid": VALID_SEQUENCE_RESPONSE,
    "step_valid": VALID_STEP_RESPONSE,
    "step_plain": VALID_STEP_RESPONSE_PLAIN_JSON,
    "final": VALID_FINAL_RESPONSE,
}

INVALID_QUERY_RESPONSES = {
    "wrong_count": QUERY_RESPONSE_WRONG_TOOL_COUNT,
    "too_many": QUERY_RESPONSE_TOO_MANY_TOOLS,
    "invalid_tool": QUERY_RESPONSE_INVALID_TOOL,
    "missing_query": QUERY_RESPONSE_MISSING_QUERY_FIELD,
    "missing_intent": QUERY_RESPONSE_MISSING_INTENT_FIELD,
    "missing_tools": QUERY_RESPONSE_MISSING_EXPECTED_TOOLS,
    "empty_tools": QUERY_RESPONSE_EMPTY_TOOLS,
    "empty_query": QUERY_RESPONSE_EMPTY_QUERY,
}

MALFORMED_JSON_RESPONSES = [
    ("unclosed_brace", MALFORMED_JSON_UNCLOSED_BRACE),
    ("unclosed_array", MALFORMED_JSON_UNCLOSED_ARRAY),
    ("missing_quote", MALFORMED_JSON_MISSING_QUOTE),
    ("trailing_comma", MALFORMED_JSON_TRAILING_COMMA),
    ("single_quotes", MALFORMED_JSON_SINGLE_QUOTES),
    ("extra_text", MALFORMED_JSON_EXTRA_TEXT_OUTSIDE),
    ("xml_instead", MALFORMED_JSON_XML_INSTEAD),
    ("incomplete_code_block", MALFORMED_JSON_IN_CODE_BLOCK),
]

STEP_SELECTION_RESPONSES = {
    "valid": VALID_STEP_RESPONSE,
    "valid_plain": VALID_STEP_RESPONSE_PLAIN_JSON,
    "with_placeholder": STEP_RESPONSE_WITH_PLACEHOLDER,
    "missing_tool": STEP_RESPONSE_MISSING_TOOL_NAME,
    "missing_args": STEP_RESPONSE_MISSING_ARGUMENTS,
    "missing_reasoning": STEP_RESPONSE_MISSING_REASONING,
    "invalid_tool": STEP_RESPONSE_INVALID_TOOL,
    "empty": STEP_RESPONSE_EMPTY,
    "not_json": STEP_RESPONSE_NOT_JSON,
    "whitespace": STEP_RESPONSE_ONLY_WHITESPACE,
    "extra_text": STEP_RESPONSE_EXTRA_TEXT_AROUND,
}
