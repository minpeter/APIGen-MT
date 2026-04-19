"""Mock LLM client for testing.

This module provides a configurable mock LLM client that can return
predefined responses for different prompt types.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union


class MockLLMClient:
    """Mock LLM client with fine-grained response control.

    This mock client can be configured with specific responses for different
    prompt types, making it useful for testing LLM-dependent logic without
    making actual API calls.

    Attributes:
        responses: List of responses to return in sequence
        response_map: Mapping of prompt patterns to responses
        call_count: Number of times generate() was called
        captured_prompts: List of all prompts sent to the client
    """

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        response_map: Optional[Dict[str, str]] = None,
    ):
        """Initialize the mock LLM client.

        Args:
            responses: List of responses to return in sequence
            response_map: Dictionary mapping prompt patterns to responses
        """
        self.responses = responses or []
        self.response_map = response_map or {}
        self.call_count = 0
        self.captured_prompts: List[List[Dict[str, str]]] = []
        self.captured_kwargs: List[Dict[str, Any]] = []
        self.should_fail: bool = False
        self.fail_after: Optional[int] = None
        self.exception_to_raise: Optional[Exception] = None

    def reset(self):
        """Reset the mock client state."""
        self.call_count = 0
        self.captured_prompts.clear()
        self.captured_kwargs.clear()
        self.should_fail = False
        self.fail_after = None
        self.exception_to_raise = None

    def set_responses(self, responses: List[str]):
        """Set a sequence of responses to return.

        Args:
            responses: List of response strings
        """
        self.responses = responses
        self.call_count = 0

    def add_response(self, response: str):
        """Add a single response to the sequence.

        Args:
            response: Response string to add
        """
        self.responses.append(response)

    def set_response_for_pattern(self, pattern: str, response: str):
        """Set a response for a specific prompt pattern.

        Args:
            pattern: Pattern to match in prompts
            response: Response to return when pattern is found
        """
        self.response_map[pattern] = response

    def set_should_fail(self, fail: bool = True, after_calls: Optional[int] = None):
        """Configure the client to fail.

        Args:
            fail: Whether to raise an exception
            after_calls: Fail only after this many calls
        """
        self.should_fail = fail
        self.fail_after = after_calls
        if fail and not self.exception_to_raise:
            self.exception_to_raise = RuntimeError("Mock LLM error")

    def set_exception(self, exception: Exception):
        """Set a custom exception to raise.

        Args:
            exception: Exception instance to raise
        """
        self.exception_to_raise = exception
        self.should_fail = True

    def _extract_prompt_text(self, messages: List[Dict[str, str]]) -> str:
        """Extract text content from messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Concatenated text content of all messages
        """
        return "\n".join(
            msg.get("content", "") for msg in messages if msg.get("content")
        )

    def _match_pattern_response(self, prompt_text: str) -> Optional[str]:
        """Check if prompt matches any configured pattern.

        Args:
            prompt_text: The full prompt text

        Returns:
            Matching response or None if no match
        """
        for pattern, response in self.response_map.items():
            if pattern in prompt_text.lower():
                return response
        return None

    def generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate a response (mock implementation).

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional arguments (captured but not used)

        Returns:
            str: The mock response text

        Raises:
            RuntimeError: If should_fail is set and conditions are met
        """
        self.captured_prompts.append(messages)
        self.captured_kwargs.append(kwargs)

        # Check if we should raise an exception
        if self.should_fail:
            if self.fail_after is None or self.call_count >= self.fail_after:
                if self.exception_to_raise:
                    raise self.exception_to_raise

        # Try pattern matching first
        prompt_text = self._extract_prompt_text(messages)
        pattern_response = self._match_pattern_response(prompt_text)
        if pattern_response is not None:
            self.call_count += 1
            return pattern_response

        # Fall back to sequence responses
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response

        # Default empty response
        self.call_count += 1
        return "{}"

    def chat(
        self, messages: List[Dict[str, str]], kwargs: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Chat interface returning response and reasoning.

        Args:
            messages: List of message dictionaries
            kwargs: Additional arguments

        Returns:
            Tuple of (response_text, reasoning_text)
        """
        response = self.generate(messages, **kwargs)
        # Extract reasoning if present (between think tags)
        reasoning_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        reasoning = reasoning_match.group(1) if reasoning_match else ""
        # Remove think tags from response
        clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        return clean_response, reasoning

    def json_output(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Any = None,
        reasoning: bool = True,
    ) -> Tuple[Any, str]:
        """Generate JSON output (mock implementation).

        Args:
            prompt: The prompt text
            system_prompt: Optional system prompt
            schema: Optional Pydantic schema
            reasoning: Whether to include reasoning

        Returns:
            Tuple of (parsed_json, reasoning_text)
        """
        messages = [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": prompt},
        ]
        response = self.generate(messages)
        reasoning_text = ""

        # Try to extract JSON from the response
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        elif response.strip().startswith("{"):
            # Extract JSON object
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end] if end > start else response
        else:
            json_str = response

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Return the raw response if JSON parsing fails
            parsed = {"raw_response": response, "error": str(e)}
            reasoning_text = f"Failed to parse JSON: {e}"

        return parsed, reasoning_text

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get mock usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_prompt_tokens": self.call_count * 100,
            "total_completion_tokens": self.call_count * 50,
            "total_tokens": self.call_count * 150,
            "num_calls": self.call_count,
            "mean_prompt_tokens_per_call": 100,
            "mean_completion_tokens_per_call": 50,
        }

    def reset_usage_stats(self):
        """Reset usage statistics (no-op for mock)."""
        self.call_count = 0


class MockLLMClientBuilder:
    """Builder for creating configured MockLLMClient instances.

    This builder helps create mock clients with specific response sequences
    for common testing scenarios.
    """

    def __init__(self):
        self.client = MockLLMClient()

    def for_successful_2_step_datapoint(self) -> "MockLLMClientBuilder":
        """Configure for a successful 2-step datapoint generation."""
        from tests.mocks.mock_llm_responses import (
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
            VALID_STEP_RESPONSE,
            VALID_FINAL_RESPONSE,
        )

        self.client.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,  # generate_user_query
            VALID_SEQUENCE_RESPONSE,          # validate_expected_tools
            VALID_STEP_RESPONSE,              # step 1
            VALID_STEP_RESPONSE,              # step 2
            VALID_FINAL_RESPONSE,             # generate final response
        ])
        return self

    def for_failed_query_generation(self) -> "MockLLMClientBuilder":
        """Configure for query generation that fails after retries."""
        from tests.mocks.mock_llm_responses import (
            QUERY_RESPONSE_WRONG_TOOL_COUNT,
            QUERY_RESPONSE_INVALID_TOOL,
            MALFORMED_JSON_UNCLOSED_BRACE,
        )

        self.client.set_responses([
            QUERY_RESPONSE_WRONG_TOOL_COUNT,
            QUERY_RESPONSE_INVALID_TOOL,
            MALFORMED_JSON_UNCLOSED_BRACE,
        ])
        return self

    def for_step_generation_failure(self) -> "MockLLMClientBuilder":
        """Configure for step generation failure."""
        from tests.mocks.mock_llm_responses import (
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
            STEP_RESPONSE_EMPTY,
        )

        self.client.set_responses([
            VALID_QUERY_RESPONSE_2_TOOLS,
            VALID_SEQUENCE_RESPONSE,
            STEP_RESPONSE_EMPTY,
        ])
        return self

    def with_pattern_responses(self, pattern_map: Dict[str, str]) -> "MockLLMClientBuilder":
        """Add pattern-based responses.

        Args:
            pattern_map: Dictionary mapping patterns to responses
        """
        for pattern, response in pattern_map.items():
            self.client.set_response_for_pattern(pattern, response)
        return self

    def with_exception(self, exception: Exception) -> "MockLLMClientBuilder":
        """Configure to raise an exception.

        Args:
            exception: Exception to raise
        """
        self.client.set_exception(exception)
        return self

    def build(self) -> MockLLMClient:
        """Build and return the configured mock client.

        Returns:
            Configured MockLLMClient instance
        """
        return self.client
