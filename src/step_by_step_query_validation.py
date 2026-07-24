"""Expected-tool validation and query examples."""

import json
from typing import Protocol, override

from step_by_step_models import ValidationResponse
from step_by_step_protocols import StepByStepMixinBase


class QueryValidationMixin(StepByStepMixinBase, Protocol):
    @override
    def validate_expected_tools(
        self,
        query: str,
        expected_tools: list[str],
        intent: str,
    ) -> tuple[bool, str]:
        if len(expected_tools) != self.target_num_actions:
            return (
                False,
                f"Tool count mismatch: expected {self.target_num_actions}, received {len(expected_tools)}",
            )

        prompt = f"""You are validating a tool sequence plan for a user query.

User Query: {query}
Intent: {intent}

Planned Tool Sequence: {json.dumps(expected_tools)}

Tool Schemas:
{self._get_tool_schemas_str(expected_tools)}

Evaluate if the sequence logically fits the query intent.

Respond with JSON:
{{
    "is_valid": true/false,
    "issues": ["list of issues if any"]
}}"""

        try:
            response_text = self._safe_llm_generate(
                [{"role": "user", "content": prompt}],
                llm=self.judge,
            ).strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            else:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]

            result = ValidationResponse.model_validate_json(response_text)
            if not result.is_valid:
                return (
                    False,
                    f"Tool sequence validation failed: {'; '.join(result.issues)}",
                )
            return True, ""
        except (RuntimeError, ValueError) as exc:
            return False, f"Validation error: {exc}"

    @override
    def _get_example_queries(self) -> str:
        """Return few-shot examples of valid queries with tool sequences."""
        examples: list[dict[str, object]] = [
            {
                "num_tools": 2,
                "query": "List items in the current directory, then create a new subdirectory.",
                "intent": "User wants to see files and create a folder",
                "expected_tools": ["ls", "mkdir"],
            },
            {
                "num_tools": 2,
                "query": "Display report.txt, then search for the word 'error' in it.",
                "intent": "User wants to read a file and find specific text",
                "expected_tools": ["cat", "grep"],
            },
            {
                "num_tools": 3,
                "query": "Create notes.txt, write 'Hello World', then display it.",
                "intent": "User wants to create and populate a file",
                "expected_tools": ["touch", "echo", "cat"],
            },
        ]

        result: list[str] = []
        for index, example in enumerate(examples, 1):
            result.extend(
                (
                    f"\n=== EXAMPLE {index} ({example['num_tools']} tools) ===",
                    f"Query: \"{example['query']}\"",
                    f"Intent: {example['intent']}",
                    f"Expected tools: {example['expected_tools']}",
                )
            )
        return "\n".join(result)
