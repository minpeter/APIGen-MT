"""Identity coherence and initial API-state preparation."""

import json
from typing import Protocol

from pydantic import ValidationError

from step_by_step_models import QueryGenerationResult, StateAdjustmentResponse
from step_by_step_protocols import (
    StepByStepMixinBase,
    is_object_map,
    is_tool_instance,
)
from step_by_step_state_helpers import (
    instance_value,
    prepare_file_state,
    prepare_message_state,
)

_IDENTITY_ATTRIBUTES = {
    "first_name",
    "last_name",
    "current_user",
    "username",
    "user_id",
}


class StatePreparationMixin(StepByStepMixinBase, Protocol):
    def _ensure_user_identity_coherence(self, text: str) -> bool:
        """Synchronize a user named in the query across API instances."""
        current_state = self.tool_manager.get_api_state()
        text_lower = text.lower()
        user_candidates: set[str] = set()
        for state in current_state.values():
            user_map = state.get("user_map")
            if is_object_map(user_map):
                user_candidates.update(
                    username
                    for username in user_map
                    if username.lower() in text_lower
                )

        if not user_candidates:
            return False
        primary_user = next(iter(user_candidates))

        adjusted = False
        for class_key in current_state:
            instance = self.tool_manager.python_tool_instances.get(class_key)
            if not is_tool_instance(instance):
                continue
            for attribute in _IDENTITY_ATTRIBUTES:
                current_value = instance_value(instance, attribute)
                if (
                    isinstance(current_value, str)
                    and current_value
                    and current_value != primary_user
                ):
                    print(
                        f"  Sync {class_key}.{attribute}: {current_value} -> {primary_user}"
                    )
                    instance.__setattr__(attribute, primary_user)
                    adjusted = True
        return adjusted

    def _stage1_5_adjust_initial_state(
        self,
        query_result: QueryGenerationResult,
    ) -> bool:
        """Ask the LLM for minimal state changes and apply them."""
        current_state = self.tool_manager.get_api_state()
        relevant_class_keys = {
            class_key
            for tool_name in query_result.expected_tools
            if (
                class_key := self.tool_manager.api_name_to_class_key.get(tool_name)
            )
        }
        relevant_state = {
            key: value
            for key, value in current_state.items()
            if key in relevant_class_keys
        }

        schema_sections: list[str] = []
        for tool_name in query_result.expected_tools:
            schema = self.tool_manager.get_tool_schema(tool_name)
            if not schema:
                continue
            schema_sections.append(
                f"\n- {tool_name}: {schema.get('description', '')}\n  Parameters: "
                + f"{json.dumps(schema.get('parameters', {}), indent=2, default=str)[:500]}\n"
            )

        prompt = "".join(
            (
                """You are preparing the initial state of API instances so that tool calls execute successfully.

=== USER QUERY ===
""",
                query_result.query,
                "\n\n=== EXPECTED TOOL SEQUENCE ===\n",
                json.dumps(query_result.expected_tools),
                "\n\n=== TOOL SCHEMAS ===\n",
                "".join(schema_sections),
                "\n\n=== CURRENT API STATE ===\n",
                json.dumps(relevant_state, indent=2, default=str)[:6000],
                """

=== RULES ===
- To add user: user_map.Username = "USR015"
- To append to list: "APPEND:array_key": value
- Set fields directly: "field_name": "value"
- Never do string operations like "APPEND:foo = bar"
- MINIMAL changes only

=== EXAMPLES ===
Add a new entry to a key-value map:
{"modifications": {"api_name": {"map_key.NewName": "USR015"}}, "reasoning": "..."}

Add new item to a queue:
{"modifications": {"api_name": {"APPEND:queue_key": {"id": 1234}}}, "reasoning": "..."}

No changes needed:
{"modifications": {}, "reasoning": "no changes needed"}

=== RESPONSE ===
Respond only with valid JSON in one of these formats""",
            )
        )

        try:
            response_text = self._safe_llm_generate(
                [{"role": "user", "content": prompt}]
            ).strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                response_text = response_text[start:end]
            result = StateAdjustmentResponse.model_validate_json(response_text)
        except (RuntimeError, ValidationError, ValueError) as exc:
            print(f" ✗ Failed to parse state adjustment response: {exc}")
            return False

        if not result.modifications:
            print(f" No modifications needed: {result.reasoning}")
            return False
        print(
            " Modifications requested: "
            + json.dumps(result.modifications, indent=2, default=str)[:1000]
        )
        print(f" Reasoning: {result.reasoning}")
        applied = self._apply_state_modifications(result.modifications)
        applied = prepare_message_state(self.tool_manager, query_result, applied)
        prepare_file_state(self.tool_manager, query_result, applied)

        if applied <= 0:
            print(" No modifications could be applied")
            return False
        print(f" ✓ Applied {applied} state modifications")
        return True
