"""Stage 0 orchestration for multi-turn dialog blueprints."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import TypeAdapter, ValidationError

if TYPE_CHECKING:
    from src.multi_turn_models import DialogBlueprint
    from src.multi_turn_protocols import ApiState, GeneratorMixinBase
else:
    from multi_turn_protocols import GeneratorMixinBase

from multi_turn_blueprint_context import build_blueprint_context
from multi_turn_blueprint_prompt import build_blueprint_prompt
from multi_turn_blueprint_validation import validate_blueprint_turns
from multi_turn_protocols import get_public_facade, string_list, string_value

_RESPONSE_ADAPTER = TypeAdapter(dict[str, object])
_TURNS_ADAPTER = TypeAdapter(list[dict[str, object]])


class BlueprintGenerationMixin(GeneratorMixinBase):
    """Generate and validate a concrete multi-turn dialog blueprint."""

    @override
    def _stage0_generate_blueprint(
        self,
        focus_category: str | None = None,
        initial_api_state: ApiState | None = None,
        max_retries: int = 3,
    ) -> DialogBlueprint | None:
        """Generate a specific dialog blueprint with concrete entities."""
        (
            tools_str,
            output_fields_str,
            output_fields_validation_map,
            initial_state_context,
            credential_context,
        ) = build_blueprint_context(self, focus_category, initial_api_state)

        facade = get_public_facade()
        prompt = build_blueprint_prompt(
            self,
            focus_category,
            tools_str,
            output_fields_str,
            initial_state_context,
            credential_context,
            facade.get_domain_hints,
        )

        accumulated_feedback = ""
        for attempt in range(max(1, max_retries)):
            try:
                if accumulated_feedback:
                    prompt_with_feedback = (
                        prompt
                        + "\n\n=== PREVIOUS ATTEMPT FEEDBACK ===\n"
                        + accumulated_feedback
                        + "\n=== END FEEDBACK ===\n"
                    )
                else:
                    prompt_with_feedback = prompt

                response = self._safe_llm_generate(
                    [{"role": "user", "content": prompt_with_feedback}]
                )
                response_text = response.strip()
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]

                result = _RESPONSE_ADAPTER.validate_json(response_text)
                turns = _TURNS_ADAPTER.validate_python(result.get("turns", []))
                if not turns or len(turns) != self.num_turns:
                    accumulated_feedback = (
                        f"Expected {self.num_turns} turns, got {len(turns)}. "
                        + f"Please generate exactly {self.num_turns} turns."
                    )
                    print(f"  ✗ {accumulated_feedback}")
                    continue

                all_tools_valid, _validation_errors = validate_blueprint_turns(
                    self,
                    turns,
                    focus_category,
                    output_fields_validation_map,
                )

                # Preserve the original structural-error fail-fast behavior.
                if not all_tools_valid:
                    break

                all_tools_valid = all(
                    self.tool_manager.tool_exists(tool_name)
                    for turn in turns
                    for tool_name in string_list(turn, "expected_tools")
                )
                if not all_tools_valid:
                    accumulated_feedback = (
                        "Some expected_tools are invalid. Please use only valid "
                        + "tool names from the provided list."
                    )
                    print(f"  ✗ {accumulated_feedback}")
                    continue

                print("  Verifying tool-query capability match...")
                capabilities_valid, capability_issues = (
                    self._verify_blueprint_capabilities(
                        turns,
                        focus_category,
                        initial_api_state,
                    )
                )
                if not capabilities_valid:
                    capability_feedback = (
                        "\n".join(capability_issues)
                        if capability_issues
                        else "Tool capabilities don't match query intents"
                    )
                    accumulated_feedback = (
                        f"Capability mismatch:\n{capability_feedback}\n\n"
                        + "Please regenerate with queries that match tool capabilities."
                    )
                    print(f"  ✗ {capability_feedback[:200]}...")
                    continue

                entity_issues = self._validate_posting_api_entities(
                    turns,
                    initial_api_state,
                )
                if entity_issues:
                    entity_feedback = "\n".join(entity_issues)
                    accumulated_feedback = (
                        f"Entity reference errors:\n{entity_feedback}\n\n"
                        + "Please regenerate with valid entity references from the API state."
                    )
                    print(f"  ✗ {accumulated_feedback[:200]}...")
                    continue

                vehicle_issues = self._validate_vehicle_control_queries(
                    turns,
                    initial_api_state,
                )
                if vehicle_issues:
                    vehicle_feedback = "\n".join(vehicle_issues)
                    accumulated_feedback = (
                        f"Vehicle state errors:\n{vehicle_feedback}\n\n"
                        + "Please regenerate with coherent vehicle state."
                    )
                    print(f"  ✗ {accumulated_feedback[:200]}...")
                    continue

                overall_task = string_value(result, "overall_task")
                print(f" ✓ Blueprint generated: {overall_task[:100]}")
                return facade.DialogBlueprint(
                    overall_task=overall_task,
                    num_turns=self.num_turns,
                    turns=turns,
                )
            except (KeyError, TypeError, ValueError, ValidationError) as error:
                accumulated_feedback = (
                    f"JSON parse error: {error}. Please return valid JSON."
                )
                print(f"  ✗ Attempt {attempt + 1}: {error}")

        print("  ✗ Failed to generate valid blueprint after 3 attempts")
        return None
