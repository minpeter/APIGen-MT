"""Declared output-type verification for step-by-step tool calls."""

from typing import Protocol, override

from step_by_step_models import ObjectMap
from step_by_step_protocols import StepByStepMixinBase, is_object_map


class OutputVerificationMixin(StepByStepMixinBase, Protocol):
    @staticmethod
    def _is_dict_wrapped_primitive(
        output: object,
        expected_type_lower: str,
    ) -> bool:
        """Check whether a mapping wraps a declared primitive value."""
        if not is_object_map(output) or not output:
            return False
        if "list" in expected_type_lower:
            return any(isinstance(value, list) for value in output.values())
        return any(
            (expected_type_lower == "float" and isinstance(value, float))
            or (
                expected_type_lower == "number"
                and isinstance(value, (int, float))
            )
            or (
                expected_type_lower == "integer"
                and isinstance(value, int)
            )
            or (
                expected_type_lower == "string"
                and isinstance(value, str)
            )
            or (
                expected_type_lower == "boolean"
                and isinstance(value, bool)
            )
            for value in output.values()
        )

    @override
    def verify_output_consistency(
        self,
        tool_name: str,
        step_number: int,
        output: object,
        expected_type: str,
        expected_description: str,
    ) -> ObjectMap:
        """Verify that a tool output matches its declared type."""
        del expected_description
        if output is None:
            return {
                "tool_name": tool_name,
                "step_number": step_number,
                "output_type_matches": False,
                "issues": ["Output is None"],
            }

        issues: list[str] = []
        output_type_matches = True
        if expected_type:
            expected = expected_type.lower()
            output_type = type(output).__name__.lower()
            type_compatible = (
                ("dict" in expected and isinstance(output, dict))
                or ("list" in expected and isinstance(output, list))
                or ("string" in expected and isinstance(output, str))
                or (
                    "number" in expected
                    and isinstance(output, (int, float))
                )
                or ("bool" in expected and isinstance(output, bool))
                or expected in output_type
                or self._is_dict_wrapped_primitive(output, expected)
            )
            if not type_compatible:
                output_type_matches = False
                issues.append(
                    f"Type mismatch: expected {expected_type}, got {output_type}"
                )

        return {
            "tool_name": tool_name,
            "step_number": step_number,
            "output_type_matches": output_type_matches,
            "issues": issues,
        }
