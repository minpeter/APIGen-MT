"""Tool-schema and API-state context for blueprint generation."""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.multi_turn_protocols import ApiState, GeneratorMixinBase
else:
    from multi_turn_protocols import GeneratorMixinBase

from multi_turn_protocols import (
    DynamicCallable,
    is_dynamic_attribute_source,
    is_object_dict,
    string_value,
)


class _ParameterView(Protocol):
    @property
    def annotation(self) -> object: ...

    @property
    def default(self) -> object: ...


def _parameter_view(parameter: inspect.Parameter) -> _ParameterView:
    return parameter


_TOOL_INSPECTION_ERRORS = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    BufferError,
    EOFError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


class BlueprintContextMixin(GeneratorMixinBase):
    """Inspect tool outputs for blueprint prompt construction."""

    def _get_tool_output_fields(
        self,
        category: str | None = None,
    ) -> dict[str, list[str]]:
        """Extract output fields by calling Python tools with minimal inputs."""
        if not self._python_tools_available:
            return {}

        result: dict[str, list[str]] = {}
        api_names: list[tuple[str, str]] = []
        for api_name, class_key in self.tool_manager.api_name_to_class_key.items():
            if category:
                tool_category = self.tool_manager.get_tool_category(api_name)
                if tool_category != category:
                    continue
            api_names.append((api_name, class_key))

        for api_name, class_key in api_names:
            instance = self.tool_manager.python_tool_instances.get(class_key)
            if not is_dynamic_attribute_source(instance):
                continue
            try:
                method = instance.__getattribute__(api_name)
            except AttributeError:
                continue
            if not isinstance(method, DynamicCallable):
                continue

            try:
                signature = inspect.signature(method)
                bound: list[object] = []
                for parameter_name, raw_parameter in signature.parameters.items():
                    if parameter_name == "self":
                        continue
                    parameter = _parameter_view(raw_parameter)
                    required = parameter.default is inspect.Parameter.empty
                    if (
                        parameter.annotation is int
                        or parameter.annotation is float
                    ) and required:
                        bound.append(1)
                    elif parameter.annotation is str and required:
                        lowered_name = parameter_name.lower()
                        if "city" in lowered_name or "location" in lowered_name:
                            bound.append("New York")
                        elif "date" in lowered_name:
                            bound.append("2025-03-15")
                        elif "token" in lowered_name:
                            bound.append("DUMMY_TOKEN")
                        elif any(
                            word in lowered_name
                            for word in ("card", "number", "id")
                        ):
                            bound.append("12345")
                        elif "message" in lowered_name or "name" in lowered_name:
                            bound.append("Test")
                        elif "currency" in lowered_name:
                            bound.append("USD")
                        elif "type" in lowered_name:
                            bound.append("basic")
                        elif any(
                            word in lowered_name
                            for word in ("cost", "balance", "value", "limit")
                        ):
                            bound.append(100.0)
                        else:
                            bound.append("x")
                    elif parameter.annotation is bool:
                        bound.append(True)

                output = method(*bound)
                if is_object_dict(output):
                    result[api_name] = sorted(
                        key for key in output if isinstance(key, str)
                    )
                else:
                    result[api_name] = []
            except _TOOL_INSPECTION_ERRORS:
                result[api_name] = ["success", "message", "id", "result", "error"]

        return result


def build_blueprint_context(
    generator: GeneratorMixinBase,
    focus_category: str | None,
    initial_api_state: ApiState | None,
) -> tuple[str, str, dict[str, list[str]], str, str]:
    """Build the schema, output-field, state, and credential prompt sections."""
    tools_json = generator.tool_manager.get_tools_json_schema()
    if focus_category:
        tools_json = [
            tool
            for tool in tools_json
            if string_value(tool, "category") == focus_category
        ]

    tools_str = json.dumps(tools_json, indent=2, ensure_ascii=False, default=str)
    output_fields_lines: list[str] = []
    output_fields_validation_map: dict[str, list[str]] = {}
    for tool in tools_json:
        name = string_value(tool, "name") or string_value(tool, "api_name")
        schema = tool.get("output_schema")
        properties: dict[object, object] = {}
        if is_object_dict(schema):
            raw_properties = schema.get("properties")
            if is_object_dict(raw_properties):
                properties = raw_properties

        property_names = [key for key in properties if isinstance(key, str)]
        if property_names:
            output_fields_lines.append(f"- {name}: {', '.join(property_names)}\n")
        else:
            output_fields_lines.append(f"- {name}\n")

        if name:
            output_fields_validation_map[name] = property_names or [
                "success",
                "message",
                "id",
                "result",
                "error",
            ]

    focus_class_keys: set[str] = set()
    if focus_category:
        for api_name, class_key in generator.tool_manager.api_name_to_class_key.items():
            if generator.tool_manager.get_tool_category(api_name) == focus_category:
                focus_class_keys.add(class_key)

    credential_lines: list[str] = []
    initial_state_lines: list[str] = []
    if initial_api_state:
        for class_key, state in initial_api_state.items():
            if focus_category and class_key not in focus_class_keys:
                continue

            state_summary = json.dumps(state, indent=2, default=str)
            initial_state_lines.append(f"\n{class_key}:\n{state_summary}")

            if all(
                key in state
                for key in ("client_id", "client_secret", "refresh_token")
            ):
                credential_lines.append(
                    f"\nCredential format: {state['client_id']}/"
                    + f"{state['client_secret']}/{state['refresh_token']}"
                )
            cards = state.get("credit_card_list")
            if is_object_dict(cards):
                card_ids = [key for key in cards if isinstance(key, str)]
                if card_ids:
                    credential_lines.append(
                        f"\nAvailable card IDs: {', '.join(card_ids)}"
                    )
            users = state.get("user_map")
            if is_object_dict(users):
                user_ids = [key for key in users if isinstance(key, str)]
                if user_ids:
                    credential_lines.append(
                        f"\nAvailable user IDs: {', '.join(user_ids[:10])}"
                    )
            if "account_type" in state and "balance" in state:
                credential_lines.append(f"\nAccount balance: {state.get('balance')}")
            if "username" in state and "password" in state:
                credential_lines.append(
                    f"\nCredentials: {state['username']}/{state['password']}"
                )

    return (
        tools_str,
        "".join(output_fields_lines),
        output_fields_validation_map,
        "".join(initial_state_lines),
        "".join(credential_lines),
    )
