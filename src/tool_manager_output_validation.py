"""Simulated tool output validation and output-format guidance."""

from __future__ import annotations

if __package__:
    from .tool_manager_types import (
        JsonModule,
        ToolManagerProtocol,
        is_object_list,
        is_string_object_dict,
        string_field,
    )
else:
    from tool_manager_types import (
        JsonModule,
        ToolManagerProtocol,
        is_object_list,
        is_string_object_dict,
        string_field,
    )

_LLM_VALIDATION_ERRORS: tuple[type[Exception], ...] = (Exception,)


def validate_tool_output(
    manager: ToolManagerProtocol,
    tool_name: str,
    output: object,
    expected_type: str,
    output_description: str,
    json_module: JsonModule,
) -> object:
    """Coerce basic types and optionally ask the LLM to validate semantics."""
    if expected_type == "unknown" or not expected_type:
        return output

    base_type = expected_type.split()[0].lower()
    if not _matches_type(output, base_type):
        try:
            output = _coerce_output(output, base_type, json_module)
        except (ValueError, TypeError, json_module.JSONDecodeError):
            print(f"Warning: Output type mismatch for {tool_name}. Expected {expected_type}, got {type(output).__name__}")

    if output_description and output_description != "Failed to predict output description":
        actual_output = (
            json_module.dumps(output, default=str)
            if is_string_object_dict(output) or is_object_list(output)
            else str(output)
        )
        validation_prompt = f"""You are an output validator. Given the following tool output and its expected description, determine if the output is plausible and matches the description.

Tool Name: {tool_name}
Expected Output Description: {output_description}
Actual Output: {actual_output}

Does the output plausibly match the description? Answer YES or NO, followed by a brief explanation.

Response format:
{{
    "VALID": "YES" or "NO",
    "REASON": "<brief explanation>"
}}"""
        try:
            if manager.llm is None:
                raise AttributeError("ToolManager has no LLM configured")
            response, _reasoning = manager.llm.json_output(
                prompt=validation_prompt, reasoning=False
            )
            if is_string_object_dict(response):
                is_valid = string_field(response, "VALID", "YES").upper() == "YES"
                if not is_valid:
                    reason = string_field(response, "REASON", "No reason provided")
                    print(f"Warning: LLM validation flagged output for {tool_name} as not matching description: {reason}")
                    return {
                        "error": f"Output validation failed: {reason}",
                        "error_type": "validation_failure",
                    }
        except _LLM_VALIDATION_ERRORS as error:
            print(f"Warning: Could not validate output for {tool_name}: {error}")
    return _opaque(output)


def _opaque(value: object) -> object:
    """Prevent container narrowing from leaking unknown element types."""
    return value


def _matches_type(output: object, base_type: str) -> bool:
    """Return whether an output matches one declared basic type."""
    if base_type == "dict":
        return isinstance(output, dict)
    if base_type == "list":
        return isinstance(output, list)
    if base_type == "string":
        return isinstance(output, str)
    if base_type == "integer":
        return isinstance(output, int)
    if base_type == "float":
        return isinstance(output, (int, float))
    if base_type == "number":
        return isinstance(output, (int, float))
    if base_type == "boolean":
        return isinstance(output, bool)
    return True


def _coerce_output(
    output: object, base_type: str, json_module: JsonModule
) -> object:
    """Apply the legacy best-effort output conversions."""
    if base_type in ("integer", "float", "number") and isinstance(output, str):
        return float(output) if "." in output else int(output)
    if base_type == "string":
        return str(output)
    if base_type == "boolean" and isinstance(output, str):
        return output.lower() in ("true", "yes", "1")
    if base_type == "list" and is_string_object_dict(output):
        return list(output.values())
    if base_type == "dict" and isinstance(output, str):
        return json_module.loads(output)
    return output


def build_output_guidance(output_type: str, output_description: str) -> str:
    """Build type examples and field requirements for simulated output."""
    guidance = ""
    base_type: str | None = None
    if output_type and output_type != "unknown":
        base_type = output_type.split()[0].lower()
        guidance += "\n\n=== REQUIRED OUTPUT TYPE ==="
        guidance += f"\nType: {output_type}"
        guidance += "\nCRITICAL: You MUST return output of exactly this type. No exceptions."
        type_examples = {
            "dict": '{"key": "value", "id": 123, "name": "example"}',
            "list": '[{"item": 1}, {"item": 2}, {"item": 3}]',
            "string": '"your string value here"',
            "integer": "42",
            "float": "3.14",
            "number": "42 or 3.14",
            "boolean": "true or false",
        }
        if base_type in type_examples:
            guidance += f"\nExample of correct {output_type} output: {type_examples[base_type]}"

    if output_description and output_description != "Failed to predict output description":
        guidance += f"\n\n=== OUTPUT DESCRIPTION ===\n{output_description}\n"
        guidance += "\nYOUR OUTPUT MUST SATISFY THIS DESCRIPTION:"
        guidance += "\n- Study the description carefully and include ALL fields/values mentioned"
        guidance += "\n- The output content must realistically match what the description promises"
        guidance += "\n- For dict outputs, ensure all keys mentioned in the description are present"
        guidance += "\n- Do NOT invent fields that aren't mentioned in the description"
        desc_lower = output_description.lower()
        if "message_id" in desc_lower or "message id" in desc_lower:
            guidance += "\n- MUST include 'message_id' field"
        if "success" in desc_lower:
            guidance += "\n- MUST include 'success' field (boolean)"
        if "timestamp" in desc_lower:
            guidance += "\n- MUST include 'timestamp' field with ISO format"
        if "status" in desc_lower:
            guidance += "\n- MUST include 'status' field"
        if "id" in desc_lower:
            if base_type is None:
                raise UnboundLocalError(
                    "cannot access local variable 'base_type' where it is not associated with a value"
                )
            if base_type == "dict":
                guidance += "\n- MUST include an 'id' or identifier field"
    return guidance


def get_output_format_instructions(
    output_type: str, output_description: str = ""
) -> str:
    """Return exact output formatting instructions for an output type."""
    base_type = output_type.split()[0].lower() if output_type else "unknown"
    desc_check = ""
    if output_description and output_description != "Failed to predict output description":
        desc_check = f"""

MANDATORY OUTPUT CONTENT CHECK:
- Your output MUST match this description: {output_description}
- Include ALL fields mentioned in the description
- Values must be realistic and appropriate for the described output"""

    instructions = {
        "dict": f"""REQUIREMENTS:
1. CRITICAL: Return ONLY a JSON OBJECT (dictionary) - NOT a string, NOT a list, NOT a number
2. The JSON object MUST have key-value pairs: {{"field1": "value1", "field2": "value2"}}
3. Example: {{"user_id": "U12345", "name": "John", "status": "active"}}
4. NO markdown formatting, NO code blocks (no ```), NO explanations - output ONLY the raw JSON object
5. For error cases ONLY, return: {{"error": "description", "error_description": "details"}}{desc_check}""",
        "list": f"""REQUIREMENTS:
1. CRITICAL: Return ONLY a JSON ARRAY (list) - NOT a dict, NOT a string, NOT a number
2. The JSON array MUST be wrapped in square brackets: [item1, item2, item3]
3. Example: [{{"id": 1}}, {{"id": 2}}] or ["item1", "item2"]
4. NO markdown formatting, NO code blocks (no ```), NO explanations - output ONLY the raw JSON array
5. For error cases ONLY, return: {{"error": "description", "error_description": "details"}}{desc_check}""",
        "string": f"""REQUIREMENTS:
1. CRITICAL: Return ONLY a PLAIN STRING value - NOT a JSON object, NOT a JSON array, NOT a number
2. Example: "U12345" or "Operation completed successfully" or just Hello World (without quotes is also acceptable)
3. The output should be the string value itself, optionally wrapped in quotes
4. NO markdown formatting, NO code blocks (no ```), NO explanations - output ONLY the raw string
5. CRITICAL: Do NOT wrap the string in a dict like {{"result": "string"}} - return ONLY the string
6. For error cases ONLY, return a JSON object: {{"error": "description"}}{desc_check}""",
        "number": f"""REQUIREMENTS:
1. CRITICAL: Return ONLY a NUMERIC value (integer or float) - NOT a string, NOT a dict, NOT a list
2. Example: 42 or 3.14
3. NO quotes around the number - output the raw number only
4. NO markdown formatting, NO code blocks, NO explanations - output ONLY the raw number
5. For error cases ONLY, return: {{"error": "description", "error_description": "details"}}{desc_check}""",
        "boolean": f"""REQUIREMENTS:
1. CRITICAL: Return ONLY a BOOLEAN value: true or false (lowercase, no quotes) - NOT a string "true"
2. Example: true (NOT "true", NOT {{"result": true}})
3. NO markdown formatting, NO code blocks, NO explanations - output ONLY the raw boolean
4. For error cases ONLY, return: {{"error": "description", "error_description": "details"}}{desc_check}""",
    }
    if base_type in ("integer", "float"):
        base_type = "number"
    return instructions.get(
        base_type,
        f"""REQUIREMENTS:
1. CRITICAL: Return a value matching the REQUIRED OUTPUT TYPE specified above
2. NO markdown formatting, NO code blocks, NO explanations - output ONLY the raw value
3. For error cases, return: {{"error": "description", "error_description": "details"}}{desc_check}""",
    )


def is_output_valid(
    output: object, expected_type: str, expected_description: str
) -> bool:
    """Check whether an output has the declared basic Python type."""
    _ = expected_description
    if expected_type == "unknown" or not expected_type:
        return True
    base_type = expected_type.split()[0].lower()
    if base_type == "integer":
        return isinstance(output, int) and not isinstance(output, bool)
    if base_type == "float":
        return isinstance(output, float)
    if base_type == "number":
        return isinstance(output, (int, float)) and not isinstance(output, bool)
    return _matches_type(output, base_type)


def get_default_output(output_type: str) -> object:
    """Return the legacy simulation fallback for an output type."""
    defaults: dict[str, object] = {
        "dict": {"status": "success", "message": "Operation completed"},
        "list": [],
        "string": "success",
        "integer": 0,
        "float": 0.0,
        "number": 0,
        "boolean": True,
    }
    base_type = output_type.split()[0].lower() if output_type else "unknown"
    return defaults.get(base_type, {"status": "completed"})
