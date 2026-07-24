"""LLM-backed tool simulation and prompt construction."""

from __future__ import annotations

import datetime

if __package__:
    from .tool_manager_types import (
        JsonModule,
        ToolManagerProtocol,
        ToolParams,
        ToolSchema,
        get_callable_attribute,
        string_field,
    )
else:
    from tool_manager_types import (
        JsonModule,
        ToolManagerProtocol,
        ToolParams,
        ToolSchema,
        get_callable_attribute,
        string_field,
    )

_SIMULATION_ERRORS: tuple[type[Exception], ...] = (Exception,)


def virtual_tool_executor(
    manager: ToolManagerProtocol,
    tool_name: str,
    params: ToolParams,
    schema: ToolSchema,
) -> object:
    """Simulate a tool call with validation and the legacy retry policy."""
    output_type = string_field(schema, "output_type", "unknown")
    output_description = string_field(schema, "output_description")
    output_guidance = _required_string(
        get_callable_attribute(manager, "_build_output_guidance")(
            output_type, output_description
        ),
        "_build_output_guidance",
    )
    prompt = _required_string(
        get_callable_attribute(manager, "_build_simulation_prompt")(
            tool_name=tool_name,
            params=params,
            schema=schema,
            output_guidance=output_guidance,
        ),
        "_build_simulation_prompt",
    )

    max_retries = 2
    validated_response: object = None
    for attempt in range(max_retries + 1):
        try:
            if manager.llm is None:
                raise AttributeError("ToolManager has no LLM configured")
            response, _reasoning = manager.llm.json_output(
                prompt=prompt, reasoning=True
            )
            validated_response = get_callable_attribute(
                manager, "_validate_tool_output"
            )(tool_name, response, output_type, output_description)
            valid = get_callable_attribute(manager, "_is_output_valid")(
                validated_response, output_type, output_description
            )
            if not isinstance(valid, bool):
                raise TypeError("_is_output_valid must return bool")
            if valid:
                return validated_response
            if attempt < max_retries:
                retry_number = attempt + 1
                print(f"    Tool simulation validation failed for {tool_name}, retrying ({retry_number}/{max_retries})...")
                correction = f"""

=== CORRECTION NEEDED ===
Your previous output did not match the expected type '{output_type}'.
Please regenerate ensuring the output is of type {output_type} and matches the description.
"""
                prompt += correction
        except _SIMULATION_ERRORS as error:
            print(f"    Error simulating tool {tool_name}: {error}")
            if attempt < max_retries:
                continue
            return get_callable_attribute(manager, "_get_default_output")(
                output_type
            )

    return validated_response


def build_simulation_prompt(
    manager: ToolManagerProtocol,
    tool_name: str,
    params: ToolParams,
    schema: ToolSchema,
    output_guidance: str,
    current_datetime: datetime.datetime,
    json_module: JsonModule,
) -> str:
    """Build the enhanced simulation prompt with type-specific examples."""
    examples = _required_string(
        get_callable_attribute(manager, "_get_few_shot_examples")(
            tool_name, schema
        ),
        "_get_few_shot_examples",
    )
    output_type = string_field(schema, "output_type", "unknown")
    output_description = string_field(schema, "output_description")
    output_format_instructions = _required_string(
        get_callable_attribute(manager, "_get_output_format_instructions")(
            output_type, output_description
        ),
        "_get_output_format_instructions",
    )
    description = string_field(schema, "description", "No description available")
    parameters = schema.get("parameters", {})
    return f"""You are an expert function simulator. Simulate the execution of the following function call.

=== FUNCTION DETAILS ===
Function Name: {tool_name}

Function Description: {description}

Function Parameters Schema:
{json_module.dumps(parameters, indent=2)}
{output_guidance}

=== ARGUMENTS PROVIDED ===
{json_module.dumps(params, indent=2, default=str)}

Current Date/Time: {current_datetime.strftime("%Y-%m-%d %H:%M:%S")}

{examples}

=== YOUR TASK ===
Generate the return value that '{tool_name}' would produce if executed with the given arguments.

{output_format_instructions}

Generate the output now:"""


def _required_string(value: object, callback_name: str) -> str:
    """Validate string-returning facade callbacks."""
    if not isinstance(value, str):
        raise TypeError(f"{callback_name} must return str")
    return value


def get_few_shot_examples(tool_name: str, schema: ToolSchema) -> str:
    """Return legacy few-shot examples selected by the declared output type."""
    _ = tool_name
    output_type = string_field(schema, "output_type", "unknown")
    examples = {
        "dict": '''
=== EXAMPLES ===

Example 1 - Tool returning user info (dict):
Function: get_user
Arguments: {"user_id": "U123"}
Expected Type: dict
Expected Description: Returns user information including user_id, username, email, created_at
Output:
{"user_id": "U123", "username": "john_doe", "email": "john@example.com", "created_at": "2024-01-15T10:30:00Z", "status": "active"}

Example 2 - Tool creating a resource (dict):
Function: create_ticket
Arguments: {"title": "Issue with login", "priority": "high"}
Expected Type: dict
Expected Description: Returns created ticket details with ticket_id, status
Output:
{"ticket_id": "TKT-2024-001", "title": "Issue with login", "priority": "high", "status": "open", "created_at": "2024-01-20T14:30:00Z"}''',
        "list": '''
=== EXAMPLES ===

Example 1 - Tool returning list of items:
Function: list_files
Arguments: {"directory": "/home/user"}
Expected Type: list
Expected Description: Returns list of filenames in the directory
Output:
["document.txt", "photo.jpg", "data.csv", "notes.md"]

Example 2 - Tool returning list of objects:
Function: get_tweet_comments
Arguments: {"tweet_id": "12345"}
Expected Type: list
Expected Description: Returns list of comments with user info and text
Output:
[{"comment_id": "C001", "user": "@alice", "text": "Great post!", "timestamp": "2024-01-20T10:00:00Z"}, {"comment_id": "C002", "user": "@bob", "text": "Thanks for sharing", "timestamp": "2024-01-20T11:30:00Z"}]''',
        "string": '''
=== EXAMPLES ===

Example 1 - Tool returning a message:
Function: generate_welcome_message
Arguments: {"username": "Alice"}
Expected Type: string
Expected Description: Returns a personalized welcome message
Output:
"Welcome to our platform, Alice! We're excited to have you join us."''',
        "integer": '''
=== EXAMPLES ===

Example 1 - Tool returning a count:
Function: count_lines
Arguments: {"file": "data.txt"}
Expected Type: integer
Expected Description: Returns the number of lines in the file
Output:
42''',
        "float": '''
=== EXAMPLES ===

Example 1 - Tool returning a price:
Function: calculate_exchange_rate
Arguments: {"from": "USD", "to": "EUR"}
Expected Type: float
Expected Description: Returns the current exchange rate
Output:
0.9234''',
    }
    base_type = output_type.split()[0].lower() if output_type else "unknown"
    return examples.get(base_type, "")
