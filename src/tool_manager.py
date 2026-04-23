from function_schema import get_function_schema
from typing import Any, Dict, List, Optional
import json
import datetime
import os
from pathlib import Path

from llm_client import LLMClient


# >>>>> Example functions (kept for reference) <<<<<

def create_calendar_event(
    summary: str,
    start_time: str,
    end_time: str,
) -> None:
    """Creates a new calendar event."""
    pass


def fetch_calendar_events(
    start_date: str,
    end_date: str,
) -> str:
    """
    Retrieves calendar events within a specified date range.
    """
    pass


def web_search(query: str) -> str:
    """
    Searches the web (DuckDuckGo) for the given query.
    Returns a JSON string containing search results.
    """
    pass


# <<<<< Example functions <<<<<


class ToolManager:
    """
    Manages a pool of tools that can be loaded from a file or defined in code.
    
    Supports loading tools from:
    1. A JSON/JSONL file containing BFCL-style tool definitions
    2. Python functions with type hints and docstrings
    """
    
    def __init__(
        self, 
        llm: LLMClient, 
        tool_pool_path: Optional[str] = None,
        tools: Optional[List] = None
    ):
        """
        Initialize the ToolManager.
        
        Args:
            llm: LLMClient instance for simulating tool execution
            tool_pool_path: Path to a JSON or JSONL file containing tool definitions.
                           If provided, tools will be loaded from this file.
            tools: Optional list of Python functions to use as tools.
                  If both tool_pool_path and tools are provided, they will be merged.
        """
        self.llm = llm
        self.tool_schemas: List[Dict[str, Any]] = []
        self.tool_implementations: Dict[str, Any] = {}
        
        # Load tools from file if path is provided
        if tool_pool_path:
            self._load_tools_from_file(tool_pool_path)
        
        # Load tools from Python functions if provided
        if tools:
            self._load_tools_from_functions(tools)
        
        # If neither file nor functions provided, use default example tools
        if not self.tool_schemas:
            self._load_default_tools()
    
    def _load_tools_from_file(self, path: str) -> None:
        """
        Load tools from a JSON or JSONL file.
        
        Args:
            path: Path to the tool definition file
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Tool pool file not found: {path}")
        
        with open(path_obj, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        tools_data = []
        
        # Try to parse as JSON first, then as JSONL
        try:
            # Try parsing as JSON array
            tools_data = json.loads(content)
            if not isinstance(tools_data, list):
                tools_data = [tools_data]
        except json.JSONDecodeError:
            # Parse as JSONL (one JSON object per line)
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    try:
                        tools_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line: {line[:50]}... Error: {e}")
        
        for tool_data in tools_data:
            self._add_tool_from_bfcl_definition(tool_data)
    
    def _add_tool_from_bfcl_definition(self, tool_data: Dict[str, Any]) -> None:
        """
        Add a tool from BFCL-style definition.
        
        Args:
            tool_data: Dictionary containing tool definition
        """
        tool_name = tool_data.get('api_name') or tool_data.get('name')
        if not tool_name:
            print(f"Warning: Skipping tool without name: {tool_data}")
            return
        
        # Build the schema in the format expected by the system
        parameters = tool_data.get('parameters', {})
        properties = parameters.get('properties', {})
        required = parameters.get('required', [])
        optional = parameters.get('optional', [])
        
        # Build parameter schema
        param_schema = {
            "type": "object",
            "properties": {},
            "required": required
        }
        
        # Convert properties
        for param_name, param_info in properties.items():
            if isinstance(param_info, dict):
                param_type = param_info.get('type', 'string')
                # Map types to JSON Schema types
                type_mapping = {
                    'STRING': 'string',
                    'NUMBER': 'number',
                    'INTEGER': 'integer',
                    'FLOAT': 'number',
                    'BOOLEAN': 'boolean',
                    'ARRAY': 'array',
                    'OBJECT': 'object',
                    'DATE': 'string',
                    'TUPLE': 'array',
                }
                json_type = type_mapping.get(param_type.upper(), 'string')
                
                param_schema["properties"][param_name] = {
                    "type": json_type,
                    "description": param_info.get('description', '')
                }
                
                # Add default if present
                if 'default' in param_info:
                    param_schema["properties"][param_name]["default"] = param_info['default']
            else:
                # Simple type definition
                param_schema["properties"][param_name] = {
                    "type": "string",
                    "description": ""
                }
        
        # Create the tool schema with output information
        schema = {
            "name": tool_name,
            "description": tool_data.get('api_description') or tool_data.get('tool_description') or '',
            "parameters": param_schema,
            # Include output type and description for new format
            "output_type": tool_data.get('output_type', 'unknown'),
            "output_description": tool_data.get('output_description', ''),
            # Include category for grouping
            "category": tool_data.get('category', 'Unknown')
        }
        
        self.tool_schemas.append(schema)
        
        # Store implementation info for virtual execution
        self.tool_implementations[tool_name] = {
            "type": "bfcl",
            "data": tool_data
        }
    
    def _load_tools_from_functions(self, tools: List) -> None:
        """
        Load tools from a list of Python functions.
        
        Args:
            tools: List of Python function objects
        """
        for tool_func in tools:
            schema = get_function_schema(tool_func)
            self.tool_schemas.append(schema)
            self.tool_implementations[tool_func.__name__] = {
                "type": "python",
                "func": tool_func
            }
    
    def _load_default_tools(self) -> None:
        """Load default example tools."""
        default_tools = [
            create_calendar_event,
            fetch_calendar_events,
            web_search,
        ]
        for tool_func in default_tools:
            schema = get_function_schema(tool_func)
            self.tool_schemas.append(schema)
            self.tool_implementations[tool_func.__name__] = {
                "type": "python",
                "func": tool_func
            }
    
    def get_categories(self) -> List[str]:
        """
        Get a list of unique categories across all tools.

        Returns:
            List[str]: List of unique category names
        """
        categories = set()
        for tool in self.tool_schemas:
            category = tool.get('category', 'Unknown')
            categories.add(category)
        return sorted(list(categories))

    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get a list of tools that belong to the specified category.

        Args:
            category: The category to filter tools by

        Returns:
            List[Dict[str, Any]]: List of tool schemas in the specified category
        """
        return [tool for tool in self.tool_schemas if tool.get('category') == category]

    def get_tool_category(self, tool_name: str) -> Optional[str]:
        """
        Get the category for a specific tool.

        Args:
            tool_name: The name of the tool

        Returns:
            Optional[str]: The category of the tool, or None if not found
        """
        for tool in self.tool_schemas:
            if tool.get('name') == tool_name:
                return tool.get('category')
        return None

    def get_tools_json_schema(self) -> List[Dict[str, Any]]:
        """Get all tool schemas in JSON format."""
        return self.tool_schemas

    def get_tools_with_descriptions(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get tools with their full descriptions.

        Args:
            category: Optional category to filter by

        Returns:
            List[Dict[str, Any]]: List of tools with descriptions
        """
        if category:
            return self.get_tools_by_category(category)
        return self.tool_schemas

    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """
        Get the schema for a specific tool by name.

        Args:
            tool_name: The name of the tool to get the schema for

        Returns:
            dict: The schema for the tool
            
        Raises:
            ValueError: If the tool does not exist
        """
        for tool in self.tool_schemas:
            if tool["name"] == tool_name:
                return tool

        available_tools = [tool["name"] for tool in self.tool_schemas]
        raise ValueError(
            f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}"
        )
    
    def tool_exists(self, tool_name: str) -> bool:
        """
        Check if a tool with the given name exists in the available tools.
        
        Args:
            tool_name: The name of the tool to check
            
        Returns:
            bool: True if the tool exists, False otherwise
        """
        return any(tool["name"] == tool_name for tool in self.tool_schemas)
    
    def invoke_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        Invoke a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to invoke
            params: Parameters to pass to the tool
            
        Returns:
            The result of the tool invocation
            
        Raises:
            ValueError: If the tool does not exist
        """
        # Check if tool exists and get its schema
        tool_schema = None
        for tool in self.tool_schemas:
            if tool["name"] == tool_name:
                tool_schema = tool
                break
        
        if tool_schema is None:
            available_tools = [tool["name"] for tool in self.tool_schemas]
            raise ValueError(
                f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}"
            )
        
        # Get implementation info
        impl_info = self.tool_implementations.get(tool_name)
        
        if impl_info and impl_info.get("type") == "python":
            # Call the actual Python function
            func = impl_info.get("func")
            if func:
                try:
                    return func(**params)
                except Exception as e:
                    return {"error": str(e)}
        
        # Use virtual tool executor for BFCL-style tools or when no implementation exists
        return self.__virtual_tool_executor(tool_name, params, schema=tool_schema)
    
    def __virtual_tool_executor(
        self, tool_name: str, params: dict, schema: dict
    ) -> Any:
        """
        Simulate tool execution using LLM with enhanced prompts and retry logic.

        Args:
            tool_name: Name of the tool
            params: Parameters for the tool
            schema: Tool schema

        Returns:
            Simulated tool output
        """
        # Extract output type and description from schema
        output_type = schema.get('output_type', 'unknown')
        output_description = schema.get('output_description', '')

        # Build enhanced output guidance with type-specific examples
        output_guidance = self._build_output_guidance(output_type, output_description)

        # Build the enhanced prompt with few-shot examples
        prompt = self._build_simulation_prompt(
            tool_name=tool_name,
            params=params,
            schema=schema,
            output_guidance=output_guidance
        )

        # Try with retries for validation failures
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response, _ = self.llm.json_output(
                    prompt=prompt,
                    reasoning=True,
                )

                # Validate the response
                validated_response = self._validate_tool_output(
                    tool_name, response, output_type, output_description
                )

                # Check if validation passed
                if self._is_output_valid(validated_response, output_type, output_description):
                    return validated_response

                if attempt < max_retries:
                    print(f"    Tool simulation validation failed for {tool_name}, retrying ({attempt + 1}/{max_retries})...")
                    # Add correction guidance for retry
                    prompt += f"\n\n=== CORRECTION NEEDED ===\nYour previous output did not match the expected type '{output_type}'.\nPlease regenerate ensuring the output is of type {output_type} and matches the description.\n"

            except Exception as e:
                print(f"    Error simulating tool {tool_name}: {e}")
                if attempt < max_retries:
                    continue
                # Return a sensible default on final failure
                return self._get_default_output(output_type)

        return validated_response if 'validated_response' in locals() else self._get_default_output(output_type)

    def _validate_tool_output(
        self, 
        tool_name: str, 
        output: Any, 
        expected_type: str, 
        output_description: str
    ) -> Any:
        """
        Validate that the simulated tool output matches the declared output type and description.
        
        Args:
            tool_name: Name of the tool
            output: The simulated output to validate
            expected_type: The declared output type
            output_description: The declared output description
            
        Returns:
            The validated output (unchanged if valid)
        """
        if expected_type == 'unknown' or not expected_type:
            # No validation if type is unknown
            return output
            
        # Basic type checking
        type_mapping = {
            'string': str,
            'integer': int,
            'float': (int, float),
            'boolean': bool,
            'list': list,
            'dict': dict,
            'number': (int, float),
        }
        
        # Handle compound types like "integer or string"
        base_type = expected_type.split()[0].lower()  # Take first word
        expected_python_type = type_mapping.get(base_type, None)
        
        if expected_python_type:
            if not isinstance(output, expected_python_type):
                # Try to convert if possible
                try:
                    if base_type in ('integer', 'float', 'number'):
                        if isinstance(output, str):
                            output = float(output) if '.' in output else int(output)
                    elif base_type == 'string':
                        output = str(output)
                    elif base_type == 'boolean':
                        if isinstance(output, str):
                            output = output.lower() in ('true', 'yes', '1')
                    elif base_type == 'list':
                        if isinstance(output, dict):
                            output = list(output.values())
                    elif base_type == 'dict':
                        if isinstance(output, str):
                            output = json.loads(output)
                except (ValueError, TypeError, json.JSONDecodeError):
                    # If conversion fails, return as-is but log warning
                    print(f"Warning: Output type mismatch for {tool_name}. Expected {expected_type}, got {type(output).__name__}")
        
        # Use LLM to verify output matches description
        if output_description and output_description != 'Failed to predict output description':
            validation_prompt = f"""You are an output validator. Given the following tool output and its expected description, determine if the output is plausible and matches the description.

Tool Name: {tool_name}
Expected Output Description: {output_description}
Actual Output: {json.dumps(output, default=str) if isinstance(output, (dict, list)) else str(output)}

Does the output plausibly match the description? Answer YES or NO, followed by a brief explanation.

Response format:
{{
    "VALID": "YES" or "NO",
    "REASON": "<brief explanation>"
}}"""

            try:
                validation_response, _ = self.llm.json_output(
                    prompt=validation_prompt,
                    reasoning=False,
                )
                
                if validation_response:
                    is_valid = validation_response.get('VALID', 'YES').upper() == 'YES'
                    if not is_valid:
                        reason = validation_response.get('REASON', 'No reason provided')
                        print(f"Warning: LLM validation flagged output for {tool_name} as not matching description: {reason}")
                        # Return error to trigger retry instead of returning invalid output
                        return {"error": f"Output validation failed: {reason}", "error_type": "validation_failure"}
            except Exception as e:
                # If validation fails, just log and continue
                print(f"Warning: Could not validate output for {tool_name}: {e}")

        return output

    def _build_output_guidance(self, output_type: str, output_description: str) -> str:
        """Build enhanced output guidance with type-specific examples and field requirements."""
        guidance = ""

        if output_type and output_type != 'unknown':
            guidance += f"\n\n=== REQUIRED OUTPUT TYPE ==="
            guidance += f"\nType: {output_type}"
            guidance += f"\nCRITICAL: You MUST return output of exactly this type. No exceptions."

            # Add type-specific examples
            type_examples = {
                'dict': '{"key": "value", "id": 123, "name": "example"}',
                'list': '[{"item": 1}, {"item": 2}, {"item": 3}]',
                'string': '"your string value here"',
                'integer': '42',
                'float': '3.14',
                'number': '42 or 3.14',
                'boolean': 'true or false',
            }

            base_type = output_type.split()[0].lower()
            if base_type in type_examples:
                guidance += f"\nExample of correct {output_type} output: {type_examples[base_type]}"

        if output_description and output_description != 'Failed to predict output description':
            guidance += f"\n\n=== OUTPUT DESCRIPTION ==="
            guidance += f"\n{output_description}"
            guidance += f"\n"
            guidance += f"\nYOUR OUTPUT MUST SATISFY THIS DESCRIPTION:"
            guidance += f"\n- Study the description carefully and include ALL fields/values mentioned"
            guidance += f"\n- The output content must realistically match what the description promises"
            guidance += f"\n- For dict outputs, ensure all keys mentioned in the description are present"
            guidance += f"\n- Do NOT invent fields that aren't mentioned in the description"

            # Parse description for common field patterns and add explicit requirements
            desc_lower = output_description.lower()
            if 'message_id' in desc_lower or 'message id' in desc_lower:
                guidance += f"\n- MUST include 'message_id' field"
            if 'success' in desc_lower:
                guidance += f"\n- MUST include 'success' field (boolean)"
            if 'timestamp' in desc_lower:
                guidance += f"\n- MUST include 'timestamp' field with ISO format"
            if 'status' in desc_lower:
                guidance += f"\n- MUST include 'status' field"
            if 'id' in desc_lower and base_type == 'dict':
                guidance += f"\n- MUST include an 'id' or identifier field"

        return guidance

    def _get_output_format_instructions(self, output_type: str, output_description: str = "") -> str:
        """Get specific formatting instructions based on the output type and description."""
        base_type = output_type.split()[0].lower() if output_type else 'unknown'

        # Common instructions about matching description
        desc_check = ""
        if output_description and output_description != 'Failed to predict output description':
            desc_check = f"""

MANDATORY OUTPUT CONTENT CHECK:
- Your output MUST match this description: {output_description}
- Include ALL fields mentioned in the description
- Values must be realistic and appropriate for the described output"""

        if base_type == 'dict':
            return f"""REQUIREMENTS:
1. CRITICAL: Return ONLY a JSON OBJECT (dictionary) - NOT a string, NOT a list, NOT a number
2. The JSON object MUST have key-value pairs: {{"field1": "value1", "field2": "value2"}}
3. Example: {{"user_id": "U12345", "name": "John", "status": "active"}}
4. NO markdown formatting, NO code blocks (no ```), NO explanations - output ONLY the raw JSON object
5. For error cases ONLY, return: {{"error": "description", "error_description": "details"}}{desc_check}"""
        elif base_type == 'list':
            return f"""REQUIREMENTS:
1. CRITICAL: Return ONLY a JSON ARRAY (list) - NOT a dict, NOT a string, NOT a number
2. The JSON array MUST be wrapped in square brackets: [item1, item2, item3]
3. Example: [{{"id": 1}}, {{"id": 2}}] or ["item1", "item2"]
4. NO markdown formatting, NO code blocks (no ```), NO explanations - output ONLY the raw JSON array
5. For error cases ONLY, return: {{"error": "description", "error_description": "details"}}{desc_check}"""
        elif base_type == 'string':
            return f"""REQUIREMENTS:
1. CRITICAL: Return ONLY a PLAIN STRING value - NOT a JSON object, NOT a JSON array, NOT a number
2. Example: "U12345" or "Operation completed successfully" or just Hello World (without quotes is also acceptable)
3. The output should be the string value itself, optionally wrapped in quotes
4. NO markdown formatting, NO code blocks (no ```), NO explanations - output ONLY the raw string
5. CRITICAL: Do NOT wrap the string in a dict like {{"result": "string"}} - return ONLY the string
6. For error cases ONLY, return a JSON object: {{"error": "description"}}{desc_check}"""
        elif base_type in ('integer', 'float', 'number'):
            return f"""REQUIREMENTS:
1. CRITICAL: Return ONLY a NUMERIC value (integer or float) - NOT a string, NOT a dict, NOT a list
2. Example: 42 or 3.14
3. NO quotes around the number - output the raw number only
4. NO markdown formatting, NO code blocks, NO explanations - output ONLY the raw number
5. For error cases ONLY, return: {{"error": "description", "error_description": "details"}}{desc_check}"""
        elif base_type == 'boolean':
            return f"""REQUIREMENTS:
1. CRITICAL: Return ONLY a BOOLEAN value: true or false (lowercase, no quotes) - NOT a string "true"
2. Example: true (NOT "true", NOT {{"result": true}})
3. NO markdown formatting, NO code blocks, NO explanations - output ONLY the raw boolean
4. For error cases ONLY, return: {{"error": "description", "error_description": "details"}}{desc_check}"""
        else:
            return f"""REQUIREMENTS:
1. CRITICAL: Return a value matching the REQUIRED OUTPUT TYPE specified above
2. NO markdown formatting, NO code blocks, NO explanations - output ONLY the raw value
3. For error cases, return: {{"error": "description", "error_description": "details"}}{desc_check}"""

    def _build_simulation_prompt(self, tool_name: str, params: dict, schema: dict, output_guidance: str) -> str:
        """Build the enhanced simulation prompt with examples."""

        # Build few-shot examples based on tool type
        examples = self._get_few_shot_examples(tool_name, schema)

        # Determine output format instructions based on output type and description
        output_type = schema.get('output_type', 'unknown')
        output_description = schema.get('output_description', '')
        output_format_instructions = self._get_output_format_instructions(output_type, output_description)

        prompt = f"""You are an expert function simulator. Simulate the execution of the following function call.

=== FUNCTION DETAILS ===
Function Name: {tool_name}

Function Description: {schema.get('description', 'No description available')}

Function Parameters Schema:
{json.dumps(schema.get('parameters', {}), indent=2)}
{output_guidance}

=== ARGUMENTS PROVIDED ===
{json.dumps(params, indent=2, default=str)}

Current Date/Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{examples}

=== YOUR TASK ===
Generate the return value that '{tool_name}' would produce if executed with the given arguments.

{output_format_instructions}

Generate the output now:"""

        return prompt

    def _get_few_shot_examples(self, tool_name: str, schema: dict) -> str:
        """Generate few-shot examples based on the tool type."""
        output_type = schema.get('output_type', 'unknown')

        # Examples for common tool patterns
        examples = {
            'dict': '''
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

            'list': '''
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

            'string': '''
=== EXAMPLES ===

Example 1 - Tool returning a message:
Function: generate_welcome_message
Arguments: {"username": "Alice"}
Expected Type: string
Expected Description: Returns a personalized welcome message
Output:
"Welcome to our platform, Alice! We're excited to have you join us."''',

            'integer': '''
=== EXAMPLES ===

Example 1 - Tool returning a count:
Function: count_lines
Arguments: {"file": "data.txt"}
Expected Type: integer
Expected Description: Returns the number of lines in the file
Output:
42''',

            'float': '''
=== EXAMPLES ===

Example 1 - Tool returning a price:
Function: calculate_exchange_rate
Arguments: {"from": "USD", "to": "EUR"}
Expected Type: float
Expected Description: Returns the current exchange rate
Output:
0.9234'''
        }

        base_type = output_type.split()[0].lower() if output_type else 'unknown'
        return examples.get(base_type, "")

    def _is_output_valid(self, output: Any, expected_type: str, expected_description: str) -> bool:
        """Quick validation check to see if output matches expected type."""
        if expected_type == 'unknown' or not expected_type:
            return True

        # Basic type checking
        base_type = expected_type.split()[0].lower()

        type_checks = {
            'dict': lambda x: isinstance(x, dict),
            'list': lambda x: isinstance(x, list),
            'string': lambda x: isinstance(x, str),
            'integer': lambda x: isinstance(x, int) and not isinstance(x, bool),
            'float': lambda x: isinstance(x, float),
            'number': lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
            'boolean': lambda x: isinstance(x, bool),
        }

        check_func = type_checks.get(base_type)
        if check_func:
            return check_func(output)

        return True

    def _get_default_output(self, output_type: str) -> Any:
        """Return a sensible default output for the given type."""
        defaults = {
            'dict': {"status": "success", "message": "Operation completed"},
            'list': [],
            'string': "success",
            'integer': 0,
            'float': 0.0,
            'number': 0,
            'boolean': True,
        }

        base_type = output_type.split()[0].lower() if output_type else 'unknown'
        return defaults.get(base_type, {"status": "completed"})


# Default tools for backward compatibility (can be imported if needed)
DEFAULT_TOOLS = [
    create_calendar_event,
    fetch_calendar_events,
    web_search,
]


if __name__ == "__main__":
    # Example: Load tools from BFCL tool pool file
    tool_pool_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "..", 
        "data", 
        "magnet_mt", 
        "output", 
        "tool_pool.jsonl"
    )
    
    print("Initializing ToolManager with tool pool file...")
    tool_manager = ToolManager(
        llm=LLMClient(),
        tool_pool_path=tool_pool_path
    )
    
    # Get available tools
    tools = tool_manager.get_tools_json_schema()
    print(f"Available tools ({len(tools)}):")
    for tool in tools[:5]:  # Show first 5
        print(f"  - {tool['name']}: {tool['description'][:50]}...")
    if len(tools) > 5:
        print(f"  ... and {len(tools) - 5} more")
    
    # Test invoking a tool
    if tools:
        test_tool = tools[0]["name"]
        print(f"\nTesting tool invocation: {test_tool}")
        try:
            result = tool_manager.invoke_tool(test_tool, {})
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")