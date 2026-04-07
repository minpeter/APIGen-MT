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
        
        # Create the tool schema
        schema = {
            "name": tool_name,
            "description": tool_data.get('api_description') or tool_data.get('tool_description') or '',
            "parameters": param_schema
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
    
    def get_tools_json_schema(self) -> List[Dict[str, Any]]:
        """Get all tool schemas in JSON format."""
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
        Simulate tool execution using LLM.
        
        Args:
            tool_name: Name of the tool
            params: Parameters for the tool
            schema: Tool schema
            
        Returns:
            Simulated tool output
        """
        prompt = f"""You are an expert function simulator. Based on the following function description and the provided arguments, simulate the execution of this function call.

Function Name: {tool_name}

Function Description: {schema["description"]}

Function Schema:
{json.dumps(schema, indent=2)}

Arguments Provided:
{json.dumps(params, indent=2)}

Current Date/Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} (Assume this is the time of execution)

Task:
Generate a plausible JSON response string that represents what the function '{tool_name}' would return if it were actually executed with the given arguments.
- Consider the function's description (e.g., does it fetch data, create something, authorize, search?).
- Consider the argument values (e.g., dates, search terms).
- If the function description mentions potential errors (like needing authorization for 'fetch_calendar_events'), sometimes simulate those error responses.
- If the function returns nothing on success (like 'create_calendar_event' or 'authorize_calendar_access'), return a JSON indicating success, like '{{"status": "success"}}' or an empty JSON object '{{}}'.
- For functions returning data (like 'fetch_calendar_events' or 'web_search'), generate realistic-looking example data formatted as a JSON string.
- Ensure your entire output is ONLY the JSON string, without any introductory text, explanations, or markdown formatting like ```json ... ```. Just the raw JSON string.
"""

        response, _ = self.llm.json_output(
            system_prompt="You are an expert function simulator outputting only JSON strings.",
            prompt=prompt,
            reasoning=True,
        )
        return response


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