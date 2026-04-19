"""Mock Tool Manager for testing.

This module provides a configurable mock ToolManager that returns
canned tool schemas and outputs for testing without loading real tools.
"""

import json
from typing import Any, Dict, List, Optional


class MockToolManager:
    """Mock ToolManager with predefined tool schemas and outputs.

    This mock manager simulates a pool of tools for testing the
    step-by-step datapoint generation without requiring actual tool files.

    Attributes:
        tool_schemas: List of tool schema dictionaries
        tool_outputs: Mapping of tool names to canned outputs
        captured_invocations: List of all tool invocations made
    """

    DEFAULT_TOOLS = [
        {
            "name": "search_flights",
            "description": "Search for flights between two locations",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Departure city"},
                    "destination": {"type": "string", "description": "Arrival city"},
                    "date": {"type": "string", "description": "Travel date"},
                },
                "required": ["origin", "destination"],
            },
            "output_type": "list",
            "output_description": "List of available flights with prices and times",
            "category": "Travel",
        },
        {
            "name": "book_hotel",
            "description": "Book a hotel room at a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Hotel location"},
                    "check_in": {"type": "string", "description": "Check-in date"},
                    "check_out": {"type": "string", "description": "Check-out date"},
                    "guests": {"type": "integer", "description": "Number of guests"},
                },
                "required": ["location", "check_in"],
            },
            "output_type": "dict",
            "output_description": "Booking confirmation with reservation ID",
            "category": "Travel",
        },
        {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "country": {"type": "string", "description": "Country code"},
                },
                "required": ["city"],
            },
            "output_type": "dict",
            "output_description": "Weather data including temperature and conditions",
            "category": "Information",
        },
        {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                },
                "required": ["to", "subject"],
            },
            "output_type": "dict",
            "output_description": "Send status and message ID",
            "category": "Communication",
        },
        {
            "name": "search_restaurants",
            "description": "Search for restaurants in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or area"},
                    "cuisine": {"type": "string", "description": "Type of cuisine"},
                    "rating": {"type": "number", "description": "Minimum rating"},
                },
                "required": ["location"],
            },
            "output_type": "list",
            "output_description": "List of restaurants with ratings and details",
            "category": "Food",
        },
        {
            "name": "make_reservation",
            "description": "Make a restaurant reservation",
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant_id": {"type": "string", "description": "Restaurant ID"},
                    "date": {"type": "string", "description": "Reservation date"},
                    "time": {"type": "string", "description": "Reservation time"},
                    "party_size": {"type": "integer", "description": "Number of people"},
                },
                "required": ["restaurant_id", "date", "time"],
            },
            "output_type": "dict",
            "output_description": "Reservation confirmation with booking ID",
            "category": "Food",
        },
        {
            "name": "create_calendar_event",
            "description": "Create a new calendar event",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Event title"},
                    "start_time": {"type": "string", "description": "Start time"},
                    "end_time": {"type": "string", "description": "End time"},
                    "attendees": {"type": "array", "description": "List of attendees"},
                },
                "required": ["summary", "start_time"],
            },
            "output_type": "dict",
            "output_description": "Event creation confirmation with event ID",
            "category": "Productivity",
        },
        {
            "name": "get_reviews",
            "description": "Get reviews for a business or service",
            "parameters": {
                "type": "object",
                "properties": {
                    "business_id": {"type": "string", "description": "Business identifier"},
                    "limit": {"type": "integer", "description": "Maximum reviews"},
                },
                "required": ["business_id"],
            },
            "output_type": "list",
            "output_description": "List of reviews with ratings and comments",
            "category": "Information",
        },
    ]

    DEFAULT_OUTPUTS = {
        "search_flights": [
            {"flight_id": "FL001", "airline": "TestAir", "price": 299, "time": "10:00"},
            {"flight_id": "FL002", "airline": "DemoAir", "price": 349, "time": "14:00"},
        ],
        "book_hotel": {
            "confirmation_id": "HT12345",
            "status": "confirmed",
            "price": 150,
        },
        "get_weather": {
            "temperature": 72,
            "conditions": "sunny",
            "humidity": 45,
        },
        "send_email": {
            "status": "sent",
            "message_id": "MSG-789",
        },
        "search_restaurants": [
            {"id": "R001", "name": "Test Bistro", "rating": 4.5},
            {"id": "R002", "name": "Demo Diner", "rating": 4.2},
        ],
        "make_reservation": {
            "booking_id": "BK-456",
            "status": "confirmed",
            "restaurant": "Test Bistro",
        },
        "create_calendar_event": {
            "event_id": "EV-789",
            "status": "created",
            "calendar_link": "https://calendar.example.com/EV-789",
        },
        "get_reviews": [
            {"rating": 5, "comment": "Excellent service!", "author": "User1"},
            {"rating": 4, "comment": "Good experience", "author": "User2"},
        ],
    }

    def __init__(
        self,
        tools: Optional[List[Dict]] = None,
        outputs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the mock ToolManager.

        Args:
            tools: List of tool schemas (uses DEFAULT_TOOLS if None)
            outputs: Mapping of tool names to outputs (uses DEFAULT_OUTPUTS if None)
        """
        self.tool_schemas = tools or self.DEFAULT_TOOLS.copy()
        self.tool_outputs = outputs or self.DEFAULT_OUTPUTS.copy()
        self.captured_invocations: List[Dict[str, Any]] = []
        self.should_fail: bool = False
        self.fail_tool: Optional[str] = None

    def reset(self):
        """Reset the mock manager state."""
        self.captured_invocations.clear()
        self.should_fail = False
        self.fail_tool = None

    def set_should_fail(self, tool_name: Optional[str] = None):
        """Configure the manager to fail on tool invocation.

        Args:
            tool_name: Specific tool to fail on, or None for all tools
        """
        self.should_fail = True
        self.fail_tool = tool_name

    def add_tool(self, tool_schema: Dict[str, Any], output: Any = None):
        """Add a custom tool schema.

        Args:
            tool_schema: Tool schema dictionary
            output: Canned output for this tool
        """
        self.tool_schemas.append(tool_schema)
        if output is not None:
            self.tool_outputs[tool_schema["name"]] = output

    def get_tools_json_schema(self) -> List[Dict[str, Any]]:
        """Get all tool schemas.

        Returns:
            List of tool schema dictionaries
        """
        return self.tool_schemas

    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get schema for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool schema dictionary

        Raises:
            ValueError: If tool doesn't exist
        """
        for tool in self.tool_schemas:
            if tool["name"] == tool_name:
                return tool
        raise ValueError(f"Tool '{tool_name}' not found")

    def get_categories(self) -> List[str]:
        """Get unique categories from all tools.

        Returns:
            Sorted list of category names
        """
        categories = set()
        for tool in self.tool_schemas:
            categories.add(tool.get("category", "Unknown"))
        return sorted(list(categories))

    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get tools filtered by category.

        Args:
            category: Category name to filter by

        Returns:
            List of tool schemas in the category
        """
        return [
            tool for tool in self.tool_schemas
            if tool.get("category", "Unknown") == category
        ]

    def get_tool_category(self, tool_name: str) -> Optional[str]:
        """Get category for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Category name or None if not found
        """
        for tool in self.tool_schemas:
            if tool["name"] == tool_name:
                return tool.get("category")
        return None

    def tool_exists(self, tool_name: str) -> bool:
        """Check if a tool exists.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool exists, False otherwise
        """
        return any(tool["name"] == tool_name for tool in self.tool_schemas)

    def invoke_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Simulate tool invocation.

        Args:
            tool_name: Name of the tool to invoke
            params: Parameters for the tool

        Returns:
            Canned output for the tool

        Raises:
            ValueError: If tool doesn't exist or should_fail is set
        """
        invocation = {"tool_name": tool_name, "params": params}
        self.captured_invocations.append(invocation)

        if self.should_fail:
            if self.fail_tool is None or self.fail_tool == tool_name:
                raise ValueError(f"Mock tool invocation failed for {tool_name}")

        if not self.tool_exists(tool_name):
            raise ValueError(f"Tool '{tool_name}' not found")

        # Return canned output or empty dict
        return self.tool_outputs.get(tool_name, {})

    def get_captured_invocations(self) -> List[Dict[str, Any]]:
        """Get all captured tool invocations.

        Returns:
            List of invocation dictionaries
        """
        return self.captured_invocations

    def get_invocation_count(self, tool_name: Optional[str] = None) -> int:
        """Get count of invocations.

        Args:
            tool_name: Specific tool to count, or None for all

        Returns:
            Number of invocations
        """
        if tool_name is None:
            return len(self.captured_invocations)
        return sum(
            1 for inv in self.captured_invocations
            if inv["tool_name"] == tool_name
        )


class MockToolManagerBuilder:
    """Builder for creating configured MockToolManager instances."""

    def __init__(self):
        self.tools = []
        self.outputs = {}

    def with_travel_tools(self) -> "MockToolManagerBuilder":
        """Add travel-related tools."""
        travel_tools = [
            t for t in MockToolManager.DEFAULT_TOOLS
            if t["category"] == "Travel"
        ]
        self.tools.extend(travel_tools)
        for tool in travel_tools:
            if tool["name"] in MockToolManager.DEFAULT_OUTPUTS:
                self.outputs[tool["name"]] = MockToolManager.DEFAULT_OUTPUTS[tool["name"]]
        return self

    def with_food_tools(self) -> "MockToolManagerBuilder":
        """Add food/restaurant-related tools."""
        food_tools = [
            t for t in MockToolManager.DEFAULT_TOOLS
            if t["category"] == "Food"
        ]
        self.tools.extend(food_tools)
        for tool in food_tools:
            if tool["name"] in MockToolManager.DEFAULT_OUTPUTS:
                self.outputs[tool["name"]] = MockToolManager.DEFAULT_OUTPUTS[tool["name"]]
        return self

    def with_communication_tools(self) -> "MockToolManagerBuilder":
        """Add communication-related tools."""
        comm_tools = [
            t for t in MockToolManager.DEFAULT_TOOLS
            if t["category"] == "Communication"
        ]
        self.tools.extend(comm_tools)
        for tool in comm_tools:
            if tool["name"] in MockToolManager.DEFAULT_OUTPUTS:
                self.outputs[tool["name"]] = MockToolManager.DEFAULT_OUTPUTS[tool["name"]]
        return self

    def with_custom_tool(
        self,
        name: str,
        description: str,
        category: str = "Custom",
        parameters: Optional[Dict] = None,
        output_type: str = "dict",
        output_description: str = "",
        output: Any = None,
    ) -> "MockToolManagerBuilder":
        """Add a custom tool.

        Args:
            name: Tool name
            description: Tool description
            category: Tool category
            parameters: Parameter schema
            output_type: Expected output type
            output_description: Output description
            output: Canned output
        """
        tool = {
            "name": name,
            "description": description,
            "category": category,
            "parameters": parameters or {"type": "object", "properties": {}, "required": []},
            "output_type": output_type,
            "output_description": output_description,
        }
        self.tools.append(tool)
        self.outputs[name] = output if output is not None else {"result": "success"}
        return self

    def build(self) -> MockToolManager:
        """Build and return the configured manager.

        Returns:
            Configured MockToolManager instance
        """
        return MockToolManager(tools=self.tools or None, outputs=self.outputs or None)
