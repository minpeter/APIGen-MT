"""Typed test doubles for step-generation unit tests."""

from typing import TypeIs

from apigen_step_by_step import StepByStepGenerator
from step_by_step_models import (
    ObjectMap,
    StateSnapshot,
    StepSelectionResult,
    TrajectoryStep,
)
from step_by_step_protocols import is_object_list


class StepGenerationHarness(StepByStepGenerator):
    """Expose the protected step-generation surface for direct unit tests."""

    def generate_next_step(
        self,
        query: str,
        trajectory: list[TrajectoryStep],
        execution_context: ObjectMap,
        expected_tools: list[str],
        step_num: int = 1,
    ) -> StepSelectionResult:
        return self._generate_next_step(
            query,
            trajectory,
            execution_context,
            expected_tools,
            step_num,
        )

    def simulate_tool_execution(
        self,
        tool_name: str,
        arguments: object,
        execution_context: ObjectMap,
    ) -> object:
        return self._simulate_tool_execution(
            tool_name,
            arguments,
            execution_context,
        )

    def generate_final_response(
        self,
        query: str,
        trajectory: list[TrajectoryStep],
        execution_context: ObjectMap,
    ) -> str:
        return self._generate_final_response(
            query,
            trajectory,
            execution_context,
        )


class StepGenerationToolManager:
    """Minimal complete tool-manager contract used by step-generation tests."""

    def __init__(self) -> None:
        names_and_categories = (
            ("search_flights", "Travel"),
            ("book_hotel", "Travel"),
            ("get_weather", "Information"),
            ("send_email", "Communication"),
            ("search_restaurants", "Food"),
            ("make_reservation", "Food"),
            ("create_calendar_event", "Productivity"),
            ("get_reviews", "Information"),
        )
        self.tool_schemas: list[ObjectMap] = [
            {
                "name": name,
                "description": name.replace("_", " "),
                "category": category,
                "output_type": "dict",
                "output_description": "Test output",
            }
            for name, category in names_and_categories
        ]
        self.tool_outputs: ObjectMap = {
            "search_flights": [
                {
                    "flight_id": "FL001",
                    "airline": "TestAir",
                    "price": 299,
                }
            ],
            "book_hotel": {
                "confirmation_id": "HT12345",
                "status": "confirmed",
            },
        }
        self.captured_invocations: list[ObjectMap] = []
        self.python_tool_instances: dict[str, object] = {}
        self.api_name_to_class_key: dict[str, str] = {}

    def reset(self) -> None:
        self.captured_invocations.clear()

    def get_captured_invocations(self) -> list[ObjectMap]:
        return self.captured_invocations

    def get_tools_json_schema(self) -> list[ObjectMap]:
        return self.tool_schemas

    def get_tool_schema(self, tool_name: str) -> ObjectMap:
        for tool in self.tool_schemas:
            if tool.get("name") == tool_name:
                return tool
        raise ValueError(f"Tool {tool_name!r} not found")

    def get_categories(self) -> list[str]:
        return sorted(
            {
                category
                for tool in self.tool_schemas
                if isinstance(category := tool.get("category"), str)
            }
        )

    def get_tools_by_category(self, category: str) -> list[ObjectMap]:
        return [tool for tool in self.tool_schemas if tool.get("category") == category]

    def get_tool_category(self, tool_name: str) -> str | None:
        category = self.get_tool_schema(tool_name).get("category")
        return category if isinstance(category, str) else None

    def tool_exists(self, tool_name: str) -> bool:
        return any(tool.get("name") == tool_name for tool in self.tool_schemas)

    def invoke_tool(self, tool_name: str, params: ObjectMap) -> object:
        self.captured_invocations.append({"tool_name": tool_name, "params": params})
        if not self.tool_exists(tool_name):
            raise ValueError(f"Tool {tool_name!r} not found")
        return self.tool_outputs.get(tool_name, {})

    def get_api_state(self) -> StateSnapshot:
        return {}

    def restore_api_state(self, state: StateSnapshot) -> None:
        del state

    def initialize_api_state(self, force_new: bool = False) -> None:
        del force_new

    def has_python_implementation(self, tool_name: str) -> bool:
        del tool_name
        return False

    def invoke_python_tool(
        self,
        tool_name: str,
        params: ObjectMap,
    ) -> object:
        raise NotImplementedError(f"No Python implementation for {tool_name}: {params}")


def is_string_list(value: object) -> TypeIs[list[str]]:
    """Narrow an opaque decoded value to a list of strings."""
    return is_object_list(value) and all(isinstance(item, str) for item in value)
