from __future__ import annotations

import copy
from typing import ClassVar

import pytest
from pydantic import TypeAdapter

from apigen_multi_turn import (
    DialogBlueprint,
    MultiTurnConversation,
    MultiTurnGenerator,
    Turn,
)
from apigen_step_by_step import (
    QueryGenerationResult,
    StateVerificationResult,
    StepByStepGenerator,
    ToolCallWithOutput,
    TrajectoryStep,
)
from multi_turn_protocols import BlueprintTurn
from step_by_step_models import ObjectMap, StateSnapshot
from tests.mocks.mock_llm_client import MockLLMClient
from trajectory_replay import verify_trajectory_replay


class CounterToolManager:
    tool_schemas: ClassVar[list[ObjectMap]] = [
        {
            "name": "increment_counter",
            "description": "Increment a counter",
            "category": "Counter",
            "output_type": "dict",
            "output_description": "Current counter value",
        },
        {
            "name": "read_counter",
            "description": "Read a counter",
            "category": "Counter",
            "output_type": "dict",
            "output_description": "Current counter value",
        },
        {
            "name": "explode",
            "description": "Raise after mutating a counter",
            "category": "Counter",
            "output_type": "dict",
            "output_description": "Never returned",
        },
    ]

    def __init__(self, value: int = 0) -> None:
        self.value: int = value
        self.unsafe_tools: set[str] = set()
        self.invoked_tools: list[str] = []
        self.restore_count: int = 0
        self.python_tool_instances: dict[str, object] = {"counter": self}
        self.api_name_to_class_key: dict[str, str] = {
            "increment_counter": "counter",
            "read_counter": "counter",
            "explode": "counter",
        }

    def get_api_state(self) -> StateSnapshot:
        return {"counter": {"value": self.value}}

    def restore_api_state(self, state: StateSnapshot) -> None:
        self.restore_count += 1
        value = copy.deepcopy(state)["counter"]["value"]
        assert isinstance(value, int)
        self.value = value

    def initialize_api_state(self, force_new: bool = False) -> None:
        del force_new

    def has_python_implementation(self, tool_name: str) -> bool:
        return tool_name in {"increment_counter", "read_counter", "explode"}

    def is_replay_safe(self, tool_name: str) -> bool:
        return tool_name not in self.unsafe_tools

    def invoke_python_tool(
        self,
        tool_name: str,
        params: ObjectMap,
    ) -> dict[str, int]:
        self.invoked_tools.append(tool_name)
        if tool_name == "increment_counter":
            amount = params["amount"]
            assert isinstance(amount, int)
            self.value += amount
            return {"value": self.value}
        if tool_name == "read_counter":
            return {"value": self.value}
        self.value += 1
        raise RuntimeError("replay failed")

    def invoke_tool(self, tool_name: str, params: ObjectMap) -> object:
        return self.invoke_python_tool(tool_name, params)

    def get_tool_schema(self, tool_name: str) -> ObjectMap:
        return next(
            tool for tool in self.tool_schemas if tool["name"] == tool_name
        )

    def get_tool_category(self, tool_name: str) -> str:
        del tool_name
        return "Counter"

    def get_tools_json_schema(self) -> list[ObjectMap]:
        return self.tool_schemas

    def get_tools_by_category(self, category: str) -> list[ObjectMap]:
        return [
            tool for tool in self.tool_schemas if tool["category"] == category
        ]

    def get_categories(self) -> list[str]:
        return ["Counter"]

    def tool_exists(self, tool_name: str) -> bool:
        return any(tool["name"] == tool_name for tool in self.tool_schemas)


class DeterministicTestToolManager:
    """Complete manager contract for deterministic multi-turn tests."""

    def __init__(self) -> None:
        self.tool_schemas: list[ObjectMap] = [
            {
                "name": "search_flights",
                "description": "Find a flight",
                "category": "Travel",
                "output_type": "list",
                "output_description": "Available flights",
            }
        ]
        self.tool_outputs: ObjectMap = {
            "search_flights": [
                {"flight_id": "FL001", "price": 299},
            ]
        }
        self.captured_invocations: list[ObjectMap] = []
        self.should_fail: bool = False
        self.fail_tool: str | None = None
        self.python_tool_instances: dict[str, object] = {}
        self.api_name_to_class_key: dict[str, str] = {}

    def get_tools_json_schema(self) -> list[ObjectMap]:
        return self.tool_schemas

    def get_tool_schema(self, tool_name: str) -> ObjectMap:
        return next(
            tool for tool in self.tool_schemas if tool["name"] == tool_name
        )

    def get_categories(self) -> list[str]:
        return ["Travel"]

    def get_tools_by_category(self, category: str) -> list[ObjectMap]:
        return [
            tool for tool in self.tool_schemas if tool["category"] == category
        ]

    def get_tool_category(self, tool_name: str) -> str | None:
        category = self.get_tool_schema(tool_name).get("category")
        return category if isinstance(category, str) else None

    def tool_exists(self, tool_name: str) -> bool:
        return any(tool["name"] == tool_name for tool in self.tool_schemas)

    def invoke_tool(self, tool_name: str, params: ObjectMap) -> object:
        del params
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


@pytest.fixture
def mock_tools() -> DeterministicTestToolManager:
    return DeterministicTestToolManager()


class BlueprintValidationGenerator(MultiTurnGenerator):
    def verify_blueprint_capabilities(
        self,
        turns: list[BlueprintTurn],
    ) -> tuple[bool, list[str]]:
        return self._verify_blueprint_capabilities(turns)


def counter_step(
    tool_name: str,
    arguments: dict[str, object],
    output: dict[str, int],
    pre_value: int,
    post_value: int,
    *,
    state_valid: bool = True,
) -> TrajectoryStep:
    return TrajectoryStep(
        step_number=1,
        tool_calls=[
            ToolCallWithOutput(
                tool_name=tool_name,
                arguments=arguments,
                output=output,
            )
        ],
        pre_state={"counter": {"value": pre_value}},
        post_state={"counter": {"value": post_value}},
        state_verification=StateVerificationResult(
            is_valid=state_valid,
            reasoning="fixture",
            issues=[] if state_valid else ["invalid transition"],
            state_changes_summary="fixture",
        ),
    )


def test_valid_mutating_transition_remains_accepted(mock_llm: MockLLMClient):
    manager = CounterToolManager(value=41)
    generator = StepByStepGenerator(mock_llm, manager, num_actions=1)
    step = counter_step(
        "increment_counter",
        {"amount": 1},
        {"value": 1},
        0,
        1,
    )

    result = generator.run_full_verification(
        "Increment the counter",
        [step],
        {},
    )

    assert result.overall_verification_passed is True
    assert manager.value == 41


def test_valid_read_only_transition_remains_accepted(mock_llm: MockLLMClient):
    manager = CounterToolManager(value=41)
    generator = StepByStepGenerator(mock_llm, manager, num_actions=1)
    step = counter_step(
        "read_counter",
        {},
        {"value": 0},
        0,
        0,
    )

    result = generator.run_full_verification("Read the counter", [step], {})

    assert result.overall_verification_passed is True
    assert manager.value == 41


def test_valid_multi_call_step_remains_accepted(mock_llm: MockLLMClient):
    manager = CounterToolManager(value=41)
    generator = StepByStepGenerator(mock_llm, manager, num_actions=2)
    step = TrajectoryStep(
        step_number=1,
        tool_calls=[
            ToolCallWithOutput(
                tool_name="increment_counter",
                arguments={"amount": 1},
                output={"value": 1},
            ),
            ToolCallWithOutput(
                tool_name="increment_counter",
                arguments={"amount": 1},
                output={"value": 2},
            ),
        ],
        pre_state={"counter": {"value": 0}},
        post_state={"counter": {"value": 2}},
    )

    result = generator.run_full_verification(
        "Increment the counter twice",
        [step],
        {},
    )

    assert result.overall_verification_passed is True
    assert manager.value == 41


def test_non_replayable_tool_is_marked_unavailable():
    manager = CounterToolManager(value=41)
    step = TrajectoryStep(
        step_number=1,
        tool_calls=[
            ToolCallWithOutput(
                tool_name="remote_tool",
                arguments={},
                output={"ok": True},
            )
        ],
        pre_state={"counter": {"value": 0}},
        post_state={"counter": {"value": 0}},
    )

    result = verify_trajectory_replay(manager, [step])

    assert result.status == "unavailable"
    assert result.is_valid is False
    assert result.checked_calls == 0
    assert result.unavailable_tools == ["remote_tool"]
    assert manager.value == 41


def test_unsafe_local_tool_is_not_invoked():
    manager = CounterToolManager(value=41)
    manager.unsafe_tools.add("increment_counter")
    step = counter_step(
        "increment_counter",
        {"amount": 1},
        {"value": 1},
        0,
        1,
    )

    result = verify_trajectory_replay(manager, [step])

    assert result.status == "unavailable"
    assert result.is_valid is False
    assert result.unavailable_tools == ["increment_counter"]
    assert manager.invoked_tools == []
    assert manager.restore_count == 0
    assert manager.value == 41


def test_replay_resolves_arguments_from_prior_outputs():
    manager = CounterToolManager(value=41)
    trajectory = [
        counter_step(
            "read_counter",
            {},
            {"value": 2},
            2,
            2,
        ),
        counter_step(
            "increment_counter",
            {"amount": "{{read_counter_output.value}}"},
            {"value": 4},
            2,
            4,
        ),
    ]
    trajectory[1].step_number = 2

    result = verify_trajectory_replay(manager, trajectory)

    assert result.status == "verified"
    assert result.is_valid is True
    assert manager.value == 41


def test_tampered_output_is_rejected_by_replay(mock_llm: MockLLMClient):
    manager = CounterToolManager(value=41)
    generator = StepByStepGenerator(mock_llm, manager, num_actions=1)
    step = counter_step(
        "increment_counter",
        {"amount": 1},
        {"value": 999},
        0,
        1,
    )

    result = generator.run_full_verification(
        "Increment the counter",
        [step],
        {},
    )

    assert result.overall_verification_passed is False
    assert manager.value == 41


def test_tampered_post_state_is_rejected_by_replay(mock_llm: MockLLMClient):
    manager = CounterToolManager(value=41)
    generator = StepByStepGenerator(mock_llm, manager, num_actions=1)
    step = counter_step(
        "increment_counter",
        {"amount": 1},
        {"value": 1},
        0,
        7,
    )

    result = generator.run_full_verification(
        "Increment the counter",
        [step],
        {},
    )

    assert result.overall_verification_passed is False
    assert manager.value == 41


def test_replay_exception_is_rejected_and_live_state_restored(mock_llm: MockLLMClient):
    manager = CounterToolManager(value=41)
    generator = StepByStepGenerator(mock_llm, manager, num_actions=1)
    step = counter_step("explode", {}, {"value": 1}, 0, 1)

    result = generator.run_full_verification("Explode", [step], {})

    assert result.overall_verification_passed is False
    assert manager.value == 41


def test_failed_recorded_state_verdict_is_not_ignored(mock_llm: MockLLMClient):
    manager = CounterToolManager(value=41)
    generator = StepByStepGenerator(mock_llm, manager, num_actions=1)
    step = counter_step(
        "read_counter",
        {},
        {"value": 0},
        0,
        0,
        state_valid=False,
    )

    result = generator.run_full_verification("Read the counter", [step], {})

    assert result.overall_verification_passed is False


def test_state_transition_judge_error_fails_closed(mock_llm: MockLLMClient):
    manager = CounterToolManager(value=41)
    generator = StepByStepGenerator(mock_llm, manager, num_actions=1)
    mock_llm.set_responses(["not-json"])

    result = generator.verify_state_transition(
        "increment_counter",
        {"amount": 1},
        {"value": 1},
        {"counter": {"value": 0}},
        {"counter": {"value": 1}},
    )

    assert result.is_valid is False
    assert result.issues
    assert "judge call failed" in result.issues[0].lower()


def test_invocation_order_rejects_out_of_order_steps(
    mock_llm: MockLLMClient,
):
    manager = CounterToolManager(value=41)
    generator = StepByStepGenerator(mock_llm, manager, num_actions=2)
    first = counter_step("read_counter", {}, {"value": 0}, 0, 0)
    second = counter_step("read_counter", {}, {"value": 0}, 0, 0)
    first.step_number = 2
    second.step_number = 1

    result = generator.verify_invocation_order(
        "Read the counter twice",
        [first, second],
    )

    assert result["order_is_correct"] is False
    assert result["issues"]


def test_invocation_order_rejects_expected_tool_mismatch(
    mock_llm: MockLLMClient,
):
    manager = CounterToolManager(value=41)
    generator = StepByStepGenerator(mock_llm, manager, num_actions=1)
    step = counter_step("read_counter", {}, {"value": 0}, 0, 0)

    result = generator.verify_invocation_order(
        "Increment the counter",
        [step],
        ["increment_counter"],
    )

    assert result["order_is_correct"] is False
    string_issues = TypeAdapter(list[str]).validate_python(result["issues"])
    assert string_issues
    assert "Expected tool sequence" in string_issues[0]


def test_multi_turn_datapoint_contains_final_verification(
    monkeypatch: pytest.MonkeyPatch,
    mock_llm: MockLLMClient,
):
    manager = CounterToolManager(value=0)
    generator = MultiTurnGenerator(
        mock_llm,
        manager,
        num_turns=1,
        num_actions=1,
    )
    blueprint = DialogBlueprint(
        overall_task="Read the counter",
        num_turns=1,
        turns=[
            {
                "user_query": "Read the counter",
                "expected_tools": ["read_counter"],
            }
        ],
    )
    query = QueryGenerationResult(
        query="Read the counter",
        intent="Counter",
        expected_tools=["read_counter"],
    )
    trajectory = [
        TrajectoryStep(
            step_number=1,
            tool_calls=[
                ToolCallWithOutput(
                    tool_name="read_counter",
                    arguments={},
                    output={"value": 0},
                )
            ],
            pre_state={"counter": {"value": 0}},
            post_state={"counter": {"value": 0}},
        )
    ]
    def generate_blueprint(*_args: object) -> DialogBlueprint:
        return blueprint

    def generate_turn_query(**_kwargs: object) -> QueryGenerationResult:
        return query

    def generate_tools(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[list[TrajectoryStep], ObjectMap]:
        return trajectory, {}

    def generate_final_response(*_args: object) -> str:
        return "The counter is 0"

    monkeypatch.setattr(
        generator,
        "_stage0_generate_blueprint",
        generate_blueprint,
    )
    monkeypatch.setattr(
        generator,
        "_generate_turn_query",
        generate_turn_query,
    )
    monkeypatch.setattr(
        generator,
        "_stage2_generate_tools",
        generate_tools,
    )
    monkeypatch.setattr(
        generator,
        "_generate_final_response",
        generate_final_response,
    )

    datapoint = generator.generate_multi_turn_datapoint()

    assert datapoint is not None
    assert datapoint.verification_result is not None
    assert datapoint.verification_result["overall_verification_passed"] is True


def test_multi_turn_query_retries_control_blueprint_attempts(
    monkeypatch: pytest.MonkeyPatch,
    mock_llm: MockLLMClient,
):
    manager = CounterToolManager()
    generator = MultiTurnGenerator(
        mock_llm,
        manager,
        num_turns=1,
        num_actions=1,
    )
    observed_retries: list[int] = []

    def capture_retries(
        _focus_category: str | None = None,
        _initial_api_state: StateSnapshot | None = None,
        max_retries: int = 3,
    ) -> None:
        observed_retries.append(max_retries)

    monkeypatch.setattr(
        generator,
        "_stage0_generate_blueprint",
        capture_retries,
    )

    result = generator.generate_multi_turn_datapoint(query_retries=7)

    assert result is None
    assert observed_retries == [7]


def test_resumed_multi_turn_datapoint_contains_final_verification(
    mock_llm: MockLLMClient,
):
    manager = CounterToolManager(value=41)
    generator = MultiTurnGenerator(
        mock_llm,
        manager,
        num_turns=1,
        num_actions=1,
    )
    step = counter_step("read_counter", {}, {"value": 0}, 0, 0)
    conversation = MultiTurnConversation(
        overall_task="Read the counter",
        turns=[
            Turn(
                turn_number=1,
                user_query="Read the counter",
                steps=[step],
                assistant_response="The counter is 0",
            )
        ],
        initial_api_state={"counter": {"value": 0}},
    )
    blueprint = DialogBlueprint(
        overall_task="Read the counter",
        num_turns=1,
        turns=[
            {
                "user_query": "Read the counter",
                "expected_tools": ["read_counter"],
            }
        ],
    )
    checkpoint: ObjectMap = {
        "blueprint": blueprint.model_dump(),
        "partial_conversation": conversation.model_dump(),
        "execution_context": {},
        "completed_turns": 1,
        "initial_api_state": {"counter": {"value": 0}},
        "focus_category": "Counter",
    }

    datapoint = generator.continue_from_checkpoint(checkpoint)

    assert datapoint is not None
    assert datapoint.verification_result is not None
    assert datapoint.verification_result["overall_verification_passed"] is True


def test_blueprint_capability_judge_error_fails_closed(
    mock_llm: MockLLMClient,
    mock_tools: DeterministicTestToolManager,
):
    generator = BlueprintValidationGenerator(
        mock_llm,
        mock_tools,
        num_turns=1,
        num_actions=1,
    )
    mock_llm.set_responses(["not-json"])

    is_valid, issues = generator.verify_blueprint_capabilities(
        [
            {
                "user_query": "Find a flight",
                "expected_tools": ["search_flights"],
            }
        ]
    )

    assert is_valid is False
    assert issues
    assert "capability check error" in issues[0].lower()
