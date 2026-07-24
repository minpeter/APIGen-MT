"""Typed structural boundaries shared by step-by-step mixins."""

from __future__ import annotations

from typing import Protocol, TypeIs, override

from pydantic import TypeAdapter, ValidationError

from step_by_step_models import (
    ObjectMap,
    QueryGenerationResult,
    StateSnapshot,
    StateVerificationResult,
    StepByStepDatapoint,
    TokenUsageStats,
    TrajectoryStep,
    VerificationResult,
)

type LLMMessage = dict[str, str]

_OBJECT_MAP_ADAPTER: TypeAdapter[ObjectMap] = TypeAdapter(ObjectMap)


def parse_object_map(value: str) -> ObjectMap:
    return _OBJECT_MAP_ADAPTER.validate_json(value)


def is_object_map(value: object) -> TypeIs[ObjectMap]:
    try:
        _ = _OBJECT_MAP_ADAPTER.validate_python(value, strict=True)
    except ValidationError:
        return False
    return True


def is_object_list(value: object) -> TypeIs[list[object]]:
    return isinstance(value, list)


def is_object_tuple(value: object) -> TypeIs[tuple[object, ...]]:
    return isinstance(value, tuple)


def placeholder_keys(value: str) -> list[str]:
    keys: list[str] = []
    position = 0
    while (start := value.find("{{", position)) >= 0:
        end = value.find("}}", start + 2)
        if end < 0:
            break
        key = value[start + 2:end]
        if "{" not in key and "}" not in key:
            keys.append(key)
        position = end + 2
    return keys


class ToolInstance(Protocol):
    """Object supporting typed dynamic attribute access."""

    @override
    def __getattribute__(self, name: str) -> object: ...

    @override
    def __setattr__(self, name: str, value: object) -> None: ...


def is_tool_instance(value: object) -> TypeIs[ToolInstance]:
    return value is not None


class LLM(Protocol):
    def generate(
        self,
        messages: list[LLMMessage],
        **kwargs: object,
    ) -> str: ...

    def get_token_usage(self) -> dict[str, int]: ...


class StepByStepToolManager(Protocol):
    python_tool_instances: dict[str, object]
    api_name_to_class_key: dict[str, str]

    def get_api_state(self) -> StateSnapshot: ...
    def restore_api_state(self, state: StateSnapshot) -> None: ...
    def initialize_api_state(self, force_new: bool = False) -> None: ...
    def has_python_implementation(self, tool_name: str) -> bool: ...
    def invoke_python_tool(self, tool_name: str, params: ObjectMap) -> object: ...
    def invoke_tool(self, tool_name: str, params: ObjectMap) -> object: ...
    def get_tool_schema(self, tool_name: str) -> ObjectMap: ...
    def get_tool_category(self, tool_name: str) -> str | None: ...
    def get_tools_json_schema(self) -> list[ObjectMap]: ...
    def get_tools_by_category(self, category: str) -> list[ObjectMap]: ...
    def get_categories(self) -> list[str]: ...
    def tool_exists(self, tool_name: str) -> bool: ...


class StepByStepMixinBase(Protocol):
    """Common state and cross-mixin method surface for extracted mixins."""

    llm: LLM
    judge: LLM
    tool_manager: StepByStepToolManager
    validate_outputs: bool
    target_num_actions: int
    _python_tools_available: bool
    _accumulated_prompt_tokens: int
    _accumulated_completion_tokens: int
    _accumulated_total_tokens: int
    _accumulated_llm_calls: int
    _initial_token_usage: dict[str, int] | None

    def _safe_llm_generate(
        self,
        messages: list[LLMMessage],
        max_retries: int = 5,
        llm: LLM | None = None,
        **kwargs: object,
    ) -> str: ...

    def _get_tool_schemas_str(self, tools_subset: list[str] | None = None) -> str: ...
    def _get_example_queries(self) -> str: ...
    def validate_expected_tools(
        self,
        query: str,
        expected_tools: list[str],
        intent: str,
    ) -> tuple[bool, str]: ...
    def _process_placeholders(
        self,
        arguments: ObjectMap,
        execution_context: ObjectMap,
    ) -> ObjectMap: ...
    def _apply_state_modifications(self, modifications: ObjectMap) -> int: ...
    def _generate_tool_arguments(
        self,
        tool_name: str,
        query: str,
        trajectory: list[TrajectoryStep],
        execution_context: ObjectMap,
        feedback: str | None = None,
    ) -> tuple[ObjectMap | None, str | None]: ...
    def _verify_tool_query_consistency(
        self,
        tool_name: str,
        arguments: ObjectMap,
        query: str,
        trajectory: list[TrajectoryStep],
        execution_context: ObjectMap,
    ) -> tuple[bool, str]: ...
    def _simulate_tool_execution(
        self,
        tool_name: str,
        arguments: ObjectMap,
        execution_context: ObjectMap,
    ) -> object: ...

    @staticmethod
    def _detect_tool_error(
        tool_name: str,
        output: ObjectMap,
    ) -> tuple[bool, str]: ...

    def verify_output_consistency(
        self,
        tool_name: str,
        step_number: int,
        output: object,
        expected_type: str,
        expected_description: str,
    ) -> ObjectMap: ...
    def verify_state_transition(
        self,
        tool_name: str,
        tool_arguments: ObjectMap,
        tool_output: object,
        pre_state: StateSnapshot,
        post_state: StateSnapshot,
    ) -> StateVerificationResult: ...
    def _replay_state(
        self,
        trajectory: list[TrajectoryStep],
        baseline_state: StateSnapshot | None = None,
    ) -> None: ...
    def run_full_verification(
        self,
        query: str,
        trajectory: list[TrajectoryStep],
        execution_context: ObjectMap,
        initial_api_state: StateSnapshot | None = None,
        expected_tools: list[str] | None = None,
    ) -> VerificationResult: ...
    def _generate_final_response(
        self,
        query: str,
        trajectory: list[TrajectoryStep],
        execution_context: ObjectMap,
    ) -> str: ...
    def _update_token_usage(self) -> None: ...
    def _get_token_stats(self) -> TokenUsageStats: ...
    def _stage1_generate_query(
        self,
        focus_category: str | None,
        context_hint: str | None,
        max_retries: int,
        query_seed: ObjectMap | None = None,
        initial_api_state: StateSnapshot | None = None,
    ) -> QueryGenerationResult | None: ...
    def _stage2_generate_tools(
        self,
        query_result: QueryGenerationResult,
        max_retries_per_tool: int,
        initial_execution_context: ObjectMap | None = None,
    ) -> tuple[list[TrajectoryStep] | None, ObjectMap | None]: ...
    def _stage3_finalize(
        self,
        query_result: QueryGenerationResult,
        trajectory: list[TrajectoryStep],
        execution_context: ObjectMap,
        focus_category: str | None,
        initial_api_state: StateSnapshot | None = None,
    ) -> StepByStepDatapoint | None: ...
