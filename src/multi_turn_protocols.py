"""Typed contracts shared by the runtime-neutral multi-turn mixins."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, TypeGuard, override, runtime_checkable

if TYPE_CHECKING:
    from src.multi_turn_models import (
        DialogBlueprint,
        MultiTurnConversation,
        MultiTurnDatapoint,
        Turn,
    )
    from step_by_step_models import (
        QueryGenerationResult,
        TokenUsageStats,
        ToolCallWithOutput,
        TrajectoryStep,
        VerificationResult,
    )
    from step_by_step_protocols import LLM, LLMMessage, StepByStepToolManager


type ApiState = dict[str, dict[str, object]]
type BlueprintTurn = dict[str, object]
type Checkpoint = dict[str, object]
type CheckpointCallback = Callable[[Checkpoint], None]
type ExecutionContext = dict[str, object]
type ObjectDict = dict[object, object]


@runtime_checkable
class DynamicAttributeSource(Protocol):
    """Object whose runtime attributes are intentionally data-driven."""

    @override
    def __getattribute__(self, name: str) -> object: ...


@runtime_checkable
class DynamicCallable(Protocol):
    """Callable discovered through a runtime tool registry."""

    def __call__(self, *args: object, **kwargs: object) -> object: ...


class ToolCallView(Protocol):
    """Typed read-only view over the permissive serialized tool-call model."""

    @property
    def tool_name(self) -> str: ...

    @property
    def arguments(self) -> dict[str, object]: ...

    @property
    def output(self) -> object: ...


class VerificationResultView(Protocol):
    """Final-verification result consumed by multi-turn assembly."""

    @property
    def overall_verification_passed(self) -> bool: ...

    def model_dump(self) -> dict[str, object]: ...


@runtime_checkable
class MultiTurnFacade(Protocol):
    """Public model seam retained for monkeypatching and pickle paths."""

    DialogBlueprint: type[DialogBlueprint]
    MultiTurnConversation: type[MultiTurnConversation]
    MultiTurnDatapoint: type[MultiTurnDatapoint]
    Turn: type[Turn]

    def filter_api_state(
        self,
        full_state: ApiState,
        tool_names: list[str],
    ) -> ApiState: ...

    def get_domain_hints(self, category: str) -> str: ...


if TYPE_CHECKING:
    def _missing_tool_manager() -> StepByStepToolManager:
        raise NotImplementedError


    class GeneratorMixinBase:
        """Members supplied by the composed generator and its base facade."""

        num_turns: int = 0
        target_num_actions: int = 0
        _python_tools_available: bool = False
        tool_manager: StepByStepToolManager = _missing_tool_manager()

        def _reset_token_tracking(self) -> None: ...

        def _capture_initial_usage(self) -> None: ...

        def _update_token_usage(self) -> None: ...

        def _get_token_stats(self) -> TokenUsageStats: ...

        def _safe_llm_generate(
            self,
            messages: list[LLMMessage],
            max_retries: int = 5,
            llm: LLM | None = None,
            **kwargs: object,
        ) -> str:
            del messages, max_retries, llm, kwargs
            raise NotImplementedError

        def _stage0_generate_blueprint(
            self,
            focus_category: str | None = None,
            initial_api_state: ApiState | None = None,
            max_retries: int = 3,
        ) -> DialogBlueprint | None:
            del focus_category, initial_api_state, max_retries
            raise NotImplementedError

        def _ensure_user_identity_coherence(self, text: str) -> bool:
            del text
            raise NotImplementedError

        def _generate_turn_query(
            self,
            blueprint: DialogBlueprint,
            conversation: MultiTurnConversation,
            turn_index: int,
        ) -> QueryGenerationResult | None:
            del blueprint, conversation, turn_index
            raise NotImplementedError

        def _stage2_generate_tools(
            self,
            query_result: QueryGenerationResult,
            max_retries_per_tool: int,
            initial_execution_context: ExecutionContext | None = None,
        ) -> tuple[list[TrajectoryStep] | None, ExecutionContext]:
            del query_result, max_retries_per_tool, initial_execution_context
            raise NotImplementedError

        @staticmethod
        def _validate_tool_arguments(
            trajectory: list[TrajectoryStep],
        ) -> list[str]:
            del trajectory
            raise NotImplementedError

        def _validate_cross_turn_consistency(
            self,
            trajectory: list[TrajectoryStep],
            execution_context: ExecutionContext,
        ) -> list[str]:
            del trajectory, execution_context
            raise NotImplementedError

        def _complete_generated_turn(
            self,
            turn_number: int,
            query_result: QueryGenerationResult,
            trajectory: list[TrajectoryStep],
            turn_execution_context: ExecutionContext,
            execution_context: ExecutionContext,
            tools_used: set[str],
            categories_used: set[str],
        ) -> Turn:
            del (
                turn_number,
                query_result,
                trajectory,
                turn_execution_context,
                execution_context,
                tools_used,
                categories_used,
            )
            raise NotImplementedError

        def _generate_final_response(
            self,
            query: str,
            trajectory: list[TrajectoryStep],
            execution_context: ExecutionContext,
        ) -> str:
            del query, trajectory, execution_context
            raise NotImplementedError

        def _verify_blueprint_capabilities(
            self,
            turns: list[BlueprintTurn],
            focus_category: str | None = None,
            initial_api_state: ApiState | None = None,
        ) -> tuple[bool, list[str]]:
            del turns, focus_category, initial_api_state
            raise NotImplementedError

        def _validate_posting_api_entities(
            self,
            turns: list[BlueprintTurn],
            initial_api_state: ApiState | None = None,
        ) -> list[str]:
            del turns, initial_api_state
            raise NotImplementedError

        def _validate_vehicle_control_queries(
            self,
            turns: list[BlueprintTurn],
            initial_api_state: ApiState | None = None,
        ) -> list[str]:
            del turns, initial_api_state
            raise NotImplementedError

        def run_full_verification(
            self,
            query: str,
            trajectory: list[TrajectoryStep],
            execution_context: ExecutionContext,
            initial_api_state: ApiState | None = None,
            expected_tools: list[str] | None = None,
        ) -> VerificationResult:
            del (
                query,
                trajectory,
                execution_context,
                initial_api_state,
                expected_tools,
            )
            raise NotImplementedError
else:
    class GeneratorMixinBase:
        """Empty runtime base; composition supplies the declared members."""


def is_dynamic_attribute_source(
    source: object,
) -> TypeGuard[DynamicAttributeSource]:
    """Narrow a registered instance to its dynamic attribute interface."""
    return hasattr(source, "__getattribute__")


def get_public_facade() -> MultiTurnFacade:
    """Return the compatibility facade after checking its public model seam."""
    module = import_module("apigen_multi_turn")
    if not isinstance(module, MultiTurnFacade):
        raise TypeError("apigen_multi_turn does not expose the multi-turn facade")
    return module


def is_object_dict(value: object) -> TypeGuard[ObjectDict]:
    """Narrow an unstructured nested value to a dictionary."""
    return isinstance(value, dict)


def is_object_list(value: object) -> TypeGuard[list[object]]:
    """Narrow an unstructured nested value to a list."""
    return isinstance(value, list)


def string_list(mapping: Mapping[str, object], key: str) -> list[str]:
    """Read a list of strings from an unstructured mapping."""
    value = mapping.get(key)
    if not is_object_list(value):
        return []
    return [item for item in value if isinstance(item, str)]


def string_value(
    mapping: Mapping[str, object],
    key: str,
    default: str = "",
) -> str:
    """Read a string from an unstructured mapping."""
    value = mapping.get(key, default)
    return value if isinstance(value, str) else default


def tool_call_view(tool_call: ToolCallWithOutput) -> ToolCallView:
    """Expose a permissive Pydantic tool call through a typed protocol."""
    return tool_call


def verification_result_view(
    result: VerificationResult,
) -> VerificationResultView:
    """Expose the permissive verification model through a typed protocol."""
    return result
