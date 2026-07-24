"""Compatibility facade for composed multi-turn generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.apigen_step_by_step import StepByStepGenerator
    from src.domain_hints import get_domain_hints
    from src.multi_turn_assistant_responses import (
        AssistantResponsesMixin as _AssistantResponsesMixin,
    )
    from src.multi_turn_blueprint_context import (
        BlueprintContextMixin as _BlueprintContextMixin,
    )
    from src.multi_turn_blueprints import (
        BlueprintGenerationMixin as _BlueprintGenerationMixin,
    )
    from src.multi_turn_capability_validation import (
        CapabilityValidationMixin as _CapabilityValidationMixin,
    )
    from src.multi_turn_checkpoint import (
        CheckpointRestorationMixin as _CheckpointRestorationMixin,
    )
    from src.multi_turn_consistency import (
        ConsistencyValidationMixin as _ConsistencyValidationMixin,
    )
    from src.multi_turn_entity_validation import (
        EntityValidationMixin as _EntityValidationMixin,
    )
    from src.multi_turn_generation import TurnGenerationMixin as _TurnGenerationMixin
    from src.multi_turn_models import (
        DialogBlueprint,
        MultiTurnConversation,
        MultiTurnDatapoint,
        Turn,
    )
    from src.multi_turn_user_simulation import (
        UserSimulationMixin as _UserSimulationMixin,
    )
    from src.step_by_step_models import (
        ConversationTrajectory,
        QueryGenerationResult,
        StepByStepDatapoint,
        TokenUsageStats,
        ToolCallWithOutput,
        TrajectoryStep,
    )
    from src.tool_manager import filter_api_state
    from step_by_step_protocols import LLM as LLMClient
    from step_by_step_protocols import StepByStepToolManager as ToolManager
else:
    from apigen_step_by_step import StepByStepGenerator
    from domain_hints import get_domain_hints
    from llm_client import LLMClient
    from multi_turn_assistant_responses import (
        AssistantResponsesMixin as _AssistantResponsesMixin,
    )
    from multi_turn_blueprint_context import (
        BlueprintContextMixin as _BlueprintContextMixin,
    )
    from multi_turn_blueprints import (
        BlueprintGenerationMixin as _BlueprintGenerationMixin,
    )
    from multi_turn_capability_validation import (
        CapabilityValidationMixin as _CapabilityValidationMixin,
    )
    from multi_turn_checkpoint import (
        CheckpointRestorationMixin as _CheckpointRestorationMixin,
    )
    from multi_turn_consistency import (
        ConsistencyValidationMixin as _ConsistencyValidationMixin,
    )
    from multi_turn_entity_validation import (
        EntityValidationMixin as _EntityValidationMixin,
    )
    from multi_turn_generation import TurnGenerationMixin as _TurnGenerationMixin
    from multi_turn_models import (
        DialogBlueprint,
        MultiTurnConversation,
        MultiTurnDatapoint,
        Turn,
    )
    from multi_turn_user_simulation import UserSimulationMixin as _UserSimulationMixin
    from step_by_step_models import (
        ConversationTrajectory,
        QueryGenerationResult,
        StepByStepDatapoint,
        TokenUsageStats,
        ToolCallWithOutput,
        TrajectoryStep,
    )
    from tool_manager import ToolManager, filter_api_state

__all__ = [
    "ConversationTrajectory",
    "DialogBlueprint",
    "LLMClient",
    "MultiTurnConversation",
    "MultiTurnDatapoint",
    "MultiTurnGenerator",
    "QueryGenerationResult",
    "StepByStepDatapoint",
    "TokenUsageStats",
    "ToolCallWithOutput",
    "ToolManager",
    "TrajectoryStep",
    "Turn",
    "filter_api_state",
    "get_domain_hints",
]

# Keep model introspection and pickling paths compatible with the original facade.
for _model in (Turn, MultiTurnConversation, MultiTurnDatapoint, DialogBlueprint):
    _model.__module__ = __name__


class MultiTurnGenerator(
    _CheckpointRestorationMixin,
    _TurnGenerationMixin,
    _BlueprintGenerationMixin,
    _BlueprintContextMixin,
    _UserSimulationMixin,
    _AssistantResponsesMixin,
    _CapabilityValidationMixin,
    _EntityValidationMixin,
    _ConsistencyValidationMixin,
    StepByStepGenerator,
):
    """Generate conversations containing multiple tool-backed user turns."""

    num_turns: int

    def __init__(
        self,
        llm_client: LLMClient,
        tool_manager: ToolManager,
        num_turns: int = 2,
        validate_outputs: bool = True,
        num_actions: int = 1,
        judge_client: LLMClient | None = None,
    ) -> None:
        super().__init__(
            llm_client,
            tool_manager,
            validate_outputs,
            judge_client=judge_client,
            num_actions=num_actions,
        )
        self.num_turns = num_turns
