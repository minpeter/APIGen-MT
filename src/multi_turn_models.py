"""Pydantic models for multi-turn generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.step_by_step_models import TokenUsageStats, TrajectoryStep
else:
    from step_by_step_models import TokenUsageStats, TrajectoryStep


type ApiState = dict[str, dict[str, object]]


class Turn(BaseModel):
    """A single user-assistant turn in a multi-turn conversation."""

    turn_number: int
    user_query: str
    query_intent: str = ""
    steps: list[TrajectoryStep] = Field(default_factory=list)
    assistant_response: str = ""
    expected_tools: list[str] = Field(default_factory=list)
    execution_context: dict[str, object] = Field(default_factory=dict)


class MultiTurnConversation(BaseModel):
    """Complete multi-turn conversation trajectory."""

    overall_task: str = ""
    turns: list[Turn] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    categories_used: list[str] = Field(default_factory=list)
    initial_api_state: ApiState | None = None


class MultiTurnDatapoint(BaseModel):
    """Complete multi-turn datapoint."""

    conversation: MultiTurnConversation
    generation_metadata: dict[str, object] = Field(default_factory=dict)
    verification_result: dict[str, object] | None = None
    token_usage: TokenUsageStats = Field(default_factory=TokenUsageStats)
    initial_api_state: ApiState | None = None


class DialogBlueprint(BaseModel):
    """Blueprint for a multi-turn dialog."""

    overall_task: str
    num_turns: int
    turns: list[dict[str, object]] = Field(default_factory=list)
