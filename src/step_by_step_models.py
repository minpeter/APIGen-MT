"""Serialized models shared by the step-by-step generation pipeline."""

from typing import Literal

from pydantic import BaseModel, Field

type ObjectMap = dict[str, object]
type StateSnapshot = dict[str, ObjectMap]


class ToolCallWithOutput(BaseModel):
    """A single tool call with its simulated output."""

    tool_name: str
    arguments: ObjectMap = Field(default_factory=dict)
    output: object = None


class StateVerificationResult(BaseModel):
    """LLM-as-judge verdict on a single state transition."""

    is_valid: bool = True
    reasoning: str = ""
    issues: list[str] = Field(default_factory=list)
    state_changes_summary: str = ""


class TrajectoryStep(BaseModel):
    """A single step in the conversation trajectory."""

    step_number: int
    tool_calls: list[ToolCallWithOutput] = Field(default_factory=list)
    reasoning: str | None = None
    pre_state: StateSnapshot | None = None
    post_state: StateSnapshot | None = None
    state_verification: StateVerificationResult | None = None


class ConversationTrajectory(BaseModel):
    """Complete conversation trajectory for a datapoint."""

    query: str
    steps: list[TrajectoryStep] = Field(default_factory=list)
    final_response: str
    tools_used: list[str] = Field(default_factory=list)
    categories_used: list[str] = Field(default_factory=list)
    initial_api_state: StateSnapshot | None = None


class TokenUsageStats(BaseModel):
    """Token usage statistics for a single datapoint."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_llm_calls: int = 0


class StepByStepDatapoint(BaseModel):
    """Complete datapoint generated step-by-step."""

    trajectory: ConversationTrajectory
    generation_metadata: ObjectMap = Field(default_factory=dict)
    verification_result: ObjectMap | None = None
    token_usage: TokenUsageStats = Field(default_factory=TokenUsageStats)
    initial_api_state: StateSnapshot | None = None
    intermediate_api_states: list[ObjectMap] = Field(default_factory=list)


class ReplayIssue(BaseModel):
    step_number: int
    tool_name: str
    check: str
    expected: object = None
    actual: object = None
    error: str | None = None


class DeterministicReplayResult(BaseModel):
    status: Literal["verified", "failed", "unavailable"]
    is_valid: bool
    checked_calls: int = 0
    unavailable_tools: list[str] = Field(default_factory=list)
    issues: list[ReplayIssue] = Field(default_factory=list)
    final_state_digest: str | None = None


class VerificationResult(BaseModel):
    """Complete verification result for a generated datapoint."""

    query: str
    tool_relevance_checks: list[ObjectMap] = Field(default_factory=list)
    order_is_correct: bool
    order_verification_details: str = ""
    output_validations: list[ObjectMap] = Field(default_factory=list)
    placeholder_resolution: ObjectMap = Field(default_factory=dict)
    deterministic_replay: DeterministicReplayResult | None = None
    overall_verification_passed: bool
    verification_summary: str = ""


class StepSelectionResult(BaseModel):
    """Result of LLM selecting the next tool/step."""

    tool_name: str
    arguments: ObjectMap = Field(default_factory=dict)
    reasoning: str


class QueryGenerationResult(BaseModel):
    """Result of generating a user query."""

    query: str
    intent: str
    expected_tools: list[str] = Field(default_factory=list)


class ValidationResponse(BaseModel):
    """Structured response shared by LLM validation checks."""

    is_valid: bool = False
    issues: list[str] = Field(default_factory=list)


class StateAdjustmentResponse(BaseModel):
    """Structured initial-state adjustment proposed by the LLM."""

    modifications: ObjectMap = Field(default_factory=dict)
    reasoning: str = ""
