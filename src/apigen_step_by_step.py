"""Compatibility facade for the composed step-by-step generator."""

import json
import random
import time
from typing import override

import requests

from config_pool import CITIES, PERSONAS
from llm_remote_client import LLMClient
from step_by_step_argument_generation import ArgumentGenerationMixin
from step_by_step_finalization import FinalizationMixin
from step_by_step_models import (
    ConversationTrajectory,
    DeterministicReplayResult,
    ObjectMap,
    QueryGenerationResult,
    ReplayIssue,
    StateSnapshot,
    StateVerificationResult,
    StepByStepDatapoint,
    StepSelectionResult,
    TokenUsageStats,
    ToolCallWithOutput,
    TrajectoryStep,
    VerificationResult,
)
from step_by_step_output_verification import OutputVerificationMixin
from step_by_step_placeholders import PlaceholderProcessingMixin
from step_by_step_protocols import (
    LLM,
    LLMMessage,
    StepByStepToolManager,
    is_object_map,
)
from step_by_step_query_generation import QueryGenerationMixin
from step_by_step_query_validation import QueryValidationMixin
from step_by_step_runtime import RuntimeMixin
from step_by_step_state_management import StateManagementMixin
from step_by_step_state_preparation import StatePreparationMixin
from step_by_step_tool_generation import ToolGenerationMixin
from step_by_step_tool_steps import ToolStepsMixin
from step_by_step_verification import VerificationMixin
from step_by_step_verification_state import StateVerificationMixin

__all__ = [
    "ConversationTrajectory", "DeterministicReplayResult",
    "LLMClient",
    "QueryGenerationResult",
    "ReplayIssue",
    "StateVerificationResult",
    "StepByStepDatapoint",
    "StepByStepGenerator",
    "StepSelectionResult",
    "TokenUsageStats",
    "ToolCallWithOutput",
    "TrajectoryStep",
    "VerificationResult",
]


class StepByStepGenerator(
    RuntimeMixin,
    PlaceholderProcessingMixin,
    QueryValidationMixin,
    QueryGenerationMixin,
    ToolStepsMixin,
    ArgumentGenerationMixin,
    ToolGenerationMixin,
    StatePreparationMixin,
    StateManagementMixin,
    FinalizationMixin,
    OutputVerificationMixin,
    VerificationMixin,
    StateVerificationMixin,
):
    """Create datapoints step by step with immediate tool simulation."""

    def __init__(
        self,
        llm_client: LLM,
        tool_manager: StepByStepToolManager,
        validate_outputs: bool = True,
        judge_client: LLM | None = None,
        num_actions: int = 1,
    ) -> None:
        self.llm: LLM = llm_client
        self.judge: LLM = judge_client or llm_client
        self.tool_manager: StepByStepToolManager = tool_manager
        self.validate_outputs: bool = validate_outputs
        self.target_num_actions: int = max(1, int(num_actions or 1))
        self._python_tools_available: bool = bool(
            tool_manager.python_tool_instances
        )
        self._accumulated_prompt_tokens: int = 0
        self._accumulated_completion_tokens: int = 0
        self._accumulated_total_tokens: int = 0
        self._accumulated_llm_calls: int = 0
        self._initial_token_usage: dict[str, int] | None = None

    @override
    def _safe_llm_generate(
        self,
        messages: list[LLMMessage],
        max_retries: int = 5,
        llm: LLM | None = None,
        **kwargs: object,
    ) -> str:
        """Call the LLM with application-level transient-error retries."""
        client = llm or self.llm
        for attempt in range(max_retries):
            try:
                result = client.generate(messages, **kwargs)
                if not result.strip():
                    raise ValueError("LLM returned empty response")
                return result
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
            ) as exc:
                delay = min(2 * (1 << attempt), 60) + random.uniform(0, 2)
                print("".join((
                    " [_safe_llm_generate] Transient error ",
                    f"(attempt {attempt + 1}/{max_retries}): ",
                    f"{type(exc).__name__}: {exc}, retrying in {delay:.1f}s...",
                )))
                time.sleep(delay)
            except json.JSONDecodeError as exc:
                delay = min(2 * (1 << attempt), 30) + random.uniform(0, 1)
                print("".join((
                    " [_safe_llm_generate] JSON Error ",
                    f"(attempt {attempt + 1}/{max_retries}): {exc}",
                )))
                time.sleep(delay)
            except (RuntimeError, ValueError) as exc:
                delay = min(2 * (1 << attempt), 30) + random.uniform(0, 1)
                print("".join((
                    " [_safe_llm_generate] Error ",
                    f"(attempt {attempt + 1}/{max_retries}): ",
                    f"{type(exc).__name__}: {exc}, retrying in {delay:.1f}s...",
                )))
                time.sleep(delay)
        raise RuntimeError(
            f"LLM generate failed after {max_retries} application-level retries"
        )

    def generate_datapoint(
        self,
        focus_category: str | None = None,
        context_hint: str | None = None,
        query_retries: int = 5,
        tool_retries: int = 3,
    ) -> StepByStepDatapoint | None:
        """Run query generation, state preparation, tools, and finalization."""
        print("\n" + "=" * 70)
        print("STEP-BY-STEP DATAPOINT GENERATION (Refactored)")
        print("=" * 70)
        self._reset_token_tracking()
        self._capture_initial_usage()

        query_seed: ObjectMap = {
            "persona": random.choice(PERSONAS),
            "city": random.choice(CITIES),
        }
        persona = query_seed.get("persona")
        city = query_seed.get("city")
        persona_name = persona.get("name", "") if is_object_map(persona) else ""
        city_name = city.get("city", "") if is_object_map(city) else ""
        print(f" Persona seed: {persona_name}, {city_name}")

        initial_api_state: StateSnapshot | None = None
        if self._python_tools_available:
            self.tool_manager.initialize_api_state(force_new=True)
            initial_api_state = self.tool_manager.get_api_state()
            print(f" Captured initial API state ({len(initial_api_state)} class keys)")

        print("\n" + "-" * 70)
        print("STAGE 1: Generate and Verify Query")
        print("-" * 70)
        query_result = self._stage1_generate_query(
            focus_category,
            context_hint,
            query_retries,
            query_seed,
            initial_api_state,
        )
        if query_result is None:
            print("\n✗ Stage 1 failed: Could not generate valid query")
            print(
                "  Token usage for failed datapoint: "
                + f"{self._accumulated_total_tokens:,} tokens, "
                + f"{self._accumulated_llm_calls} calls"
            )
            return None

        self._update_token_usage()
        print("\n✓ Stage 1 complete: Query generated and verified")
        print(f" Query: {query_result.query}")
        print(f" Expected tools: {query_result.expected_tools}")
        print(f" Tokens so far: {self._accumulated_total_tokens:,}")

        if self._python_tools_available:
            print("\n" + "-" * 70)
            print("STAGE 1.4: Ensure User Identity Coherence")
            print("-" * 70)
            if self._ensure_user_identity_coherence(query_result.query):
                initial_api_state = self.tool_manager.get_api_state()
                print(
                    f" ✓ Identity coherence adjusted, re-captured ({len(initial_api_state)} class keys)"
                )
            else:
                print(" No identity adjustment needed")

        if self._python_tools_available and query_result.expected_tools:
            print("\n" + "-" * 70)
            print("STAGE 1.5: Adjust Initial API State")
            print("-" * 70)
            if self._stage1_5_adjust_initial_state(query_result):
                initial_api_state = self.tool_manager.get_api_state()
                print(
                    f" ✓ API state adjusted, re-captured ({len(initial_api_state)} class keys)"
                )
                self._update_token_usage()
                print(f" Tokens so far: {self._accumulated_total_tokens:,}")
            else:
                print(" ⚠ State adjustment failed or not needed")

        print("\n" + "-" * 70)
        print("STAGE 2: Generate Tool Invocations")
        print("-" * 70)
        trajectory, execution_context = self._stage2_generate_tools(
            query_result,
            tool_retries,
        )
        if trajectory is None or execution_context is None:
            print("\n✗ Stage 2 failed: Could not generate all tool invocations")
            return None

        self._update_token_usage()
        print(f"\n✓ Stage 2 complete: Generated {len(trajectory)} tool invocations")
        print(f"  Tokens so far: {self._accumulated_total_tokens:,}")
        print("\n" + "-" * 70)
        print("STAGE 3: Finalize Datapoint")
        print("-" * 70)
        datapoint = self._stage3_finalize(
            query_result,
            trajectory,
            execution_context,
            focus_category,
            initial_api_state,
        )
        if datapoint is None:
            print("\n✗ Stage 3 failed: Could not finalize datapoint")
            return None

        print("\n" + "=" * 70)
        print("✓ DATAPOINT GENERATION COMPLETE (VERIFIED)")
        print("=" * 70)
        print(f" Query: {datapoint.trajectory.query}")
        print(f" Tools used: {datapoint.trajectory.tools_used}")
        print(f" Steps: {len(datapoint.trajectory.steps)}")
        print(" Verification: PASSED")
        return datapoint


if __name__ == "__main__":
    import runpy

    _ = runpy.run_module("step_by_step_cli", run_name="__main__")
