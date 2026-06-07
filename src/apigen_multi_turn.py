"""Multi-turn conversation generator with step-by-step tool simulation.

Extends StepByStepGenerator to produce multi-turn conversations where
a separate LLM generates each user turn based on the dialog blueprint
and the current point in the conversation.
"""

import json
import copy
import time
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from apigen_step_by_step import (
    StepByStepGenerator,
    StepByStepDatapoint,
    ConversationTrajectory,
    TokenUsageStats,
    TrajectoryStep,
    ToolCallWithOutput,
    QueryGenerationResult,
    filter_api_state,
)
from llm_client import LLMClient
from tool_manager import ToolManager


class Turn(BaseModel):
    """A single user-assistant turn in a multi-turn conversation."""

    turn_number: int
    user_query: str
    query_intent: str = ""
    steps: List[TrajectoryStep] = Field(default_factory=list)
    assistant_response: str = ""
    expected_tools: List[str] = Field(default_factory=list)
    execution_context: Dict[str, Any] = Field(default_factory=dict)


class MultiTurnConversation(BaseModel):
    """Complete multi-turn conversation trajectory."""

    overall_task: str = ""
    turns: List[Turn] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    categories_used: List[str] = Field(default_factory=list)
    initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None


class MultiTurnDatapoint(BaseModel):
    """Complete multi-turn datapoint."""

    conversation: MultiTurnConversation
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)
    verification_result: Optional[Dict[str, Any]] = None
    token_usage: TokenUsageStats = Field(default_factory=TokenUsageStats)
    initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None


class DialogBlueprint(BaseModel):
    """Blueprint for a multi-turn dialog."""

    overall_task: str
    num_turns: int
    turns: List[Dict[str, Any]] = Field(default_factory=list)


class MultiTurnGenerator(StepByStepGenerator):
    """Generator for multi-turn conversations with step-by-step tool simulation.

    Flow:
      1. Generate dialog blueprint (overall task + per-turn goals)
      2. For each turn:
         a. Generate user query based on blueprint + conversation history
         b. Stage 1.5: Adjust API state for expected tools
         c. Stage 2: Generate and execute tool invocations
         d. Generate assistant response
         e. Persist execution context for next turn
      3. Assemble multi-turn datapoint
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_manager: ToolManager,
        num_turns: int = 2,
        actions_per_turn: int = 2,
        validate_outputs: bool = True,
    ):
        super().__init__(llm_client, tool_manager, actions_per_turn, validate_outputs)
        self.num_turns = num_turns

    # ─────────────────────── Public entry point ───────────────────────

    def generate_multi_turn_datapoint(
            self,
            focus_category: Optional[str] = None,
            query_retries: int = 3,
            tool_retries: int = 3,
    ) -> Optional[MultiTurnDatapoint]:
        """Generate a multi-turn datapoint."""

        self._reset_token_tracking()
        self._capture_initial_usage()

        # Initialize API state for the whole conversation
        initial_api_state = None
        if self._python_tools_available:
            self.tool_manager.initialize_api_state()
            initial_api_state = self.tool_manager.get_api_state()
            print(f" Captured initial API state ({len(initial_api_state)} class keys)")

        # Stage 0: Generate dialog blueprint
        print("\n" + "=" * 70)
        print("STAGE 0: Generate Dialog Blueprint")
        print("=" * 70)
        blueprint = self._stage0_generate_blueprint(focus_category)
        if blueprint is None:
            print("✗ Stage 0 failed: Could not generate dialog blueprint")
            return None
        self._update_token_usage()
        print(f" Overall task: {blueprint.overall_task}")
        for i, t in enumerate(blueprint.turns, 1):
            uq = t.get('user_query', '')
            print(f"   Turn {i}: {uq[:80]}...")

        conversation = MultiTurnConversation(overall_task=blueprint.overall_task)
        execution_context: Dict[str, Any] = {}
        tools_used = set()
        categories_used = set()

        for turn_idx in range(blueprint.num_turns):
            print(f"\n{'=' * 70}")
            print(f"TURN {turn_idx + 1}/{blueprint.num_turns}")
            print("=" * 70)

            turn_spec = blueprint.turns[turn_idx] if turn_idx < len(blueprint.turns) else {}

            # Stage 1: Generate user query for this turn
            query_result = self._generate_turn_query(
                blueprint=blueprint,
                conversation=conversation,
                turn_index=turn_idx,
            )
            if query_result is None:
                print(f"✗ Turn {turn_idx + 1} failed: Could not generate query")
                return None
            self._update_token_usage()

            # Stage 1.5: Adjust API state for this turn's expected tools
            if self._python_tools_available and query_result.expected_tools:
                print(f"\n Adjusting API state for turn {turn_idx + 1}...")
                adjusted = self._stage1_5_adjust_initial_state(query_result)
                if adjusted:
                    print(" ✓ State adjusted")
                else:
                    print(" ⚡ No adjustment needed")

            # Stage 2: Generate and execute tool invocations
            trajectory, ec = self._stage2_generate_tools(query_result, tool_retries)
            if trajectory is None:
                print(f"✗ Turn {turn_idx + 1} failed: Could not generate tool calls")
                return None
            self._update_token_usage()

            # Merge turn context into persistent execution_context
            for k, v in ec.items():
                execution_context[k] = v

            # Generate assistant response for this turn
            assistant_response = self._generate_final_response(
                query_result.query, trajectory, execution_context
            )
            self._update_token_usage()

            # Collect tools and categories
            for step in trajectory:
                for tc in step.tool_calls:
                    tools_used.add(tc.tool_name)
                    cat = self.tool_manager.get_tool_category(tc.tool_name)
                    if cat:
                        categories_used.add(cat)

            turn = Turn(
                turn_number=turn_idx + 1,
                user_query=query_result.query,
                query_intent=query_result.intent,
                steps=trajectory,
                assistant_response=assistant_response,
                expected_tools=query_result.expected_tools,
                execution_context=dict(execution_context),
            )
            conversation.turns.append(turn)

            print(f"\n✓ Turn {turn_idx + 1} complete")
            print(f"   Query: {query_result.query[:80]}...")
            print(f"   Steps: {len(trajectory)}")

        # Stage 3: Assemble final datapoint
        conversation.tools_used = sorted(tools_used)
        conversation.categories_used = sorted(categories_used)
        conversation.initial_api_state = filter_api_state(initial_api_state, list(tools_used)) if initial_api_state else None

        datapoint = MultiTurnDatapoint(
            conversation=conversation,
            generation_metadata={
                "num_turns": self.num_turns,
                "actions_per_turn": self.num_actions,
                "focus_category": focus_category,
                "overall_task": blueprint.overall_task,
                "blueprint_queries": [t.get("user_query", "") for t in blueprint.turns],
                "turn_expected_tools": [t.get("expected_tools", []) for t in blueprint.turns],
            },
            token_usage=self._get_token_stats(),
            initial_api_state=conversation.initial_api_state,
        )

        print("\n" + "=" * 70)
        print("✓ MULTI-TURN DATAPOINT GENERATION COMPLETE")
        print("=" * 70)
        print(f" Turns: {len(conversation.turns)}")
        print(f" Tools used: {conversation.tools_used}")
        print(f" Total tool calls: {sum(len(t.steps) for t in conversation.turns)}")

        return datapoint

    # ─────────────────────── Stage 0: Blueprint ───────────────────────

    def _stage0_generate_blueprint(
            self, focus_category: Optional[str] = None
    ) -> Optional[DialogBlueprint]:
        """Generate a highly specific dialog blueprint with concrete entities and full user queries."""
        tools_str = self._get_tools_with_descriptions_str(category=focus_category)

        prompt = f"""You are designing a multi-turn user-agent conversation for testing a tool-calling system.

The conversation has EXACTLY {self.num_turns} turns. Each turn works like this:
  1. The USER says something (a natural request mentioning concrete entities)
  2. The AGENT calls EXACTLY {self.num_actions} tools to fulfill that request
  3. The AGENT responds to the user with a summary of what was done

The conversation should feel like a REAL user chatting with a support agent across multiple steps.

=== AVAILABLE TOOLS ===
{tools_str}

=== REQUIREMENTS ===
1. Each turn's user_query must include SPECIFIC concrete entities: usernames, passwords, user IDs, ticket IDs, stock symbols, dates, prices, amounts, etc.
2. Each turn's user_query must require EXACTLY {self.num_actions} tools to fulfill
3. The conversation should flow naturally — each turn builds on the previous one's results
4. AUTH: For the FIRST turn that uses an auth-gated tool (place_order, post_tweet, send_message, close_ticket, book_flight, etc.), the user MUST include the login tool (trading_login, authenticate_twitter, message_login, ticket_login, authenticate_travel) in expected_tools. Later turns can skip the login tool because auth persists.
5. Use STORED credentials from the API state: trader_admin/TradeAdmin2024! for trading, tech_user/TechUser2024! for posting, support_agent/SupportAgent2024! for tickets, travel_client_001/s3cretK3y!/refresh_abc123 for travel, valid user IDs (USR005-USR014) for messaging.

=== EXAMPLES OF GOOD TURN QUERIES ===
- "Log me into the trading platform as trader_admin with password TradeAdmin2024! and then place a buy order for 100 shares of MSFT at market price."
- "Now check my transaction history and add NVDA to my watchlist."
- "Show me the info for AAPL stock and filter stocks in the Technology sector."
- "Log into the ticket system as support_agent with password SupportAgent2024! and create a ticket titled 'Network outage' with critical priority."
- "Resolve ticket #123456 with resolution 'Rebooted the server' and close it."
- "Authenticate me on Twitter as tech_user with password TechUser2024! and post a tweet saying 'Great day for AI!'"
- "Get the user ID for Sarah and send her a message saying 'Meeting at 2pm'."
- "Find the nearest airport to Miami and then get the flight cost from there to New York in economy class."

=== OUTPUT FORMAT ===
Respond ONLY with valid JSON. Each turn must have a SPECIFIC user_query with concrete entities:
{{
  "overall_task": "Specific description of the full conversation scenario with concrete entities",
  "turns": [
    {{
      "user_query": "The exact user request for this turn — MUST include concrete names, IDs, passwords, numbers",
      "expected_tools": ["tool1", "tool2"]
    }},
    ...
  ]
}}"""

        if focus_category:
            prompt += f"\n\nFocus primarily on the '{focus_category}' category for tool selection, but you can use tools from other categories if they fit the conversation."

        for attempt in range(3):
            try:
                response = self._safe_llm_generate([{"role": "user", "content": prompt}])
                response_text = response.strip()

                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]

                result = json.loads(response_text)
                turns = result.get("turns", [])
                if not turns or len(turns) != self.num_turns:
                    print(f"  ✗ Expected {self.num_turns} turns, got {len(turns)}")
                    continue

                all_tools_valid = all(
                    self.tool_manager.tool_exists(t)
                    for t_dict in turns
                    for t in t_dict.get("expected_tools", [])
                )
                if not all_tools_valid:
                    print("  ✗ Some expected_tools are invalid")
                    continue

                print(f" ✓ Blueprint generated: {result.get('overall_task', '')[:100]}")
                return DialogBlueprint(
                    overall_task=result.get("overall_task", ""),
                    num_turns=self.num_turns,
                    turns=turns,
                )
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"  ✗ Attempt {attempt + 1}: {e}")
                continue

        print("  ✗ Failed to generate valid blueprint after 3 attempts")
        return None

    # ─────────────────────── Turn query generation ───────────────────────

    def _generate_turn_query(
            self,
            blueprint: DialogBlueprint,
            conversation: MultiTurnConversation,
            turn_index: int,
    ) -> Optional[QueryGenerationResult]:
        """Use the blueprint's pre-written user query for this turn.

        The blueprint already includes a specific user_query for each turn
        with concrete entities (names, IDs, credentials). This avoids the
        inconsistency and extra LLM cost of per-turn query generation.
        """
        turn_spec = blueprint.turns[turn_index] if turn_index < len(blueprint.turns) else {}
        user_query = turn_spec.get("user_query", "")
        expected_tools = turn_spec.get("expected_tools", [])

        if not user_query or len(expected_tools) != self.num_actions:
            print(f"  ✗ Turn {turn_index + 1}: Blueprint has invalid query ({len(expected_tools)} tools, need {self.num_actions})")
            return None

        invalid = [t for t in expected_tools if not self.tool_manager.tool_exists(t)]
        if invalid:
            print(f"  ✗ Turn {turn_index + 1}: Invalid tools in blueprint: {invalid}")
            return None

        print(f"  ✓ Using blueprint query for turn {turn_index + 1}")
        print(f"   Query: {user_query[:80]}...")
        print(f"   Tools: {expected_tools}")
        return QueryGenerationResult(query=user_query, intent="", expected_tools=expected_tools)

    # ─────────────────────── Helpers ───────────────────────

    def _format_conversation_history(self, conversation: MultiTurnConversation) -> str:
        """Format completed turns as readable history for the LLM."""
        if not conversation.turns:
            return ""

        lines = []
        for turn in conversation.turns:
            lines.append(f"--- Turn {turn.turn_number} ---")
            lines.append(f"User: {turn.user_query}")
            for step in turn.steps:
                for tc in step.tool_calls:
                    output_preview = str(tc.output)[:100] if tc.output else ""
                    lines.append(f"  → {tc.tool_name}({json.dumps(tc.arguments, default=str)[:200]}) -> {output_preview}")
            lines.append(f"Assistant: {turn.assistant_response}")

        return "\n".join(lines)

    @staticmethod
    def _assign_tools_to_turns(blueprint: DialogBlueprint, all_tool_names: List[str]) -> Dict[int, List[str]]:
        """Distribute flat tool list across turns based on blueprint."""
        tools_by_turn: Dict[int, List[str]] = {}
        idx = 0
        for t_idx, t_spec in enumerate(blueprint.turns):
            turn_tools = t_spec.get("expected_tools", [])
            tools_by_turn[t_idx] = turn_tools
        return tools_by_turn