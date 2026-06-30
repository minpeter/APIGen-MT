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

            # Stage 2: Generate and execute tool invocations (pass persistent execution_context)
            # Note: State adjustment removed - tool calls modify API state which persists,
            # and we pass current API state snapshot to the tool manager LLM
            trajectory, ec = self._stage2_generate_tools(query_result, tool_retries, initial_execution_context=execution_context)
            if trajectory is None:
                print(f"✗ Turn {turn_idx + 1} failed: Could not generate tool calls")
                return None
            self._update_token_usage()

            # Merge turn context into persistent execution_context
            for k, v in ec.items():
                execution_context[k] = v

            # Store turn outputs for TURN{N} placeholder resolution
            turn_output_aggregate = {}
            for step in trajectory:
                for tc in step.tool_calls:
                    if tc.output and isinstance(tc.output, dict):
                        # Store each tool's output by tool name
                        turn_output_aggregate[tc.tool_name] = tc.output
            if 'turn_outputs' not in execution_context:
                execution_context['turn_outputs'] = []
            execution_context['turn_outputs'].append(turn_output_aggregate)

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
        tools_str = self._get_tools_with_descriptions_str(category=focus_category, compact=True)

        output_fields_map = {
            'Storage': {
                # Fields from simulation output (which transforms Python returns)
                'mkdir': ['success', 'message'],
                'touch': ['success', 'file_name', 'message'],
                'cd': ['success', 'current_path', 'error'],
                'cat': ['content', 'file_name', 'error'],
                'echo': ['success', 'id', 'file_name', 'content', 'status'],
                'ls': ['id', 'path', 'files', 'show_hidden', 'total_count'],
                'rm': ['success', 'error'],
                'mv': ['success', 'source', 'destination', 'message', 'error'],
                'cp': ['success', 'source', 'destination', 'message', 'error'],
                'grep': ['matches', 'count', 'error'],
                'wc': ['lines', 'words', 'characters', 'file_name', 'mode'],
                'head': ['first_n_lines', 'file_name'],
                'tail': ['last_lines', 'file_name'],
                'find': ['files', 'error'],
                'du': ['total_size', 'unit', 'error'],
            },
            'Travel Booking': {
                'authenticate_travel': ['success', 'access_token', 'token_type', 'expires_in'],
                'book_flight': ['success', 'booking_id', 'flight_number', 'total_cost'],
                'get_flight_cost': ['success', 'price_usd', 'flight_number', 'currency'],
                'get_nearest_airport_by_city': ['success', 'airport_name', 'iata_code', 'distance'],
                'cancel_booking': ['success', 'cancel_status', 'booking_id'],
                'retrieve_invoice': ['success', 'invoice_id', 'amount', 'line_items'],
                'purchase_insurance': ['success', 'insurance_policy_id', 'amount_charged'],
            }
        }

        output_fields_str = ""
        for cat, fields in output_fields_map.items():
            if focus_category is None or cat == focus_category:
                output_fields_str += f"\n=== {cat} OUTPUT FIELDS ===\n"
                for tool, flds in fields.items():
                    output_fields_str += f"- {tool}: {', '.join(flds)}\n"

        prompt = f"""Design a {self.num_turns}-turn user-agent conversation. Each turn: USER request → AGENT calls EXACTLY {self.num_actions} tools → AGENT responds.

=== AVAILABLE TOOLS ===
{tools_str}

=== OUTPUT SCHEMAS (use these exact field names in placeholders) ===
{output_fields_str}

=== REQUIREMENTS ===
1. Each turn: specific entities (IDs, names, dates, prices) + EXACTLY {self.num_actions} tools
2. Conversation flows naturally, each turn builds on previous
3. Auth persists across turns - login only in FIRST turn needing auth (don't re-login)
4. expected_tools: EXACTLY {self.num_actions} tools per turn
5. Credentials: trader_admin/TradeAdmin2024! (trading), tech_user/TechUser2024! (posting), support_agent/SupportAgent2024! (tickets), travel_client_001/s3cretK3y!/refresh_abc123 (travel), USR005-USR014 (messaging)
6. Cross-turn refs: use EXACT output field names like {{{{TURN1.mkdir.dir_name}}}}, {{{{TURN3.touch.file_name}}}}, etc.

=== EXAMPLES ===
- "Log into trading as trader_admin/TradeAdmin2024! and buy 100 MSFT shares." (trading_login, place_order)
- "Create ticket 'Network outage' with critical priority." (ticket_login, create_ticket)
- "Post tweet 'Great day for AI!'" (authenticate_twitter, post_tweet)
- "Get the user ID for Sarah and send her a message." (get_user_id, send_message)

=== OUTPUT ===
{{"overall_task": "scenario", "turns": [{{"user_query": "request", "expected_tools": ["t1", "t2"]}}, ...]}}"""

        if focus_category:
            prompt += f"\n\nIMPORTANT: Use ONLY tools from '{focus_category}' category. Do NOT use tools from other categories."
        else:
            prompt += "\n\nYou may use tools from any category."

        accumulated_feedback = ""
        for attempt in range(3):
            try:
                if accumulated_feedback:
                    prompt_with_feedback = prompt + f"\n\n=== PREVIOUS ATTEMPT FEEDBACK ===\n{accumulated_feedback}\n=== END FEEDBACK ===\n"
                else:
                    prompt_with_feedback = prompt

                response = self._safe_llm_generate([{"role": "user", "content": prompt_with_feedback}])
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
                    accumulated_feedback = f"Expected {self.num_turns} turns, got {len(turns)}. Please generate exactly {self.num_turns} turns."
                    print(f"  ✗ {accumulated_feedback}")
                    continue

                validation_errors = []
                all_tools_valid = True
                for i, t in enumerate(turns):
                    expected = t.get("expected_tools", [])
                    if len(expected) != self.num_actions:
                        validation_errors.append(f"Turn {i+1} has {len(expected)} tools, need exactly {self.num_actions}: {expected}")
                        all_tools_valid = False
                        break

                    # Validate category if focus_category is specified
                    if focus_category:
                        for tool_name in expected:
                            tool_cat = self.tool_manager.get_tool_category(tool_name)
                            if tool_cat != focus_category:
                                validation_errors.append(f"Turn {i+1} tool '{tool_name}' is from category '{tool_cat}', not '{focus_category}'. Use only {focus_category} tools.")
                                all_tools_valid = False
                                break
                        if not all_tools_valid:
                            break

                    # Validate placeholder references in user_query
                    query = t.get("user_query", "")
                    import re
                    placeholders = re.findall(r'\{\{TURN(\d+)\.(\w+)\.(\w+)\}\}', query)
                    for p in placeholders:
                        ref_turn_idx = int(p[0]) - 1
                        ref_tool = p[1]
                        ref_field = p[2]
                        if ref_turn_idx >= i:
                            validation_errors.append(f"Turn {i+1} placeholder references future turn {p[0]}")
                            all_tools_valid = False
                            break
                        if ref_turn_idx < len(turns):
                            ref_tools = turns[ref_turn_idx].get("expected_tools", [])
                            if ref_tool not in ref_tools:
                                validation_errors.append(f"Turn {i+1} references {ref_tool} from turn {p[0]}, but that turn uses {ref_tools}")
                                all_tools_valid = False
                                break
                            # Validate that the placeholder field exists in tool output using known schema
                            output_fields_map = {
                                'mkdir': ['success', 'message', 'dir_name'],
                                'touch': ['success', 'message', 'file_name', 'created'],
                                'cd': ['success', 'message', 'current_path'],
                                'cat': ['success', 'content', 'file_name'],
                                'echo': ['success', 'message', 'file_name', 'id'],
                                'ls': ['success', 'files', 'path', 'id'],
                                'rm': ['success', 'message'],
                                'mv': ['success', 'message', 'source', 'destination'],
                                'cp': ['success', 'message', 'source', 'destination'],
                                'grep': ['success', 'lines', 'count'],
                                'wc': ['success', 'lines', 'words', 'chars', 'file_name'],
                            }
                            known_fields = output_fields_map.get(ref_tool, ['success', 'message', 'id', 'result'])
                            if ref_field not in known_fields:
                                validation_errors.append(f"Turn {i+1} placeholder {{TURN{p[0]}.{ref_tool}.{ref_field}}}: '{ref_field}' not in {ref_tool} output. Use: {known_fields}")
                                # Don't fail - just warn
                    if not all_tools_valid:
                        break

                # Validate cross-turn entity references
                cross_turn_entity_tools = {
                    'comment': ('tweet_id', 'post_tweet'),
                    'retweet': ('tweet_id', 'post_tweet'),
                    'mention': ('tweet_id', 'post_tweet'),
                    'edit_ticket': ('ticket_id', 'create_ticket'),
                    'resolve_ticket': ('ticket_id', 'create_ticket'),
                    'close_ticket': ('ticket_id', 'create_ticket'),
                    'delete_message': ('message_id', 'send_message'),
                    'purchase_insurance': ('booking_id', 'book_flight'),
                }
                for i, t in enumerate(turns):
                    if i == 0:
                        continue
                    expected = t.get("expected_tools", [])
                    query = t.get("user_query", "")
                    for tool_name in expected:
                        if tool_name in cross_turn_entity_tools:
                            id_field, create_tool = cross_turn_entity_tools[tool_name]
                            if i > 0 and i - 1 < len(turns):
                                prior_tools = turns[i - 1].get("expected_tools", [])
                                if create_tool in prior_tools:
                                    placeholder_pattern = f'{{{{TURN{i}.{create_tool}.{id_field}}}}}'
                                    if placeholder_pattern not in query:
                                        validation_errors.append(f"Turn {i+1} uses '{tool_name}' to operate on {create_tool} result but query lacks placeholder '{id_field}'")
                                        all_tools_valid = False
                                        break
                    if not all_tools_valid:
                        break

                if not all_tools_valid:
                    accumulated_feedback = "\n".join(validation_errors) if validation_errors else "Validation failed. Please check tool categories and placeholders."
                    print(f"  ✗ {accumulated_feedback}")
                    continue

                all_tools_valid = all(
                    self.tool_manager.tool_exists(t)
                    for t_dict in turns
                    for t in t_dict.get("expected_tools", [])
                )
                if not all_tools_valid:
                    accumulated_feedback = "Some expected_tools are invalid. Please use only valid tool names from the provided list."
                    print(f"  ✗ {accumulated_feedback}")
                    continue

                print(f" ✓ Blueprint generated: {result.get('overall_task', '')[:100]}")
                return DialogBlueprint(
                    overall_task=result.get("overall_task", ""),
                    num_turns=self.num_turns,
                    turns=turns,
                )
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                accumulated_feedback = f"JSON parse error: {e}. Please return valid JSON."
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

        Placeholders in the format {{TURN{N}.{tool_name}.{output_key}}} are
        resolved using the actual tool outputs from prior turns.
        """
        turn_spec = blueprint.turns[turn_index] if turn_index < len(blueprint.turns) else {}
        user_query = turn_spec.get("user_query", "")

        user_query = self._resolve_turn_placeholders(user_query, turn_index, conversation)

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

    def _resolve_turn_placeholders(
            self,
            query: str,
            turn_index: int,
            conversation: MultiTurnConversation,
    ) -> str:
        """Resolve {{TURN{N}.{tool_name}.{output_key}}} placeholders in a query.

        Looks up the actual output value from a prior turn's tool execution
        and substitutes it into the query.
        """
        import re
        pattern = re.compile(r"\{\{TURN(\d+)\.(\w+)\.(\w+)\}\}")

        def replacer(match):
            ref_turn = int(match.group(1))
            tool_name = match.group(2)
            output_key = match.group(3)

            if ref_turn > turn_index:
                return match.group(0)

            ref_turn_idx = ref_turn - 1
            if ref_turn_idx >= len(conversation.turns):
                return match.group(0)

            prior_turn = conversation.turns[ref_turn_idx]
            for step in prior_turn.steps:
                for tc in step.tool_calls:
                    if tc.tool_name == tool_name:
                        output = tc.output
                        if isinstance(output, dict):
                            if output_key in output:
                                return str(output[output_key])
                            for k, v in output.items():
                                if output_key.lower() in k.lower() or k.lower() in output_key.lower():
                                    return str(v)
                            if len(output) == 1:
                                return str(list(output.values())[0])
            return match.group(0)

        resolved = pattern.sub(replacer, query)
        if resolved != query:
            print(f"   Resolved placeholders: {query[:60]}... -> {resolved[:60]}...")
        return resolved

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