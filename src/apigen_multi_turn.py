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
from domain_hints import get_domain_hints


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
        validate_outputs: bool = True,
    ):
        super().__init__(llm_client, tool_manager, validate_outputs)
        self.num_turns = num_turns

    def continue_from_checkpoint(
            self,
            checkpoint: dict,
            focus_category: Optional[str] = None,
            query_retries: int = 3,
            tool_retries: int = 3,
    ) -> Optional[MultiTurnDatapoint]:
        """Continue generating a multi-turn datapoint from a checkpoint.

        Args:
            checkpoint: Dict containing:
                - 'blueprint': The dialog blueprint
                - 'partial_conversation': Partially completed MultiTurnConversation
                - 'execution_context': Dict of execution context
                - 'completed_turns': Number of turns completed
                - 'initial_api_state': API state before this datapoint started
                - 'token_usage': Accumulated token usage so far
            focus_category: Category for tool filtering
            query_retries: Max retries for query generation
            tool_retries: Max retries for tool generation

        Returns:
            MultiTurnDatapoint if successful, None otherwise
        """
        blueprint_data = checkpoint.get('blueprint')
        if not blueprint_data:
            print("✗ Checkpoint missing blueprint")
            return None

        from apigen_multi_turn import DialogBlueprint, MultiTurnConversation

        try:
            blueprint = DialogBlueprint(
                overall_task=blueprint_data.get('overall_task', ''),
                num_turns=blueprint_data.get('num_turns', self.num_turns),
                turns=blueprint_data.get('turns', [])
            )
        except Exception as e:
            print(f"✗ Failed to reconstruct blueprint: {e}")
            return None

        partial_conv_data = checkpoint.get('partial_conversation', {})
        try:
            conversation = MultiTurnConversation(
                overall_task=partial_conv_data.get('overall_task', blueprint.overall_task),
                turns=partial_conv_data.get('turns', []),
                tools_used=partial_conv_data.get('tools_used', []),
                categories_used=partial_conv_data.get('categories_used', []),
            )
        except Exception as e:
            print(f"✗ Failed to reconstruct conversation: {e}")
            return None

        completed_turns = checkpoint.get('completed_turns', len(conversation.turns))
        execution_context = checkpoint.get('execution_context', {})
        initial_api_state = checkpoint.get('initial_api_state')

        print(f"\n{'=' * 70}")
        print(f"RESUMING FROM CHECKPOINT")
        print(f"=" * 70)
        print(f" Overall task: {blueprint.overall_task}")
        print(f" Completed turns: {completed_turns}/{blueprint.num_turns}")
        print(f" Remaining turns: {blueprint.num_turns - completed_turns}")

        # Initialize API state and replay completed turns
        if self._python_tools_available and initial_api_state:
            print("\n Restoring API state from checkpoint...")
            self.tool_manager.restore_api_state(initial_api_state)

            # Replay completed turns to restore side-effects from their tool calls
            if completed_turns > 0:
                print(f" Replaying {completed_turns} turns to restore state...")
                for turn_idx in range(completed_turns):
                    if turn_idx >= len(conversation.turns):
                        break
                    turn = conversation.turns[turn_idx]
                    for step in turn.steps:
                        for tc in step.tool_calls:
                            if self.tool_manager.has_python_implementation(tc.tool_name):
                                self.tool_manager.invoke_python_tool(tc.tool_name, tc.arguments)
                print(f" Replayed {completed_turns} turns to restore state")

        self._update_token_usage()

        tools_used = set(conversation.tools_used)
        categories_used = set(conversation.categories_used)

        # Process remaining turns
        for turn_idx in range(completed_turns, blueprint.num_turns):
            print(f"\n{'=' * 70}")
            print(f"TURN {turn_idx + 1}/{blueprint.num_turns} (resumed)")
            print("=" * 70)

            turn_spec = blueprint.turns[turn_idx] if turn_idx < len(blueprint.turns) else {}

            query_result = self._generate_turn_query(
                blueprint=blueprint,
                conversation=conversation,
                turn_index=turn_idx,
            )
            if query_result is None:
                print(f"✗ Turn {turn_idx + 1} failed: Could not generate query")
                return None
            self._update_token_usage()

            trajectory = None
            for attempt in range(tool_retries):
                raw_trajectory, ec = self._stage2_generate_tools(
                    query_result, tool_retries - attempt, initial_execution_context=execution_context
                )
                if raw_trajectory is None:
                    print(f"✗ Turn {turn_idx + 1}: Could not generate tool calls")
                    return None

                errors = self._validate_tool_arguments(raw_trajectory)
                cross_errors = self._validate_cross_turn_consistency(raw_trajectory, execution_context)
                all_errors = errors + cross_errors
                if not all_errors:
                    trajectory = raw_trajectory
                    break

                print(f"  ⚠ Turn {turn_idx + 1} validation failed (attempt {attempt + 1}/{tool_retries}):")
                for err in errors:
                    print(f"    arg: {err}")
                for err in cross_errors:
                    print(f"    cross: {err}")
                if attempt < tool_retries - 1:
                    print(f"  Retrying turn {turn_idx + 1}...")

            if trajectory is None:
                print(f"✗ Turn {turn_idx + 1}: Too many validation failures, rejecting datapoint")
                return None
            self._update_token_usage()

            # Merge turn context into persistent execution_context
            for k, v in ec.items():
                execution_context[k] = v

            turn_output_aggregate = {}
            for step in trajectory:
                for tc in step.tool_calls:
                    if tc.output and isinstance(tc.output, dict):
                        turn_output_aggregate[tc.tool_name] = tc.output
            if 'turn_outputs' not in execution_context:
                execution_context['turn_outputs'] = []
            execution_context['turn_outputs'].append(turn_output_aggregate)

            assistant_response = self._generate_final_response(
                query_result.query, trajectory, execution_context
            )
            self._update_token_usage()

            for step in trajectory:
                for tc in step.tool_calls:
                    tools_used.add(tc.tool_name)
                    cat = self.tool_manager.get_tool_category(tc.tool_name)
                    if cat:
                        categories_used.add(cat)

            from apigen_multi_turn import Turn
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

            print(f"\n✓ Turn {turn_idx + 1} complete (resumed)")
            print(f"   Query: {query_result.query[:80]}...")

        # Finalize
        conversation.tools_used = sorted(tools_used)
        conversation.categories_used = sorted(categories_used)
        conversation.initial_api_state = initial_api_state

        datapoint = MultiTurnDatapoint(
            conversation=conversation,
            generation_metadata={
                "num_turns": self.num_turns,
                "focus_category": focus_category,
                "overall_task": blueprint.overall_task,
                "resumed_from_turn": completed_turns,
                "blueprint_queries": [t.get("user_query", "") for t in blueprint.turns],
                "turn_expected_tools": [t.get("expected_tools", []) for t in blueprint.turns],
            },
            token_usage=self._get_token_stats(),
            initial_api_state=conversation.initial_api_state,
        )

        print("\n" + "=" * 70)
        print("✓ RESUMED MULTI-TURN DATAPOINT GENERATION COMPLETE")
        print("=" * 70)
        print(f" Turns: {len(conversation.turns)}")
        print(f" Tools used: {conversation.tools_used}")
        print(f" Total tool calls: {sum(len(t.steps) for t in conversation.turns)}")

        return datapoint

    # ─────────────────────── Public entry point ───────────────────────

    def generate_multi_turn_datapoint(
            self,
            focus_category: Optional[str] = None,
            query_retries: int = 3,
            tool_retries: int = 3,
            checkpoint_callback: Optional[callable] = None,
    ) -> Optional[MultiTurnDatapoint]:
        """Generate a multi-turn datapoint.

        Args:
            focus_category: Category for tool filtering
            query_retries: Max retries for query generation
            tool_retries: Max retries for tool generation
            checkpoint_callback: Optional callback(state_dict) called after each
                turn completes. The state_dict contains:
                - 'blueprint': The dialog blueprint
                - 'partial_conversation': Partially completed MultiTurnConversation
                - 'execution_context': Current execution context
                - 'completed_turns': Number of turns completed
                - 'initial_api_state': API state before this datapoint started
                - 'focus_category': The category being used
        """

        self._reset_token_tracking()
        self._capture_initial_usage()

        # Initialize API state for the whole conversation
        initial_api_state = None
        if self._python_tools_available:
            self.tool_manager.initialize_api_state(force_new=True)
            initial_api_state = self.tool_manager.get_api_state()
            print(f" Captured initial API state ({len(initial_api_state)} class keys)")

        # Stage 0: Generate dialog blueprint
        print("\n" + "=" * 70)
        print("STAGE 0: Generate Dialog Blueprint")
        print("=" * 70)
        blueprint = self._stage0_generate_blueprint(focus_category, initial_api_state)
        if blueprint is None:
            print("✗ Stage 0 failed: Could not generate dialog blueprint")
            return None
        self._update_token_usage()
        print(f" Overall task: {blueprint.overall_task}")
        for i, t in enumerate(blueprint.turns, 1):
            uq = t.get('user_query', '')
            print(f"   Turn {i}: {uq[:80]}...")

        conversation = MultiTurnConversation(overall_task=blueprint.overall_task)

        # Stage 0.5: Ensure user identity coherence
        if self._python_tools_available:
            print("\n" + "-" * 70)
            print("STAGE 0.5: Ensure User Identity Coherence")
            print("-" * 70)
            identity_adjusted = self._ensure_user_identity_coherence(blueprint.overall_task)
            if identity_adjusted:
                initial_api_state = self.tool_manager.get_api_state()
                print(f" ✓ Identity coherence adjusted")
            else:
                print(f" No identity adjustment needed")

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
            trajectory = None
            for attempt in range(tool_retries):
                raw_trajectory, ec = self._stage2_generate_tools(
                    query_result, tool_retries - attempt, initial_execution_context=execution_context
                )
                if raw_trajectory is None:
                    print(f"✗ Turn {turn_idx + 1}: Could not generate tool calls")
                    return None

                errors = self._validate_tool_arguments(raw_trajectory)
                cross_errors = self._validate_cross_turn_consistency(raw_trajectory, execution_context)
                all_errors = errors + cross_errors
                if not all_errors:
                    trajectory = raw_trajectory
                    break

                print(f"  ⚠ Turn {turn_idx + 1} validation failed (attempt {attempt + 1}/{tool_retries}):")
                for err in errors:
                    print(f"    arg: {err}")
                for err in cross_errors:
                    print(f"    cross: {err}")
                if attempt < tool_retries - 1:
                    print(f"  Retrying turn {turn_idx + 1}...")

            if trajectory is None:
                print(f"✗ Turn {turn_idx + 1}: Too many validation failures, rejecting datapoint")
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

            # Save checkpoint after each turn
            if checkpoint_callback:
                checkpoint_state = {
                    'blueprint': {
                        'overall_task': blueprint.overall_task,
                        'num_turns': blueprint.num_turns,
                        'turns': blueprint.turns,
                    },
                    'partial_conversation': conversation.model_dump(),
                    'execution_context': dict(execution_context),
                    'completed_turns': turn_idx + 1,
                    'initial_api_state': initial_api_state,
                    'focus_category': focus_category,
                }
                checkpoint_callback(checkpoint_state)
                print(f"   Checkpoint saved after turn {turn_idx + 1}")

        # Stage 3: Assemble final datapoint
        conversation.tools_used = sorted(tools_used)
        conversation.categories_used = sorted(categories_used)
        conversation.initial_api_state = filter_api_state(initial_api_state, list(tools_used)) if initial_api_state else None

        datapoint = MultiTurnDatapoint(
            conversation=conversation,
            generation_metadata={
                "num_turns": self.num_turns,
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

    def _get_tool_output_fields(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """Extract output field names by calling each Python tool with valid minimal inputs.

        Returns dict mapping tool_api_name -> list of output field names.
        Uses read-only calls to avoid state mutation.
        """
        if not self._python_tools_available:
            return {}

        result: Dict[str, List[str]] = {}

        # Build api_name -> class_key mapping filtered by category
        api_names = []
        for api_name, class_key in self.tool_manager.api_name_to_class_key.items():
            if category:
                tool_cat = self.tool_manager.get_tool_category(api_name)
                if tool_cat != category:
                    continue
            api_names.append((api_name, class_key))

        for api_name, class_key in api_names:
            instance = self.tool_manager.python_tool_instances.get(class_key)
            if not instance:
                continue
            method = getattr(instance, api_name, None)
            if not method or not callable(method):
                continue

            try:
                import inspect
                sig = inspect.signature(method)
                bound = []
                for pname, param in sig.parameters.items():
                    if pname == 'self':
                        continue
                    if param.annotation in (int, float) and param.default is inspect.Parameter.empty:
                        bound.append(1)
                    elif param.annotation == str and param.default is inspect.Parameter.empty:
                        if 'city' in pname.lower() or 'location' in pname.lower():
                            bound.append('New York')
                        elif 'date' in pname.lower():
                            bound.append('2025-03-15')
                        elif 'token' in pname.lower():
                            bound.append('DUMMY_TOKEN')
                        elif 'card' in pname.lower() or 'number' in pname.lower() or 'id' in pname.lower():
                            bound.append('12345')
                        elif 'message' in pname.lower() or 'name' in pname.lower():
                            bound.append('Test')
                        elif 'currency' in pname.lower():
                            bound.append('USD')
                        elif 'type' in pname.lower():
                            bound.append('basic')
                        elif 'cost' in pname.lower() or 'balance' in pname.lower() or 'value' in pname.lower() or 'limit' in pname.lower():
                            bound.append(100.0)
                        else:
                            bound.append('x')
                    elif param.annotation == bool:
                        bound.append(True)

                out = method(*bound)
                if isinstance(out, dict):
                    result[api_name] = sorted(out.keys())
                else:
                    result[api_name] = []
            except Exception:
                result[api_name] = ['success', 'message', 'id', 'result', 'error']

        return result

    def _validate_posting_api_entities(
        self,
        turns: List[Dict[str, Any]],
        initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[str]:
        """Deterministically validate that entity references in PostingApi queries exist in state.

        This catches cases where the blueprint references usernames (like 'techguru')
        that don't exist in the API state, which the LLM judge might miss.

        Returns list of error messages (empty if all entities are valid).
        """
        import re

        issues = []
        if not initial_api_state:
            return issues

        # Find PostingAPI state
        posting_state = initial_api_state.get("posting_api") or initial_api_state.get("PostingAPI")
        if not posting_state:
            return issues

        # Collect valid usernames from state
        valid_usernames = set()

        # From 'users' dict keys
        if isinstance(posting_state.get("users"), dict):
            valid_usernames.update(posting_state["users"].keys())

        # From 'following_list'
        if isinstance(posting_state.get("following_list"), list):
            valid_usernames.update(posting_state["following_list"])

        # From tweets - usernames who have posted
        if isinstance(posting_state.get("tweets"), dict):
            for tweet in posting_state["tweets"].values():
                if isinstance(tweet, dict) and tweet.get("username"):
                    valid_usernames.add(tweet["username"])

        # The logged-in user's own username
        if posting_state.get("username"):
            valid_usernames.add(posting_state["username"])

        # Also add from comments
        if isinstance(posting_state.get("comments"), dict):
            for comment_list in posting_state["comments"].values():
                if isinstance(comment_list, list):
                    for comment in comment_list:
                        if isinstance(comment, dict) and comment.get("username"):
                            valid_usernames.add(comment["username"])

        # From retweets
        if isinstance(posting_state.get("retweets"), list):
            for retweet in posting_state["retweets"]:
                if isinstance(retweet, dict) and retweet.get("username"):
                    valid_usernames.add(retweet["username"])

        # Username extraction patterns from query text
        # More specific patterns to avoid false positives like "user stats" or "user and"
        username_patterns = [
            r'from\s+user\s+(\w+)',    # "from user techguru"
            r'follow\s+(\w+)',         # "follow techguru"
            r'following\s+(\w+)',      # "following techguru"
            r'tweets?\s+from\s+(\w+)', # "tweets from techguru"
            r'by\s+user\s+(\w+)',      # "by user techguru"
            r'username\s+(\w+)',       # "username techguru"
            r'@(\w+)',                 # "@techguru" mention
        ]

        # Check each turn's query for entity references
        for turn_idx, turn in enumerate(turns, 1):
            query = turn.get("user_query", "")
            expected_tools = turn.get("expected_tools", [])

            # Only validate for PostingApi tools
            posting_tools = {
                'get_user_tweets', 'get_user_stats', 'follow_user', 'unfollow_user',
                'authenticate_twitter', 'mention', 'comment', 'retweet',
                'search_tweets', 'get_tweet', 'get_tweet_comments'
            }
            if not any(t for t in expected_tools if t in posting_tools):
                continue

            # Extract potential usernames from query
            found_usernames = set()
            for pattern in username_patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                found_usernames.update(matches)

            # Check each found username against valid usernames
            for username in found_usernames:
                if username.lower() not in {u.lower() for u in valid_usernames}:
                    issues.append(
                        f"Turn {turn_idx}: query references username '{username}' but "
                        f"'{username}' does not exist in state. "
                        f"Valid users: {', '.join(sorted(valid_usernames))[:100]}"
                    )

        return issues

    def _validate_vehicle_control_queries(
        self,
        turns: List[Dict[str, Any]],
        initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[str]:
        """Validate that fuel-related queries are consistent with initial vehicle state.

        Checks queries like 'fill up the tank' or 'add fuel' against initial fuelLevel
        to ensure the scenario is coherent (e.g., not asking to fill when already full).

        Returns list of error messages (empty if all queries are valid).
        """
        import re

        issues = []
        if not initial_api_state:
            return issues

        # Find VehicleControlAPI state
        vehicle_state = initial_api_state.get("vehicle_control") or initial_api_state.get("VehicleControlAPI")
        if not vehicle_state:
            return issues

        initial_fuel = vehicle_state.get("fuelLevel")
        if initial_fuel is None:
            return issues

        # Fuel fill patterns that indicate user wants to add fuel
        fuel_fill_patterns = [
            r'fill.*tank',
            r'add.*fuel',
            r'top.*up',
            r'refuel',
            r'fuel.*fill',
        ]

        for turn_idx, turn in enumerate(turns, 1):
            query = turn.get("user_query", "")
            expected_tools = turn.get("expected_tools", [])

            # Only validate for vehicle control tools that add fuel
            fuel_tools = {'fillFuelTank', 'addFuel'}
            if not any(t for t in expected_tools if t in fuel_tools):
                continue

            # Check if query indicates intent to add fuel
            for pattern in fuel_fill_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    # Found a fuel fill request - check if tank is already full
                    if initial_fuel >= 50.0:
                        issues.append(
                            f"Turn {turn_idx}: query asks to 'fill/add fuel' but initial fuelLevel "
                            f"is {initial_fuel} (tank is at or above max capacity of 50.0). "
                            f"This scenario is incoherent - tank cannot be filled when full. "
                            f"Use a config with fuelLevel < 50 or change the query to match state."
                        )
                    break

        return issues

    def _verify_blueprint_capabilities(
        self,
        turns: List[Dict[str, Any]],
        focus_category: Optional[str] = None,
        initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[bool, List[str]]:
        """Verify that each turn's query intent matches tool capabilities.

        Uses LLM-as-judge to check if the user query can actually be fulfilled
        by the selected tool given its description and the current API state.

        Returns (is_valid, error_list).
        """
        if not turns:
            return False, ["No turns provided"]

        # Build tool capability context
        tool_caps = []
        for turn in turns:
            for tool_name in turn.get("expected_tools", []):
                if tool_name not in [t[0] for t in tool_caps]:
                    try:
                        schema = self.tool_manager.get_tool_schema(tool_name)
                        desc = schema.get('description', 'No description')[:300]
                        params = schema.get('parameters', {})
                        props = params.get('properties', {}) if isinstance(params, dict) else {}
                        param_info = ""
                        for pname, pinfo in props.items():
                            if isinstance(pinfo, dict):
                                param_type = pinfo.get('type', 'unknown')
                                enum_vals = pinfo.get('enum', [])
                                if enum_vals:
                                    param_info += f" (enum: {enum_vals})"
                                else:
                                    param_info += f" ({param_type})"
                        tool_caps.append((tool_name, f"{desc}{param_info}"))
                    except ValueError:
                        tool_caps.append((tool_name, "Tool description unavailable"))

        tool_cap_str = "\n".join([f"- {name}: {desc}" for name, desc in tool_caps])

        # Build state summary with actual entity values for verification
        state_summary = ""
        if initial_api_state:
            for class_key, state in initial_api_state.items():
                if isinstance(state, dict):
                    # Show full state with actual values, not just keys
                    # This allows the judge to verify entity references exist
                    state_json = json.dumps(state, indent=2, default=str)
                    state_summary += f"\n{class_key}:\n{state_json[:5000]}"

        prompt = f"""You are verifying that a dialog blueprint's user queries can be fulfilled by the selected tools.

Check each turn: does the user_query intent match what the selected tool can actually do?

=== TOOL CAPABILITIES ===
{tool_cap_str}

=== CURRENT API STATE ===
This shows what entities (files, IDs, etc.) exist. Queries should reference only these entities.
{state_summary if state_summary else "No specific state provided."}

=== BLUEPRINT TURNS ===
{json.dumps(turns, indent=2, default=str)}

=== VERIFICATION TASK ===
For each turn, verify:
1. Does the user_query ask for something the selected tool can actually do?
2. Does the query phrasing match tool capabilities? (e.g., "search all files" can't be done by a single-file grep)
3. CRITICAL: Check if query asks for something NOT in the tool's parameter enums. For example, if displayCarStatus only has options [fuel, battery, doors, climate, headlights, parkingBrake, brakePadle, engine], then asking for "cruise control speed" is INVALID.
4. Are entity names (files, IDs, etc.) consistent with the API state?
5. If multiple tools are listed, is that realistic for one turn?

IMPORTANT: Reject queries that ask for things outside the tool's enum values or capabilities, even if the tool description mentions a general category that might seem related.

Respond ONLY with valid JSON:
{{"is_valid": true/false, "issues": ["Turn N: issue description", ...]}}

If ALL turns are achievable with their selected tools, set is_valid to true with empty issues."""


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
            is_valid = result.get("is_valid", False)
            issues = result.get("issues", [])

            return is_valid, issues

        except Exception as e:
            # On error, be permissive and let execution handle it
            return True, [f"Capability check error (allowing): {str(e)[:100]}"]

    # ─────────────────────── Stage 0: Blueprint ───────────────────────

    def _stage0_generate_blueprint(
            self, focus_category: Optional[str] = None, initial_api_state: Optional[Dict[str, Any]] = None
    ) -> Optional[DialogBlueprint]:
        """Generate a highly specific dialog blueprint with concrete entities and full user queries."""
        tools_json = self.tool_manager.get_tools_json_schema()
        if focus_category:
            tools_json = [t for t in tools_json if t.get('category') == focus_category]

        # Blueprint generation needs complete schemas so it can reason about
        # required arguments, defaults, enum values, and output dependencies.
        tools_str = json.dumps(
            tools_json,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        # Build output fields dynamically from tool definitions for the prompt
        output_fields_str = ""
        for tool in tools_json:
            name = tool.get('name', tool.get('api_name', ''))
            schema = tool.get('output_schema', {})
            props = schema.get('properties', {}) if schema else {}
            if props:
                fields = ', '.join(props.keys())
                output_fields_str += f"- {name}: {fields}\n"
            else:
                output_fields_str += f"- {name}\n"

        # Build output_fields_map dynamically for placeholder validation
        output_fields_validation_map: Dict[str, List[str]] = {}
        for tool in tools_json:
            name = tool.get('name', tool.get('api_name', ''))
            if name:
                # Collect all known fields from tool definition's output schema if available,
                # otherwise use a generic fallback
                props = tool.get('output_schema', {}).get('properties', {}) if isinstance(tool, dict) else {}
                if props:
                    output_fields_validation_map[name] = list(props.keys())
                else:
                    output_fields_validation_map[name] = ['success', 'message', 'id', 'result', 'error']

        # Build set of class_keys that belong to the focus_category
        focus_class_keys = set()
        if focus_category:
            for api_name, class_key in self.tool_manager.api_name_to_class_key.items():
                tool_cat = self.tool_manager.get_tool_category(api_name)
                if tool_cat == focus_category:
                    focus_class_keys.add(class_key)

        # Inject actual credentials from initial_api_state into the prompt
        credential_context = ""
        initial_state_context = ""
        if initial_api_state:
            for class_key, state in initial_api_state.items():
                # Skip APIs not in the focus category to avoid credential_context pollution
                if focus_category and class_key not in focus_class_keys:
                    continue
                if not isinstance(state, dict):
                    continue
                    
                # Include full state structure for reference (filtered to focus category)
                state_summary = json.dumps(state, indent=2, default=str)
                initial_state_context += f"\n{class_key}:\n{state_summary}"
                
                # Credentials
                if 'client_id' in state and 'client_secret' in state and 'refresh_token' in state:
                    cid = state['client_id']
                    csec = state['client_secret']
                    rtok = state['refresh_token']
                    credential_context += f"\nCredential format: {cid}/{csec}/{rtok}"
                # Card IDs
                if 'credit_card_list' in state and isinstance(state['credit_card_list'], dict):
                    card_ids = list(state['credit_card_list'].keys())
                    if card_ids:
                        credential_context += f"\nAvailable card IDs: {', '.join(card_ids)}"
                # User list (messaging)
                if 'user_map' in state and isinstance(state['user_map'], dict):
                    user_ids = list(state['user_map'].keys())
                    if user_ids:
                        credential_context += f"\nAvailable user IDs: {', '.join(user_ids[:10])}"
                # Account balance
                if 'account_type' in state and 'balance' in state:
                    credential_context += f"\nAccount balance: {state.get('balance')}"
                # Username/password credentials
                if 'username' in state and 'password' in state:
                    credential_context += f"\nCredentials: {state['username']}/{state['password']}"

        prompt = f"""Design a {self.num_turns}-turn user-agent conversation. Each turn: USER request → AGENT calls 1-3 tools → AGENT responds.

=== AVAILABLE TOOLS ===
{tools_str}

=== OUTPUT SCHEMAS (use these exact field names in placeholders) ===
{output_fields_str}

=== REQUIREMENTS ===
 1. Each turn: specific entities (IDs, names, dates, prices) + 1-3 tools (vary naturally based on query complexity)
 2. Conversation flows naturally, each turn builds on previous
 3. Auth persists across turns - login only in FIRST turn needing auth (don't re-login)
 4. expected_tools: 1-3 tools per turn (allow natural variation)
 5. CRITICAL: expected_tools should ONLY contain tools that the user EXPLICITLY asks about or requests in their query. Do NOT add prerequisite tools (like pressing brake before starting engine) unless the user explicitly mentions them.
 6. CRITICAL: Query must ask for what the tools can provide. If a tool only accepts ONE parameter value at a time (like displayCarStatus with option=fuel OR battery, not both), the query should ask for only ONE thing per tool call. Do NOT ask for multiple items that require the same tool with different arguments.
 7. POLICY-CONTEXT CLOSURE: Every required argument for every expected tool must
    be available from the current/prior user queries, an earlier tool output, or
    a default declared in the tool schema.
 8. The Initial API State below is generator-only and will NOT be shown to the
    assistant that solves the task. Use it to choose valid existing values, but
    write every required state-derived value explicitly into the appropriate
    user_query unless an earlier tool call returns it.
 9. If a tool needs a value produced by another tool, include the producing tool
    first and use the exact output field. Never require the assistant to guess.
10. Cross-turn refs: use EXACT output field names like {{{{TURN1.tool_name.field_name}}}} where field_name matches the tool's output schema.
11. Match the exact semantic representation required by each tool. A
    human-readable label is not interchangeable with an opaque identifier, code,
    token, symbol, handle, coordinate, path, or credential.
12. General/model knowledge is not an argument source. Opaque values must be
    written in a user_query or returned by an earlier tool call.
13. Verify that each user_query plus the tool schemas and prior tool outputs is
    sufficient to determine all arguments for that turn's expected_tools.

=== EXAMPLES ===
- "Log into my account with username user123 and password SecretPass! then perform an action." (login_action, perform_action)
- "Authenticate with my credentials, then create a new item with title 'Network issue' and high priority." (login, create_item)
- "Submit a new post 'Great day for AI!' to my social feed." (authenticate_twitter, post_tweet)
- "Get the user ID for Sarah and send her a message." (get_user_id, send_message)

=== OUTPUT ===
{{"overall_task": "scenario", "turns": [{{"user_query": "request", "expected_tools": ["t1", "t2"]}}, ...]}}"""

        if focus_category:
            prompt += f"\n\nAll available tools below are from the '{focus_category}' category."

            domain_hints = get_domain_hints(focus_category)
            if domain_hints:
                prompt += f"\n\n{domain_hints}"

        if initial_state_context:
            prompt += (
                "\n\n=== GENERATOR-ONLY INITIAL API STATE ==="
                "\nThis state is not policy-visible. Any required value selected "
                "from it must be written into a user_query unless a prior tool "
                "returns it."
                f"{initial_state_context}"
            )
        
        if credential_context:
            prompt += (
                "\n\n=== GENERATOR-ONLY CREDENTIAL VALUES ==="
                "\nIf a credential is required and no tool returns it, include "
                "the exact credential naturally in the relevant user_query."
                f"{credential_context}"
            )

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
                    if not (1 <= len(expected) <= 3):
                        validation_errors.append(f"Turn {i+1} has {len(expected)} tools, need 1-3: {expected}")
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

                    # Validate duplicate tools that can't coexist (same tool called twice in one turn)
                    from collections import Counter
                    dup_tools = [t for t, c in Counter(expected).items() if c > 1]
                    if dup_tools:
                        validation_errors.append(
                            f"Turn {i+1} has duplicate tools that can't share arguments: {dup_tools}. "
                            f"A single LLM call can't generate distinct args for the same tool called twice. "
                            f"Use different tools instead."
                        )
                        all_tools_valid = False
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
                            known_fields = output_fields_validation_map.get(ref_tool, ['success', 'message', 'id', 'result'])
                            if ref_field not in known_fields:
                                validation_errors.append(f"Turn {i+1} placeholder {{TURN{p[0]}.{ref_tool}.{ref_field}}}: '{ref_field}' not in {ref_tool} output. Use: {known_fields}")
                                all_tools_valid = False
                                break
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

                # Verify tool capabilities match query intents
                print(f"  Verifying tool-query capability match...")
                cap_valid, cap_issues = self._verify_blueprint_capabilities(
                    turns, focus_category, initial_api_state
                )
                if not cap_valid:
                    cap_feedback = "\n".join(cap_issues) if cap_issues else "Tool capabilities don't match query intents"
                    accumulated_feedback = f"Capability mismatch:\n{cap_feedback}\n\nPlease regenerate with queries that match tool capabilities."
                    print(f"  ✗ {cap_feedback[:200]}...")
                    continue

                # Deterministic entity validation for PostingApi
                entity_issues = self._validate_posting_api_entities(turns, initial_api_state)
                if entity_issues:
                    entity_feedback = "\n".join(entity_issues)
                    accumulated_feedback = f"Entity reference errors:\n{entity_feedback}\n\nPlease regenerate with valid entity references from the API state."
                    print(f"  ✗ {accumulated_feedback[:200]}...")
                    continue

                # Deterministic entity validation for VehicleControl
                vehicle_issues = self._validate_vehicle_control_queries(turns, initial_api_state)
                if vehicle_issues:
                    vehicle_feedback = "\n".join(vehicle_issues)
                    accumulated_feedback = f"Vehicle state errors:\n{vehicle_feedback}\n\nPlease regenerate with coherent vehicle state."
                    print(f"  ✗ {accumulated_feedback[:200]}...")
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

        if not user_query or not (1 <= len(expected_tools) <= 3):
            print(f"  ✗ Turn {turn_index + 1}: Blueprint has invalid query ({len(expected_tools)} tools, need 1-3)")
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

    @staticmethod
    def _validate_tool_arguments(trajectory: List[TrajectoryStep]) -> List[str]:
        """Check tool call arguments and outputs for hallucination indicators.

        Returns list of error strings (empty = valid).
        Hallucinated empty required args cause datapoint rejection + retry.
        """
        errors = []
        for step in trajectory:
            for tc in step.tool_calls:
                args = tc.arguments or {}
                out = tc.output or {}
                name = tc.tool_name

                if name == 'book_flight':
                    for field in ['travel_date', 'travel_to', 'travel_from']:
                        if not args.get(field) or str(args.get(field, '')).strip() == '':
                            errors.append(
                                f"book_flight: hallucinated empty '{field}' in arguments"
                            )
                    bh = out.get('booking_history', {})
                    if not bh.get('travel_date') or not bh.get('travel_to'):
                        errors.append(
                            f"book_flight: output booking_history missing travel_date/travel_to"
                        )
                    if not out.get('booking_id'):
                        errors.append(f"book_flight: empty booking_id in output")

                elif name == 'purchase_insurance':
                    if not args.get('booking_id'):
                        errors.append("purchase_insurance: empty booking_id in arguments")
                    ins_id = out.get('insurance_id', '')
                    ins_status = out.get('insurance_status')
                    if (ins_id == '' or ins_id is None) and ins_status is False:
                        errors.append(
                            f"purchase_insurance: failed (ins_id='{ins_id}', status={ins_status}), "
                            f"likely operating on cancelled booking"
                        )

                elif name == 'retrieve_invoice':
                    inv = out.get('invoice', {})
                    if isinstance(inv, dict) and len(inv) == 0:
                        errors.append("retrieve_invoice: empty invoice dict in output")

                elif name == 'cancel_booking':
                    if not out.get('cancel_status') and out.get('cancel_status') is not None:
                        errors.append(f"cancel_booking: cancel_status=False")

                elif name == 'authenticate_travel':
                    if not out.get('success') and out.get('success') is not None:
                        errors.append(
                            f"authenticate_travel: failed success={out.get('success')} "
                            f"error={out.get('error', '')[:60]}"
                        )

                elif name == 'get_flight_cost':
                    if out.get('error'):
                        errors.append(f"get_flight_cost: error={out['error'][:80]}")

                elif name in ('ls', 'cat', 'cd', 'mkdir', 'mv', 'rm', 'rmdir', 'touch', 'cp', 'grep', 'find', 'wc', 'tail', 'echo', 'du', 'sort'):
                    if 'calls' in args:
                        errors.append(f"{name}: LLM generated 'calls' batch format - use single tool call with direct arguments")

        return errors

    def _validate_cross_turn_consistency(
        self,
        trajectory: List[TrajectoryStep],
        execution_context: Dict[str, Any],
    ) -> List[str]:
        """Validate that tool calls are consistent with prior turn outputs.

        Returns list of error strings (empty = valid).
        Cross-turn hallucination (e.g., book_flight with wrong route) causes rejection.
        """
        errors = []
        turn_outputs = execution_context.get('turn_outputs', [])

        current_tc_by_name = {}
        for step in trajectory:
            for tc in step.tool_calls:
                current_tc_by_name[tc.tool_name] = tc

        prior_tc_by_name = {}
        for turn_out in turn_outputs:
            for tool_name, output in turn_out.items():
                if tool_name not in prior_tc_by_name:
                    prior_tc_by_name[tool_name] = []
                prior_tc_by_name[tool_name].append(output)

        if 'book_flight' in current_tc_by_name:
            bf_args = current_tc_by_name['book_flight'].arguments or {}
            bf_out = current_tc_by_name['book_flight'].output or {}
            bf_from = bf_args.get('travel_from', '').upper()
            bf_to = bf_args.get('travel_to', '').upper()

            if 'get_flight_cost' in prior_tc_by_name:
                gfc_output = prior_tc_by_name['get_flight_cost'][-1]
                gfc_from = gfc_output.get('travel_from', '').upper()
                gfc_to = gfc_output.get('travel_to', '').upper()

                if gfc_from and gfc_to and (bf_from != gfc_from or bf_to != gfc_to):
                    errors.append(
                        f"book_flight: route mismatch. get_flight_cost used {gfc_from}→{gfc_to} "
                        f"but book_flight called with {bf_from}→{bf_to}"
                    )

            if 'get_nearest_airport_by_city' in prior_tc_by_name:
                airport_outputs = prior_tc_by_name['get_nearest_airport_by_city']
                prior_cities = set()
                prior_airports = set()
                for ao in airport_outputs:
                    nearest = ao.get('nearest_airport', '')
                    if nearest:
                        prior_airports.add(nearest.upper())

                if prior_airports and bf_from and bf_from.upper() not in prior_airports:
                    errors.append(
                        f"book_flight: travel_from='{bf_from}' not in prior airport lookups {prior_airports}"
                    )

        if 'purchase_insurance' in current_tc_by_name:
            pi_args = current_tc_by_name['purchase_insurance'].arguments or {}
            pi_out = current_tc_by_name['purchase_insurance'].output or {}
            pi_booking_id = pi_args.get('booking_id', '')

            if 'book_flight' in prior_tc_by_name:
                prior_booking_ids = set()
                for bo in prior_tc_by_name['book_flight']:
                    bid = bo.get('booking_id', '')
                    if bid:
                        prior_booking_ids.add(bid)

                if prior_booking_ids and pi_booking_id and pi_booking_id not in prior_booking_ids:
                    errors.append(
                        f"purchase_insurance: booking_id='{pi_booking_id}' not in prior bookings {prior_booking_ids}"
                    )

            if pi_out.get('insurance_status') is False:
                errors.append(
                    f"purchase_insurance: failed (booking_id='{pi_booking_id}', status=False)"
                )

        # General aggregate function validation: check aggregate chains use same inputs
        # This catches cases like "mean([...]) followed by min_value([DIFFERENT...])" where
        # the query says "same" values but arguments don't match
        aggregate_tools = {'mean', 'min_value', 'max_value', 'standard_deviation', 'sum_values'}
        aggregate_chains = {
            'mean': {'min_value', 'max_value', 'standard_deviation', 'sum_values'},
            'min_value': {'mean', 'max_value', 'standard_deviation', 'sum_values'},
            'max_value': {'mean', 'min_value', 'standard_deviation', 'sum_values'},
            'sum_values': {'mean', 'min_value', 'max_value', 'standard_deviation'},
            'standard_deviation': {'mean', 'min_value', 'max_value', 'sum_values'},
        }

        for current_tool_name, current_tc in current_tc_by_name.items():
            if current_tool_name not in aggregate_tools:
                continue
            current_args = current_tc.arguments or {}
            current_numbers = current_args.get('numbers', [])

            # Look for compatible prior aggregate tools
            compatible_prior = aggregate_chains.get(current_tool_name, set())
            for prior_tool_name, prior_outputs in prior_tc_by_name.items():
                if prior_tool_name not in compatible_prior:
                    continue
                if not prior_outputs:
                    continue

                prior_output = prior_outputs[-1]
                prior_numbers = prior_output.get('input_numbers', [])
                if not prior_numbers:
                    continue

                # Check if input numbers match (sets since order may differ)
                current_set = set(current_numbers) if current_numbers else set()
                prior_set = set(prior_numbers)

                if current_set and prior_set and current_set != prior_set:
                    errors.append(
                        f"{current_tool_name}: input numbers {current_numbers} do not match "
                        f"prior {prior_tool_name} inputs {prior_numbers} - query says 'same' values but "
                        f"arguments use different numbers"
                    )

        return errors

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
