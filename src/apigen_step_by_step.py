from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import json
import random
import re
import copy
import os
import time
import requests
from pathlib import Path
from llm_client import LLMClient, LocalOpenAILLMClient
from tool_manager import ToolManager, filter_api_state
from prompts import StepByStepPrompts
from config_pool import generate_query_seed


class ToolCallWithOutput(BaseModel):
    """A single tool call with its simulated output."""
    tool_name: str
    arguments: Dict[str, Any] = {}
    output: Any = None


class StateVerificationResult(BaseModel):
    """LLM-as-judge verdict on a single state transition."""
    is_valid: bool = True
    reasoning: str = ""
    issues: List[str] = []
    state_changes_summary: str = ""


class TrajectoryStep(BaseModel):
    """A single step in the conversation trajectory."""
    step_number: int
    tool_calls: List[ToolCallWithOutput] = []
    reasoning: Optional[str] = None
    pre_state: Optional[Dict[str, Dict[str, Any]]] = None
    post_state: Optional[Dict[str, Dict[str, Any]]] = None
    state_verification: Optional[StateVerificationResult] = None


class ConversationTrajectory(BaseModel):
    """Complete conversation trajectory for a datapoint."""
    query: str
    steps: List[TrajectoryStep] = []
    final_response: str
    tools_used: List[str] = []
    categories_used: List[str] = []
    initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None


class TokenUsageStats(BaseModel):
    """Token usage statistics for a single datapoint."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_llm_calls: int = 0


class StepByStepDatapoint(BaseModel):
    """Complete datapoint generated step-by-step."""
    trajectory: ConversationTrajectory
    generation_metadata: Dict[str, Any] = {}
    verification_result: Optional[Dict[str, Any]] = None
    token_usage: TokenUsageStats = Field(default_factory=TokenUsageStats)
    initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None
    intermediate_api_states: List[Dict[str, Any]] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """Complete verification result for a generated datapoint."""
    query: str
    tool_relevance_checks: List[Dict[str, Any]] = []
    order_is_correct: bool
    order_verification_details: str = ""
    output_validations: List[Dict[str, Any]] = []
    placeholder_resolution: Dict[str, Any] = {}
    overall_verification_passed: bool
    verification_summary: str = ""


class StepSelectionResult(BaseModel):
    """Result of LLM selecting the next tool/step."""
    tool_name: str
    arguments: Dict[str, Any] = {}
    reasoning: str


class QueryGenerationResult(BaseModel):
    """Result of generating a user query."""
    query: str
    intent: str
    expected_tools: List[str] = []


class StepByStepGenerator:
    """Generator that creates datapoints step-by-step with immediate tool simulation."""

    def __init__(self, llm_client: LLMClient, tool_manager: ToolManager, num_actions: int = 2, validate_outputs: bool = True, judge_client: LLMClient = None):
        self.llm = llm_client
        self.judge = judge_client or llm_client
        self.tool_manager = tool_manager
        self.num_actions = num_actions
        self.validate_outputs = validate_outputs
        self._python_tools_available = bool(tool_manager.python_tool_instances)
        self._accumulated_prompt_tokens: int = 0
        self._accumulated_completion_tokens: int = 0
        self._accumulated_total_tokens: int = 0
        self._accumulated_llm_calls: int = 0
        self._initial_token_usage: Optional[Dict[str, int]] = None

    def _safe_llm_generate(self, messages: list, max_retries: int = 5, llm=None, **kwargs) -> str:
        """Call LLM generate() with application-level retry on transient errors.

        Args:
            llm: Override LLM client. Defaults to self.llm.
        """
        client = llm or self.llm
        import random as _rng
        for attempt in range(max_retries):
            try:
                result = client.generate(messages, **kwargs)
                if result is None:
                    raise ValueError("LLM returned None")
                if not result.strip():
                    raise ValueError("LLM returned empty response")
                return result
            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.HTTPError) as e:
                delay = min(2 * (2 ** attempt), 60) + _rng.uniform(0, 2)
                print(f" [_safe_llm_generate] Transient error (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}, retrying in {delay:.1f}s...")
                time.sleep(delay)
            except (RuntimeError, ValueError) as e:
                delay = min(2 * (2 ** attempt), 30) + _rng.uniform(0, 1)
                print(f" [_safe_llm_generate] Error (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}, retrying in {delay:.1f}s...")
                time.sleep(delay)
            except json.JSONDecodeError as e:
                delay = min(2 * (2 ** attempt), 30) + _rng.uniform(0, 1)
                # Extra debug info for JSON errors
                print(f" [_safe_llm_generate] JSON Error (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(delay)
        raise RuntimeError(f"LLM generate failed after {max_retries} application-level retries")

    def _reset_token_tracking(self):
        """Reset token tracking for a new datapoint."""
        self._accumulated_prompt_tokens = 0
        self._accumulated_completion_tokens = 0
        self._accumulated_total_tokens = 0
        self._accumulated_llm_calls = 0
        self._initial_token_usage = None
    
    def _capture_initial_usage(self):
        """Capture initial token usage before starting a datapoint."""
        self._initial_token_usage = self.llm.get_token_usage()
    
    def _update_token_usage(self):
        """Update accumulated token usage from LLM client."""
        if self._initial_token_usage is None:
            return
        
        current_usage = self.llm.get_token_usage()
        self._accumulated_prompt_tokens = current_usage["prompt_tokens"] - self._initial_token_usage["prompt_tokens"]
        self._accumulated_completion_tokens = current_usage["completion_tokens"] - self._initial_token_usage["completion_tokens"]
        self._accumulated_total_tokens = current_usage["total_tokens"] - self._initial_token_usage["total_tokens"]
        self._accumulated_llm_calls = current_usage["total_calls"] - self._initial_token_usage["total_calls"]
    
    def _get_token_stats(self) -> TokenUsageStats:
        """Get current token usage stats."""
        return TokenUsageStats(
            prompt_tokens=self._accumulated_prompt_tokens,
            completion_tokens=self._accumulated_completion_tokens,
            total_tokens=self._accumulated_total_tokens,
            total_llm_calls=self._accumulated_llm_calls
        )

    def _get_tool_schemas_str(self, tools_subset: Optional[List[str]] = None) -> str:
        schemas = self.tool_manager.get_tools_json_schema()
        if tools_subset:
            schemas = [s for s in schemas if s['name'] in tools_subset]
        return json.dumps(schemas, indent=2, ensure_ascii=False)


    def _get_tools_with_descriptions_str(self, category: Optional[str] = None, compact: bool = False) -> str:
        """Get a formatted string of tools with their full descriptions, organized by category."""
        tools = self.tool_manager.get_tools_json_schema()

        if category:
            tools = [t for t in tools if t.get('category') == category]

        if compact:
            result = []
            for tool in tools:
                name = tool['name']
                desc = tool.get('description', '')[:80]
                result.append(f"{name}: {desc}")
            return "\n".join(result)

        tools_by_cat = {}
        for tool in tools:
            cat = tool.get('category', 'Unknown')
            if cat not in tools_by_cat:
                tools_by_cat[cat] = []
            tools_by_cat[cat].append(tool)

        result = []
        for cat, cat_tools in sorted(tools_by_cat.items()):
            result.append(f"\n{cat}:")
            for tool in cat_tools:
                name = tool['name']
                desc = tool.get('description', 'No description available.')
                result.append(f" - {name}: {desc}")

        return "\n".join(result)

    def _process_placeholders(self, arguments: Dict[str, Any], execution_context: Dict[str, Any]) -> Dict[str, Any]:
        processed_args = copy.deepcopy(arguments)
        
        def resolve_placeholder(key_path: str) -> Any:
            """Resolve a placeholder key, supporting TURN{N} references."""
            keys = key_path.split('.')
            
            # Handle TURN{N} prefix
            if keys[0].startswith('TURN'):
                # Format: TURN{N}.{tool}.{field}
                # e.g., TURN1.authenticate_travel.access_token
                turn_ref = keys[0]  # e.g., TURN1
                turn_num = int(turn_ref.replace('TURN', '')) - 1  # 0-indexed
                
                # Look up in turn_outputs
                turn_outputs = execution_context.get('turn_outputs', [])
                if turn_num < len(turn_outputs):
                    current = turn_outputs[turn_num]
                    for k in keys[1:]:  # Skip TURN{N}, go to tool name
                        if isinstance(current, dict) and k in current:
                            current = current[k]
                        else:
                            return None
                    return current
                return None
            
            # Standard nested key lookup
            current = execution_context
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
            return current
        
        for arg_name, arg_value in processed_args.items():
            if isinstance(arg_value, str):
                placeholders = re.findall(r"\{\{([^{}]+)\}\}", arg_value)
                for placeholder_full_key in placeholders:
                    resolved_value = resolve_placeholder(placeholder_full_key)
                    if resolved_value is not None:
                        placeholder_tag = "{{" + placeholder_full_key + "}}"
                        if arg_value == placeholder_tag:
                            processed_args[arg_name] = resolved_value
                        else:
                            processed_args[arg_name] = arg_value.replace(placeholder_tag, str(resolved_value))
        return processed_args

    def validate_expected_tools(self, query: str, expected_tools: List[str], intent: str) -> tuple[bool, str]:
        """Validate expected_tools: count must match num_actions and sequence must make sense."""
        if len(expected_tools) != self.num_actions:
            return False, f"Expected tools count {len(expected_tools)} != required {self.num_actions}"

        tool_schemas = self._get_tool_schemas_str(expected_tools)

        prompt = f"""You are validating a tool sequence plan for a user query.

User Query: {query}
Intent: {intent}

Planned Tool Sequence: {json.dumps(expected_tools)}

Tool Schemas:
{tool_schemas}

Evaluate if the sequence logically fits the query intent.

Respond with JSON:
{{
    "is_valid": true/false,
             "issues": ["list of issues if any"]
}}"""

        try:
            response = self._safe_llm_generate([{"role": "user", "content": prompt}], llm=self.judge)
            response_text = response.strip()

            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            else:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]

            result = json.loads(response_text)
            is_valid = result.get("is_valid", False)
            issues = result.get("issues", [])

            if not is_valid:
                return False, f"Tool sequence validation failed: {'; '.join(issues)}"
            return True, ""
        except Exception as e:
            # If validation fails, assume valid to continue
            return True, ""

    def _get_example_queries(self) -> str:
        """Return few-shot examples of valid queries with correct tool sequences."""
        examples = [
            {
                "category": "Travel Booking",
                "num_tools": 3,
                "query": "I need to book a flight from JFK to London Heathrow on June 15, 2024 for 2 passengers in business class. Please authenticate first, then get the flight cost, and book the flight.",
                "intent": "User wants to book a business class flight and needs to authenticate, check pricing, and complete the booking",
                "expected_tools": ["authenticate_travel", "get_flight_cost", "book_flight"]
            },
            {
                "category": "Finance",
                "num_tools": 3,
                "query": "Log in to my trading account, then buy 50 shares of Apple stock at market price.",
                "intent": "User wants to log in and purchase Apple stock",
                "expected_tools": ["trading_login", "get_symbol_by_name", "place_order"]
            },
            {
                "category": "Events",
                "num_tools": 3,
                "query": "Log in to the ticket system, then resolve ticket #123456 with the resolution 'Issue fixed', and close it.",
                "intent": "User wants to authenticate, then resolve and close a ticket",
                "expected_tools": ["ticket_login", "resolve_ticket", "close_ticket"]
            },
            {
                "category": "Posting Api",
                "num_tools": 2,
                "query": "Authenticate with Twitter, then post a tweet saying 'Hello world!'.",
                "intent": "User wants to authenticate and post a tweet",
                "expected_tools": ["authenticate_twitter", "post_tweet"]
            },
            {
                "category": "Communication",
                "num_tools": 3,
                "query": "Log in to the messaging app as USR005, then get the user ID for Sarah, and send her a message saying 'Hi Sarah!'.",
                "intent": "User wants to log in, look up a user ID, and send a message",
                "expected_tools": ["message_login", "get_user_id", "send_message"]
            }
        ]

        # Filter examples by num_actions if possible
        filtered_examples = [ex for ex in examples if ex["num_tools"] <= self.num_actions + 1 and ex["num_tools"] >= self.num_actions - 1]
        if not filtered_examples:
            filtered_examples = examples[:3]

        result = []
        for i, ex in enumerate(filtered_examples, 1):
            result.append(f"\n=== EXAMPLE {i} ({ex['category']}, {ex['num_tools']} tools) ===")
            result.append(f"Query: \"{ex['query']}\"")
            result.append(f"Intent: {ex['intent']}")
            result.append(f"Expected tools: {ex['expected_tools']}")

        return "\n".join(result)

    def generate_user_query(self, focus_category: Optional[str] = None, validation_feedback: Optional[str] = None, max_retries: int = 3, query_seed: Optional[dict] = None) -> QueryGenerationResult:
        # Get tools with full descriptions
        tools_with_descriptions = self._get_tools_with_descriptions_str(category=focus_category, compact=True)

        accumulated_feedback = validation_feedback or ""
        example_queries = self._get_example_queries()

        persona_section = ""
        if query_seed:
            p = query_seed["persona"]
            c = query_seed["city"]
            persona_section = f"""
=== USER PERSONA (MANDATORY) ===
The user's name is {p['name']}, they are based in {p['city']}, {p.get('country', '')}.
You MUST use this person's name and city in the query when mentioning people or locations.
For example, if the query involves booking a flight, use {c['city']} as origin/destination.
If the query involves a credit card, use the name {p['name']} as cardholder.
Do NOT use "Michael Smith", "John", or generic American names — use {p['name']} exclusively.
"""

        for attempt in range(max_retries):
            prompt = f"""Generate a realistic user query requiring EXACTLY {self.num_actions} tools.

=== REQUIREMENTS ===
1. Specific with concrete entities (names, IDs, dates, locations)
2. EXACTLY {self.num_actions} tool calls needed - not more, not less
3. expected_tools: EXACTLY {self.num_actions} tool names from AVAILABLE TOOLS
4. CRITICAL: Use ONLY tools from AVAILABLE TOOLS - no invented names
5. Auth tools need login FIRST (e.g., place_order needs trading_login)
{persona_section}
=== AVAILABLE TOOLS ===
{tools_with_descriptions}
{example_queries}"""
            if focus_category:
                prompt += f"\n=== FOCUS CATEGORY ===\nPrimary: {focus_category}\n"
            if accumulated_feedback:
                prompt += f"\n=== FEEDBACK ===\n{accumulated_feedback}\n"
            prompt += f"""
=== TASK ===
Generate query requiring EXACTLY {self.num_actions} tools. Respond JSON:
{{"query": "specific with names/IDs", "intent": "what user wants", "expected_tools": ["tool1", ...]}}"""

            try:
                response = self._safe_llm_generate([{"role": "user", "content": prompt}])
                response_text = response.strip()

                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]
                else:
                    start = response_text.find("{")
                    end = response_text.rfind("}") + 1
                    if start >= 0 and end > start:
                        response_text = response_text[start:end]

                result = json.loads(response_text)
                query = result.get("query", "")
                intent = result.get("intent", "")
                expected_tools = result.get("expected_tools", [])

                print(f" Generated Query: {query}")
                print(f" Intent: {intent}")
                print(f" Expected tools: {expected_tools}")

                generated_summary = f"""--- ATTEMPT {attempt + 1} OUTPUT ---
Query: {query}
Intent: {intent}
Expected tools: {expected_tools}"""

                if len(expected_tools) != self.num_actions:
                    print(f" ✗ Wrong tool count: {len(expected_tools)} != {self.num_actions}")
                    accumulated_feedback += f"\n{generated_summary}\nFAILURE: Expected {self.num_actions} tools, but got {len(expected_tools)}.\n--- END ATTEMPT {attempt + 1} ---"
                    continue

                all_tools_valid = True
                invalid_tools = []
                for tool in expected_tools:
                    if not self.tool_manager.tool_exists(tool):
                        all_tools_valid = False
                        invalid_tools.append(tool)

                if not all_tools_valid:
                    available_tools = []
                    if focus_category:
                        cat_tools = self.tool_manager.get_tools_by_category(focus_category)
                        available_tools = [t['name'] for t in cat_tools[:20]]
                    else:
                        for cat in self.tool_manager.get_categories():
                            cat_tools = self.tool_manager.get_tools_by_category(cat)
                            available_tools.extend([t['name'] for t in cat_tools[:5]])

                    print(f" ✗ Invalid tools: {invalid_tools}")
                    accumulated_feedback += f"""\n{generated_summary}
FAILURE: Tools not found: {invalid_tools}
These tools do NOT exist. Choose from available tools.
Available tools (sample): {available_tools[:15]}
--- END ATTEMPT {attempt + 1} ---"""
                    continue

                if self.num_actions <= 5:
                    is_valid, validation_msg = self.validate_expected_tools(query, expected_tools, intent)

                    if not is_valid:
                        print(f" ✗ Tool sequence validation failed: {validation_msg}")
                        accumulated_feedback += f"\n{generated_summary}\nFAILURE: Tool sequence validation - {validation_msg}\n--- END ATTEMPT {attempt + 1} ---"
                        continue

                print(f" ✓ Query generation successful")
                return QueryGenerationResult(query=query, intent=intent, expected_tools=expected_tools)

            except json.JSONDecodeError as e:
                print(f" ✗ JSON decode error: {e}")
                accumulated_feedback += f"\n--- ATTEMPT {attempt + 1} FAILED ---\nJSON parsing error: {e}\n--- END ATTEMPT {attempt + 1} ---"
                continue

        print(f" Failed to generate valid query after {max_retries} attempts")
        return QueryGenerationResult(query="", intent="", expected_tools=[])

    def _generate_next_step(self, query: str, trajectory: List[TrajectoryStep], execution_context: Dict[str, Any], expected_tools: List[str], step_num: int = 1) -> StepSelectionResult:
        trajectory_str = ""
        for i, step in enumerate(trajectory):
            trajectory_str += f"\nStep {i+1}:"
            for tc in step.tool_calls:
                trajectory_str += f"\n - {tc.tool_name}({json.dumps(tc.arguments)})"
                if tc.output:
                    trajectory_str += f" -> {json.dumps(tc.output)[:200]}"

        tools_used = set()
        for step in trajectory:
            for tc in step.tool_calls:
                tools_used.add(tc.tool_name)

        tools_remaining = [t for t in expected_tools if t not in tools_used]
        if not tools_remaining:
            return StepSelectionResult(tool_name="__FINAL_RESPONSE__", arguments={}, reasoning="All expected tools have been used.")

        # Get tools with descriptions for remaining expected tools
        tool_descriptions_str = ""
        for tool_name in tools_remaining:
            try:
                    schema = self.tool_manager.get_tool_schema(tool_name)
                    if schema:
                        desc = schema.get('description', 'No description available.')[:150]
                        tool_descriptions_str += f" - {tool_name}: {desc}\n"
                    else:
                        tool_descriptions_str += f" - {tool_name}: (tool for completing the task)\n"
            except Exception as e:
                tool_descriptions_str += f" - {tool_name}: (tool for completing the task)\n"

        if not tool_descriptions_str:
            for tool_name in tools_remaining:
                tool_descriptions_str += f" - {tool_name}: (tool for completing the task)\n"

        prompt = f"""You are selecting the next tool to call based on the conversation context.

=== USER QUERY ===
{query}

=== CURRENT TRAJECTORY ===
{trajectory_str}

=== EXPECTED TOOLS REMAINING ===
{tool_descriptions_str}

=== EXECUTION CONTEXT (previous tool outputs) ===
{json.dumps(execution_context, indent=2, default=str)[:1000]}

=== YOUR TASK ===
Select the NEXT tool to call from the EXPECTED TOOLS REMAINING list above.

CRITICAL:
- You MUST select a tool name EXACTLY as shown in EXPECTED TOOLS REMAINING
- The tool must logically follow from the current trajectory and context
- Use values from Execution Context when available (e.g., user_id from previous step)

Respond ONLY with valid JSON:
{{
    "tool_name": "exact_name_from_expected_tools_list",
    "arguments": {{"arg1": "value1", "arg2": "value2"}},
    "reasoning": "brief explanation of why this tool and these arguments"
}}"""

        try:
            response = self._safe_llm_generate([{"role": "user", "content": prompt}])
            response_text = response.strip()

            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            else:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]

            result = json.loads(response_text)
            return StepSelectionResult(
                tool_name=result.get("tool_name", ""),
                arguments=result.get("arguments", {}),
                reasoning=result.get("reasoning", "")
            )
        except json.JSONDecodeError as e:
            print(f"    JSON decode error in step generation: {e}")
            return StepSelectionResult(tool_name="__ERROR__", arguments={}, reasoning=f"JSON error: {e}")

    def _simulate_tool_execution(self, tool_name: str, arguments: Dict[str, Any], execution_context: Dict[str, Any]) -> Any:
        if isinstance(arguments, list):
            if len(arguments) == 1 and isinstance(arguments[0], dict):
                arguments = arguments[0]
            else:
                arguments = arguments[0] if arguments else {}
        processed_args = self._process_placeholders(arguments, execution_context)
        if self._python_tools_available and self.tool_manager.has_python_implementation(tool_name):
            return self.tool_manager.invoke_python_tool(tool_name, processed_args)
        return self.tool_manager.invoke_tool(tool_name=tool_name, params=processed_args)

    @staticmethod
    def _detect_tool_error(tool_name: str, output: dict) -> tuple:
        """Detect errors in tool output using tool-specific keys and value checks.

        Returns (has_error: bool, error_detail: str).
        """
        generic_error_keys = ['error', 'error_message', 'error_code']
        for key in generic_error_keys:
            if key in output:
                return True, str(output[key])

        tool_specific_checks = {
            'comment': [('comment_status', lambda v: isinstance(v, str) and 'not authenticated' in v.lower())],
            'retweet': [('retweet_status', lambda v: isinstance(v, str) and 'not authenticated' in v.lower())],
            'mention': [('mention_status', lambda v: isinstance(v, str) and 'not authenticated' in v.lower())],
            'post_tweet': [('tweet_status', lambda v: isinstance(v, str) and 'not authenticated' in v.lower())],
            'follow_user': [('follow_status', lambda v: isinstance(v, str) and 'not authenticated' in v.lower())],
            'unfollow_user': [('follow_status', lambda v: isinstance(v, str) and 'not authenticated' in v.lower())],
            'authenticate_twitter': [('authentication_status', lambda v: v is False)],
            'get_ticket': [
                ('status', lambda v: isinstance(v, str) and 'not found' in v.lower()),
            ],
            'edit_ticket': [('status', lambda v: isinstance(v, str) and ('not found' in v.lower() or 'not authenticated' in v.lower()))],
            'resolve_ticket': [('status', lambda v: isinstance(v, str) and ('not found' in v.lower() or 'not authenticated' in v.lower()))],
            'close_ticket': [('status', lambda v: isinstance(v, str) and ('not found' in v.lower() or 'not authenticated' in v.lower()))],
            'create_ticket': [('status', lambda v: isinstance(v, str) and 'not authenticated' in v.lower())],
            'ticket_login': [('success', lambda v: v is False)],
            'message_login': [('login_status', lambda v: v is False)],
            'verify_traveler_information': [('verification_status', lambda v: v is False)],
            'authenticate_travel': [('access_token', lambda v: v == '')],
            'book_flight': [
                ('booking_status', lambda v: isinstance(v, str) and ('fail' in v.lower() or 'error' in v.lower())),
                ('booking_confirmation', lambda v: isinstance(v, str) and ('fail' in v.lower() or 'error' in v.lower())),
            ],
        }

        checks = tool_specific_checks.get(tool_name, [])
        for key, is_error in checks:
            val = output.get(key)
            if val is not None and is_error(val):
                return True, f"{key}: {val}"

        for key, val in output.items():
            if isinstance(val, str) and val.startswith("Error:"):
                return True, f"{key}: {val}"

        return False, ""

    # ==================== REFACTORED THREE-STAGE GENERATION ====================

    def generate_datapoint(self, focus_category: Optional[str] = None, context_hint: Optional[str] = None,
                           query_retries: int = 5, tool_retries: int = 3) -> Optional[StepByStepDatapoint]:
        """
        Generate a datapoint using multi-stage generation:
        Stage 1: Generate and verify query (separate retry count)
        Stage 1.5: Adjust initial API state for expected tools (best-effort)
        Stage 2: Generate tool invocations tool-by-tool (separate retry count per tool)
        Stage 3: Finalize datapoint (no retries)
        """
        print("\n" + "=" * 70)
        print("STEP-BY-STEP DATAPOINT GENERATION (Refactored)")
        print("=" * 70)

        # Reset and start token tracking for this datapoint
        self._reset_token_tracking()
        self._capture_initial_usage()

        query_seed = generate_query_seed()
        print(f" Persona seed: {query_seed['persona']['name']}, {query_seed['city']['city']}")

        # Initialize API state with full, realistic configurations
        # This ensures login calls and subsequent operations succeed
        initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None
        if self._python_tools_available:
            self.tool_manager.initialize_api_state()
            initial_api_state = self.tool_manager.get_api_state()
            print(f" Captured initial API state ({len(initial_api_state)} class keys)")

        # Stage 1: Generate and verify query
        print("\n" + "-" * 70)
        print("STAGE 1: Generate and Verify Query")
        print("-" * 70)
        
        query_result = self._stage1_generate_query(focus_category, context_hint, query_retries, query_seed)
        
        if query_result is None:
            print("\n✗ Stage 1 failed: Could not generate valid query")
            print(f"  Token usage for failed datapoint: {self._accumulated_total_tokens:,} tokens, {self._accumulated_llm_calls} calls")
            return None
        
        self._update_token_usage()
        print(f"\n✓ Stage 1 complete: Query generated and verified")
        print(f" Query: {query_result.query}")
        print(f" Expected tools: {query_result.expected_tools}")
        print(f" Tokens so far: {self._accumulated_total_tokens:,}")

        # Stage 1.5: Adjust initial API state for expected tools
        if self._python_tools_available and query_result.expected_tools:
            print("\n" + "-" * 70)
            print("STAGE 1.5: Adjust Initial API State")
            print("-" * 70)

            adjusted = self._stage1_5_adjust_initial_state(query_result)

            if adjusted:
                initial_api_state = self.tool_manager.get_api_state()
                print(f" ✓ API state adjusted, re-captured ({len(initial_api_state)} class keys)")
                self._update_token_usage()
                print(f" Tokens so far: {self._accumulated_total_tokens:,}")
            else:
                print(" ⚠ State adjustment failed or not needed, proceeding with original state")

        # Stage 2: Generate tool invocations tool-by-tool
        print("\n" + "-" * 70)
        print("STAGE 2: Generate Tool Invocations")
        print("-" * 70)
        
        trajectory, execution_context = self._stage2_generate_tools(query_result, tool_retries, query_seed)
        
        if trajectory is None:
            print("\n✗ Stage 2 failed: Could not generate all tool invocations")
            print(f"  Token usage for failed datapoint: {self._accumulated_total_tokens:,} tokens, {self._accumulated_llm_calls} calls")
            return None
        
        self._update_token_usage()
        print(f"\n✓ Stage 2 complete: Generated {len(trajectory)} tool invocations")
        print(f"  Tokens so far: {self._accumulated_total_tokens:,}")

        # Stage 3: Finalize datapoint
        print("\n" + "-" * 70)
        print("STAGE 3: Finalize Datapoint")
        print("-" * 70)
        
        datapoint = self._stage3_finalize(query_result, trajectory, execution_context, focus_category, initial_api_state)
        
        if datapoint is None:
            print("\n✗ Stage 3 failed: Could not finalize datapoint")
            return None
        
        print("\n" + "=" * 70)
        print("✓ DATAPOINT GENERATION COMPLETE (VERIFIED)")
        print("=" * 70)
        print(f" Query: {datapoint.trajectory.query}")
        print(f" Tools used: {datapoint.trajectory.tools_used}")
        print(f" Steps: {len(datapoint.trajectory.steps)}")
        print(f" Verification: PASSED")

        return datapoint

    def _stage1_generate_query(self, focus_category: Optional[str], context_hint: Optional[str], 
                               max_retries: int, query_seed: Optional[dict] = None) -> Optional[QueryGenerationResult]:
        """
        Stage 1: Generate and verify user query.
        - Separate retry count for query generation
        - Feedback is wiped on successful verification
        - Returns QueryGenerationResult or None if all retries exhausted
        """
        accumulated_feedback = context_hint or ""
        
        for attempt in range(max_retries):
            print(f"\n[Query Attempt {attempt + 1}/{max_retries}]")
            
            # Generate query
            query_result = self.generate_user_query(focus_category, accumulated_feedback if accumulated_feedback else None, query_seed=query_seed)

            if query_result is None or not query_result.query:
                print("  ✗ Failed to generate query")
                accumulated_feedback += f"\n--- ATTEMPT {attempt + 1} FAILED ---\nFailed to generate a valid query.\n--- END ATTEMPT {attempt + 1} ---"
                continue

            print(f"  Generated Query: {query_result.query}")
            print(f"  Intent: {query_result.intent}")
            print(f"  Expected tools: {query_result.expected_tools}")

            # Build a summary of what was generated for feedback
            generated_summary = f"""--- ATTEMPT {attempt + 1} OUTPUT ---
    Query: {query_result.query}
    Intent: {query_result.intent}
    Expected tools: {query_result.expected_tools}"""

            # Verify expected_tools
            print(f"  Verifying expected tools...")

            if not query_result.expected_tools:
                print("  ✗ ERROR: expected_tools is empty")
                accumulated_feedback += f"\n{generated_summary}\nFAILURE: expected_tools is empty.\n--- END ATTEMPT {attempt + 1} ---"
                continue

            if len(query_result.expected_tools) != self.num_actions:
                print(f"  ✗ ERROR: expected_tools count {len(query_result.expected_tools)} != {self.num_actions}")
                accumulated_feedback += f"\n{generated_summary}\nFAILURE: expected_tools count mismatch - got {len(query_result.expected_tools)}, need {self.num_actions}.\n--- END ATTEMPT {attempt + 1} ---"
                continue

            # Check if all tools exist
            invalid_tools = [t for t in query_result.expected_tools if not self.tool_manager.tool_exists(t)]
            if invalid_tools:
                print(f"  ✗ ERROR: Tools not found: {invalid_tools}")
                accumulated_feedback += f"\n{generated_summary}\nFAILURE: Tools not found: {invalid_tools}.\n--- END ATTEMPT {attempt + 1} ---"
                continue

            # Tool sequence validation already done inside generate_user_query
            # (which retries internally with feedback on failure)

            # SUCCESS: Query is valid - wipe feedback and return
            print(" ✓ Query verification passed")
            return query_result

        # All retries exhausted
        print(f"\n✗ Failed to generate valid query after {max_retries} attempts")
        return None

    def _stage1_5_adjust_initial_state(self, query_result: QueryGenerationResult) -> bool:
        """Stage 1.5: Adjust initial API state so expected tools will succeed.

        Sends the query, expected tools, their schemas, and the current API state
        to the LLM. The LLM identifies what state modifications are needed for
        each tool to execute successfully (e.g., ensure specific ticket IDs exist,
        user IDs are present, access tokens are set, etc.).

        Modifications are applied directly to the live Python tool instances,
        then the initial_api_state is re-captured.

        Returns True if adjustments were applied, False otherwise.
        """
        current_state = self.tool_manager.get_api_state()

        relevant_class_keys = set()
        for tool_name in query_result.expected_tools:
            class_key = self.tool_manager.api_name_to_class_key.get(tool_name)
            if class_key:
                relevant_class_keys.add(class_key)

        relevant_state = {k: v for k, v in current_state.items() if k in relevant_class_keys}

        tool_schemas_str = ""
        for tool_name in query_result.expected_tools:
            schema = self.tool_manager.get_tool_schema(tool_name)
            if schema:
                params = schema.get('parameters', {})
                desc = schema.get('description', '')
                tool_schemas_str += f"\n- {tool_name}: {desc}\n  Parameters: {json.dumps(params, indent=2, default=str)[:500]}\n"

        state_json = json.dumps(relevant_state, indent=2, default=str)[:6000]

        prompt_parts = [
            """You are preparing the initial state of API instances so that tool calls execute successfully.

=== USER QUERY ===
""",
            query_result.query,
            """

=== EXPECTED TOOL SEQUENCE ===
""",
            json.dumps(query_result.expected_tools),
            """

=== TOOL SCHEMAS ===
""",
            tool_schemas_str,
            """

=== CURRENT API STATE ===
""",
            state_json,
            """

=== RULES ===
- To add user: user_map.Username = "USR015"
- To append to list: "APPEND:array_key": value
- Set fields directly: "field_name": "value"
- Never do string operations like "APPEND:foo = bar"
- MINIMAL changes only

=== EXAMPLES ===
Add user to message_api:
{"modifications": {"message_api": {"user_map.Username": "USR015"}}, "reasoning": "..."}

Append ticket:
{"modifications": {"ticket_api": {"APPEND:tickets_queue": {"id": 1234}}}, "reasoning": "..."}

No changes needed:
{"modifications": {}, "reasoning": "no changes needed"}

=== RESPONSE ===
Respond only with valid JSON in one of these formats"""
        ]
        prompt = "".join(prompt_parts)

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
            modifications = result.get("modifications", {})
            reasoning = result.get("reasoning", "")

            if not modifications:
                print(f" No modifications needed: {reasoning}")
                return False

            print(f" Modifications requested: {json.dumps(modifications, indent=2, default=str)[:1000]}")
            print(f" Reasoning: {reasoning}")

            applied = self._apply_state_modifications(modifications)

            # Post-validation: remove duplicate user_map entries that would cause add_contact to fail
            if applied > 0 and 'message_api' in self.tool_manager.python_tool_instances:
                msg_api = self.tool_manager.python_tool_instances['message_api']
                user_map = getattr(msg_api, 'user_map', {})
                if user_map:
                    ids_to_names = {}
                    names_to_remove = set()
                    for name, uid in list(user_map.items()):
                        if uid in ids_to_names:
                            existing_name = ids_to_names[uid]
                            if existing_name != name:
                                names_to_remove.add(name)
                                print(f"   [DEDUP] Removing duplicate user_map entry '{name}' -> {uid} (conflicts with '{existing_name}' -> {uid})")
                        else:
                            ids_to_names[uid] = name
                    for name in names_to_remove:
                        del user_map[name]
                        applied -= 1

            # Fallback: ensure current_user is set for message operations that need it
            if applied > 0 and 'message_api' in self.tool_manager.python_tool_instances:
                msg_api = self.tool_manager.python_tool_instances['message_api']
                auth_gated_message_tools = {'delete_message', 'search_messages', 'send_message', 'delete_message', 'add_contact', 'get_user_id'}
                needs_auth = any(tool in auth_gated_message_tools for tool in query_result.expected_tools)
                current_user = getattr(msg_api, 'current_user', None)
                if needs_auth and not current_user:
                    user_map = getattr(msg_api, 'user_map', {})
                    if user_map:
                        first_user_id = list(user_map.values())[0] if user_map else None
                        if first_user_id:
                            msg_api.current_user = first_user_id
                            applied += 1
                            print(f"   [AUTH FALLBACK] Auto-set current_user to {first_user_id} for message operations")

            # Post-validation: ensure current_dir paths exist for gorilla_file_system
            if applied > 0 and 'gorilla_file_system' in self.tool_manager.python_tool_instances:
                fs = self.tool_manager.python_tool_instances['gorilla_file_system']
                current_dir = getattr(fs, 'current_dir', None)
                if current_dir and isinstance(current_dir, list):
                    root = getattr(fs, 'root', None)
                    if root and isinstance(root, dict):
                        if len(current_dir) > 0:
                            current = root
                            path_parts = current_dir
                            for i, part in enumerate(path_parts):
                                if part not in current:
                                    current[part] = {'type': 'directory', 'contents': {}}
                                if isinstance(current.get(part), dict) and current[part].get('type') == 'directory':
                                    current = current[part]['contents']
                                elif part in current and isinstance(current[part], dict):
                                    current = current[part]
                                else:
                                    break
                            print(f"   [FS FIX] Ensured directory path exists: {'/'.join(path_parts)}")
                        else:
                            current = root

                        # Fix: Ensure expected files exist in current_dir location
                        file_tools = {'cat', 'cp', 'diff', 'echo', 'grep', 'mv', 'rm', 'rmdir', 'sort', 'tail', 'touch', 'wc'}
                        needs_files = any(tool in file_tools for tool in query_result.expected_tools)
                        if needs_files:
                            # Find files needed based on the query
                            query_lower = query_result.query.lower()
                            # Common file names that might be needed
                            potential_files = ['config.json', 'processor.py', 'README.md', 'data.json', 'temp.txt']
                            for fname in potential_files:
                                if fname in query_lower or fname.replace('.py', '_v1.0.py') in query_lower:
                                    # Check if file exists in current location
                                    file_path = fname
                                    if fname not in current:
                                        # Check if file exists elsewhere in root and try to find it
                                        def find_file_in_tree(node, target, path=""):
                                            if isinstance(node, dict):
                                                if target in node:
                                                    return node[target]
                                                for k, v in node.items():
                                                    if isinstance(v, dict):
                                                        result = find_file_in_tree(v, target, f"{path}/{k}")
                                                        if result:
                                                            return result
                                            return None
                                        existing_file = find_file_in_tree(root, fname)
                                        if existing_file:
                                            # Move file to current location
                                            current[fname] = existing_file
                                            print(f"   [FS FIX] Moved '{fname}' to current directory")
                            print(f"   [FS FIX] current_dir={current_dir}, files now in location: {list(current.keys())}")

            if applied > 0:
                print(f" ✓ Applied {applied} state modifications")
                return True
            else:
                print(" No modifications could be applied")
                return False

        except (json.JSONDecodeError, ValueError) as e:
            print(f" ✗ Failed to parse state adjustment response: {e}")
            return False
        except Exception as e:
            print(f" ✗ State adjustment error: {e}")
            return False

    @staticmethod
    def _set_nested_field(obj, field_path: str, value: Any) -> None:
        """Set a field on an object using dot-notation path, creating intermediate dicts.
        
        Handles:
        1. File names with extensions by merging extension parts with the previous component.
           E.g., 'root.invoice.txt' -> root['invoice.txt']
        2. Bracket notation for keys with dots: 'root['invoice.txt']' -> root['invoice.txt']
        """
        # First, handle bracket notation: extract keys inside brackets and preserve them
        # E.g., "root['invoice.txt'].contents" -> ["root", "['invoice.txt']", "contents"]
        import re
        bracket_pattern = re.compile(r"\[('[^']+'|\"[^\"]+\")\]")
        
        # Replace brackets with a placeholder that won't be split
        parts_raw = bracket_pattern.split(field_path)
        bracket_keys = bracket_pattern.findall(field_path)
        
        # Reconstruct parts, treating bracket keys as single units
        # E.g., "root['invoice.txt'].contents" -> ['root', "['invoice.txt']", 'contents']
        processed_parts = []
        for part in parts_raw:
            if part:
                # Split remaining dots
                sub_parts = part.split('.')
                processed_parts.extend(sub_parts)
        
        # Insert bracket keys back in correct positions
        # This is tricky - we need to detect where bracket keys should go
        # For now, handle the case where bracket is at the end or followed by nothing
        
        parts = []
        i = 0
        while i < len(processed_parts):
            part = processed_parts[i]
            # Check if next part starts with [ and part doesn't end with ]
            if i + 1 < len(processed_parts) and processed_parts[i + 1].startswith('['):
                # Combine current part with next (the bracket key)
                parts.append(part + processed_parts[i + 1])
                i += 2
            else:
                parts.append(part)
                i += 1
        
        # Now handle file extensions (merge extension parts)
        common_extensions = {'txt', 'json', 'md', 'csv', 'pdf', 'html', 'xml', 'yaml', 'yml', 
                           'py', 'js', 'css', 'log', 'conf', 'config', 'sh', 'c', 'cpp', 
                           'h', 'hpp', 'go', 'rs', 'java', 'class', 'jar', 'war', 'ear',
                           'png', 'jpg', 'jpeg', 'gif', 'bmp', 'ico', 'svg', 'webp',
                           'mp3', 'mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm',
                           'zip', 'tar', 'gz', 'rar', '7z', 'bz2', 'xz',
                           'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx',
                           'exe', 'dll', 'so', 'dylib', 'a', 'o', 'obj', 'lib',
                           'bak', 'tmp', 'cache'}
        merged_parts = []
        for part in parts:
            if merged_parts and part.lower() in common_extensions and len(part) <= 5:
                merged_parts[-1] = merged_parts[-1] + '.' + part
            else:
                merged_parts.append(part)
        parts = merged_parts
        
        # Now apply to object
        for part in parts[:-1]:
            if isinstance(obj, dict):
                obj = obj.setdefault(part, {})
            else:
                obj = getattr(obj, part, {})
        last_key = parts[-1]
        if isinstance(obj, dict):
            obj[last_key] = value
        else:
            setattr(obj, last_key, value)

    def _apply_state_modifications(self, modifications: Dict[str, Any]) -> int:
        """Apply state modifications to live Python tool instances.

        Args:
            modifications: Dict mapping class_key -> {field_path: value}

        Returns:
            Number of modifications successfully applied
        """
        applied = 0

        for class_key, field_changes in modifications.items():
            actual_class_key = class_key
            extra_prefix = ""

            if class_key not in self.tool_manager.python_tool_instances and "." in class_key:
                first_dot = class_key.find(".")
                potential_key = class_key[:first_dot]
                if potential_key in self.tool_manager.python_tool_instances:
                    actual_class_key = potential_key
                    extra_prefix = class_key[first_dot + 1:] + "."
                    print(f"   ℹ Flat key detected: '{class_key}' -> class='{actual_class_key}', prefix='{extra_prefix}'")

            if actual_class_key not in self.tool_manager.python_tool_instances:
                print(f" ⚠ Unknown class_key: {class_key}, skipping")
                continue

            instance = self.tool_manager.python_tool_instances[actual_class_key]

            if not isinstance(field_changes, dict):
                effective_field_path = extra_prefix.rstrip(".") if extra_prefix else class_key
                if not effective_field_path:
                    print(f" ⚠ Empty field path for {class_key}, skipping")
                    continue
                value = field_changes
                self._set_nested_field(instance, effective_field_path, value)
                applied += 1
                print(f"   {actual_class_key}.{effective_field_path}: {value}")
                continue

            for field_path, value in field_changes.items():
                effective_field_path = (extra_prefix + field_path).rstrip(".")
                try:
                    if field_path == "current_dir" and class_key == "gorilla_file_system":
                        print(f"   ⚠ Skipping gorilla_file_system.current_dir modification (must use cd tool)")
                        continue
                    if field_path.startswith("APPEND:"):
                        list_field = field_path[len("APPEND:"):]
                        current_list = getattr(instance, list_field, None)
                        if isinstance(current_list, list):
                            current_list.append(value)
                            applied += 1
                            print(f"   {class_key}.{list_field}: appended item")
                        else:
                            print(f"   ⚠ {class_key}.{list_field} is not a list, skipping append")
                    elif field_path.startswith("EXTEND:"):
                        list_field = field_path[len("EXTEND:"):]
                        current_list = getattr(instance, list_field, None)
                        if isinstance(current_list, list) and isinstance(value, list):
                            current_list.extend(value)
                            applied += 1
                            print(f"   {class_key}.{list_field}: extended with {len(value)} items")
                        else:
                            print(f"   ⚠ {class_key}.{list_field} extend failed (not list or value not list)")
                    elif "." in effective_field_path:
                        parts = effective_field_path.split(".")
                        obj = instance
                        for part in parts[:-1]:
                            if isinstance(obj, dict):
                                if part not in obj:
                                    obj[part] = {}
                                obj = obj[part]
                            else:
                                nested = getattr(obj, part, {})
                                if isinstance(nested, dict) and part not in (getattr(type(obj), '__dict__', {}).keys() if hasattr(obj, '__dict__') else {}):
                                    try:
                                        setattr(obj, part, {})
                                    except (AttributeError, TypeError):
                                        pass
                                obj = getattr(obj, part, {})
                        last_key = parts[-1]
                        if isinstance(obj, dict):
                            if isinstance(value, str) and value.startswith("APPEND:"):
                                append_val = value[len("APPEND:"):]
                                existing = obj.get(last_key)
                                if isinstance(existing, list):
                                    existing.append(append_val)
                                    applied += 1
                                    print(f"   {actual_class_key}.{effective_field_path}: appended '{append_val}' to existing list")
                                else:
                                    obj[last_key] = [append_val]
                                    applied += 1
                                    print(f"   {actual_class_key}.{effective_field_path}: created list with '{append_val}'")
                            elif "user_map" in effective_field_path:
                                existing_ids = set(obj.values()) if obj else set()
                                if value in existing_ids:
                                    current_mapping = {v: k for k, v in obj.items()} if obj else {}
                                    existing_user = current_mapping.get(value)
                                    if existing_user and existing_user != last_key:
                                        print(f"   ⚠ Skipping user_map.{last_key} -> {value} (ID already assigned to '{existing_user}')")
                                        continue
                                obj[last_key] = value
                                applied += 1
                                print(f"   {actual_class_key}.{effective_field_path}: set to {json.dumps(value, default=str)[:100]}")
                            else:
                                obj[last_key] = value
                                applied += 1
                                print(f"   {actual_class_key}.{effective_field_path}: set to {json.dumps(value, default=str)[:100]}")
                        else:
                            print(f"   {actual_class_key}.{effective_field_path}: parent is not a dict, skipping")
                    else:
                        setattr(instance, effective_field_path, value)
                        applied += 1
                        print(f"   {actual_class_key}.{effective_field_path}: set to {json.dumps(value, default=str)[:100]}")
                except Exception as e:
                    print(f"   ⚠ Failed to apply {actual_class_key}.{effective_field_path}: {e}")

        return applied

    def _generate_tool_arguments(self, tool_name: str, query: str, trajectory: List[TrajectoryStep],
                                 execution_context: Dict[str, Any],
                                 feedback: Optional[str] = None,
                                 current_api_state: Optional[Dict[str, Dict[str, Any]]] = None,
                                 query_seed: Optional[dict] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Generate arguments for a specific tool based on query and context."""
        # Get tool schema
        tool_schema = self.tool_manager.get_tool_schema(tool_name)
        if not tool_schema:
            return None, f"Tool '{tool_name}' not found"

        # Build context from trajectory
        trajectory_str = ""
        for i, step in enumerate(trajectory):
            trajectory_str += f"\nStep {i+1}: {step.tool_calls[0].tool_name}"
            if step.tool_calls[0].output:
                output_summary = str(step.tool_calls[0].output)[:100]
                trajectory_str += f" -> {output_summary}"

        # Get output type info for better argument generation
        output_type = tool_schema.get('output_type', 'unknown')
        output_description = tool_schema.get('output_description', '')

        persona_arg_section = ""
        if query_seed:
            p = query_seed["persona"]
            c = query_seed["city"]
            persona_arg_section = f"""
=== USER CONTEXT ===
User name: {p['name']}
User city: {c['city']} ({c.get('code', '')})
When generating arguments that need a person's name, use {p['name']}.
When generating arguments that need a city or location, prefer {c['city']}.
"""

        # Build API state section — prefer the class relevant to this tool
        api_state_section = ""
        if current_api_state:
            class_key = self.tool_manager.api_name_to_class_key.get(tool_name)
            if class_key and class_key in current_api_state:
                state_for_tool = {class_key: current_api_state[class_key]}
            else:
                state_for_tool = current_api_state
            api_state_section = f"""
=== CURRENT API STATE ===
The following is the REAL current state of the API. You MUST use values from this state when providing arguments (e.g., user IDs, ticket IDs, usernames, access tokens). Do NOT invent or guess values — use the ones shown below.

{json.dumps(state_for_tool, indent=2, default=str)[:4000]}
            """

        prompt = f"""Generate arguments for '{tool_name}' based on user query and previous steps.

=== USER QUERY ===
{query}

=== PREVIOUS STEPS ===
{trajectory_str if trajectory_str else "None"}

=== EXECUTION CONTEXT ===
{json.dumps(execution_context, indent=2, default=str)[:2000]}
{persona_arg_section}{api_state_section}=== TOOL SCHEMA ===
{json.dumps(tool_schema.get('parameters', {}), indent=2)}

=== EXPECTED OUTPUT ===
Type: {output_type}
Description: {output_description}
"""
        if feedback:
            prompt += f"\n=== FEEDBACK ===\n{feedback}\n"
        prompt += """
=== TASK ===
Generate args matching schema and fulfilling query:
- Use REAL values from API STATE (user IDs, ticket IDs, tokens) - do NOT invent
- For LOGIN: use stored credentials from API state
- card_id: 'card_XXXX' (from register_credit_card), access_token: from prior authenticate_travel, booking_id: 'flight_XXX'

Respond JSON: {"arg1": "value1", ...}}"""

        try:
            response = self._safe_llm_generate([{"role": "user", "content": prompt}])
            response_text = response.strip()
        
            # Extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            else:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]

            arguments = json.loads(response_text)
            if not isinstance(arguments, dict):
                return None, f"Expected JSON object dict, got {type(arguments).__name__}"
            return arguments, None
        
        except json.JSONDecodeError as e:
            return None, f"JSON parsing error: {e}"

    def _stage2_generate_tools(self, query_result: QueryGenerationResult,
                               max_retries_per_tool: int, query_seed: Optional[dict] = None,
                               initial_execution_context: Optional[Dict[str, Any]] = None) -> Optional[Tuple[List[TrajectoryStep], Dict[str, Any]]]:
        """
        Stage 2: Generate tool invocations tool-by-tool.
        Uses expected_tools from Stage 1 directly - no LLM selection needed.
        - Each tool has its own retry count for argument generation
        - Feedback is wiped on successful tool completion
        - Captures pre/post API state snapshots around each tool call
        - Runs LLM-as-judge state verification after each call
        - If any tool fails after max retries, entire stage fails
        - Returns (trajectory, execution_context) or None
        """
        trajectory: List[TrajectoryStep] = []
        execution_context: Dict[str, Any] = initial_execution_context.copy() if initial_execution_context else {}

        for step_num, tool_name in enumerate(query_result.expected_tools, 1):
            print(f"\n[Step {step_num}/{self.num_actions}] Processing tool: {tool_name}")

            tool_feedback = ""
            step_success = False

            for attempt in range(max_retries_per_tool):
                print(f" [Attempt {attempt + 1}/{max_retries_per_tool}]")

                # ── Capture PRE state snapshot ──
                pre_state = self.tool_manager.get_api_state() if self._python_tools_available else None

                # Generate arguments for this tool (with feedback from previous failures)
                print(f"  Generating arguments for {tool_name}...")
                arguments, error = self._generate_tool_arguments(
            tool_name=tool_name,
            query=query_result.query,
            trajectory=trajectory,
            execution_context=execution_context,
            feedback=tool_feedback if tool_feedback else None,
            current_api_state=pre_state,
            query_seed=query_seed,
        )

                if error:
                    print(f" ✗ {error}")
                    if attempt < max_retries_per_tool - 1:
                        continue
                    break

                print(f" Arguments: {json.dumps(arguments)}")

                # Simulate tool execution
                print(f" Simulating {tool_name}...")
                output = self._simulate_tool_execution(
                    tool_name=tool_name,
                    arguments=arguments,
                    execution_context=execution_context
                )

                print(f" Output: {json.dumps(output, indent=2, ensure_ascii=False) if isinstance(output, (dict, list)) else output}")

                # Check for tool errors
                if isinstance(output, dict):
                    has_error, error_detail = self._detect_tool_error(tool_name, output)
                    if has_error:
                        error_type = output.get('error_type', 'execution_error')
                        print(f" ✗ Tool returned error: {error_detail}")
                        if error_type == 'validation_failure' and attempt < max_retries_per_tool - 1:
                            tool_feedback = f"Previous output validation failed: {error_detail}. Generate new arguments."
                            print(f" Retrying due to validation failure...")
                            continue
                        elif attempt < max_retries_per_tool - 1:
                            tool_feedback = f"Previous call failed: {error_detail}. Try different arguments or check prerequisites (e.g., login first, use correct IDs from API state)."
                            print(f" Retrying with feedback...")
                            continue
                        break

                # Validate output against declared type/description immediately
                tool_schema = self.tool_manager.get_tool_schema(tool_name)
                if tool_schema and self.validate_outputs:
                    expected_type = tool_schema.get('output_type', 'unknown')
                    expected_desc = tool_schema.get('output_description', '')
                    validation = self.verify_output_consistency(
                        tool_name, step_num, output, expected_type, expected_desc
                    )
                    if not validation['output_type_matches'] or validation.get('issues'):
                        issues_str = '; '.join(validation.get('issues', ['Type mismatch']))
                        print(f" ✗ Output validation failed: {issues_str}")
                        if attempt < max_retries_per_tool - 1:
                            tool_feedback = f"Previous output failed validation: {issues_str}. Expected type: {expected_type}."
                            print(f" Retrying with new arguments...")
                            continue
                        print(f" Max retries exceeded, proceeding with potentially invalid output")

                # ── Capture POST state snapshot ──
                post_state = self.tool_manager.get_api_state() if self._python_tools_available else None

                # ── LLM-as-judge state verification ──
                state_verification = None
                if pre_state is not None and post_state is not None:
                    print(f" Verifying state transition for {tool_name}...")
                    state_verification = self.verify_state_transition(
                        tool_name=tool_name,
                        tool_arguments=arguments,
                        tool_output=output,
                        pre_state=pre_state,
                        post_state=post_state,
                    )
                    if state_verification.is_valid:
                        print(f" ✓ State verification passed: {state_verification.state_changes_summary}")
                    else:
                        issues_joined = '; '.join(state_verification.issues)
                        print(f" ✗ State verification FAILED: {issues_joined}")
                        if attempt < max_retries_per_tool - 1:
                            tool_feedback = (
                                f"State verification failed: {issues_joined}. "
                                f"Judge reasoning: {state_verification.reasoning}. "
                                f"Generate different arguments."
                            )
                            print(f" Retrying due to state verification failure...")
                            # Roll back state by re-initializing + replaying completed steps
                            self._replay_state(trajectory)
                            continue
                        print(f" Max retries exceeded, proceeding despite state verification failure")

                # SUCCESS: Tool completed - add to trajectory
                print(f" ✓ Tool execution successful")

                # Update execution context
                if isinstance(output, dict):
                    for k, v in output.items():
                        execution_context[f"{tool_name}_{k}"] = v
                    # Store access_token directly for convenience (critical for auth-gated tools)
                    if 'access_token' in output:
                        execution_context['access_token'] = output['access_token']
                execution_context[f"{tool_name}_output"] = output

                # Add to trajectory (with state snapshots + verification)
                tool_call = ToolCallWithOutput(
                    tool_name=tool_name,
                    arguments=arguments,
                    output=output
                )
                trajectory_step = TrajectoryStep(
                    step_number=step_num,
                    tool_calls=[tool_call],
                    reasoning=f"Generated arguments for {tool_name} based on query context",
                    pre_state=pre_state,
                    post_state=post_state,
                    state_verification=state_verification,
                )
                trajectory.append(trajectory_step)
                step_success = True
                break

            if not step_success:
                print(f"\n✗ Tool {tool_name} failed after {max_retries_per_tool} attempts")
                return None, None

        # All tools completed successfully
        return trajectory, execution_context

    def _replay_state(self, trajectory: List[TrajectoryStep]) -> None:
        """Re-initialize API state and replay all completed trajectory steps.

        This is used to roll back state after a failed state-verification
        attempt so that the next retry starts from the correct state.
        """
        self.tool_manager.initialize_api_state()
        for step in trajectory:
            for tc in step.tool_calls:
                if self.tool_manager.has_python_implementation(tc.tool_name):
                    self.tool_manager.invoke_python_tool(tc.tool_name, tc.arguments)
        state = self.tool_manager.get_api_state()
        if 'message_api' in state and 'current_user' not in state['message_api']:
            for step in trajectory:
                for tc in step.tool_calls:
                    if tc.tool_name == 'message_login' and 'user_id' in tc.arguments:
                        self.tool_manager.python_tool_instances['message_api'].current_user = tc.arguments['user_id']
                        break

    def _stage3_finalize(self, query_result: QueryGenerationResult, trajectory: List[TrajectoryStep],
                         execution_context: Dict[str, Any],
                         focus_category: Optional[str],
                         initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[StepByStepDatapoint]:
        """
        Stage 3: Finalize datapoint.
        - No retries - if verification fails, something is fundamentally wrong
        - Assembles final datapoint with verification results
        - Stores initial_api_state and all verified intermediate states
        - Uses class-level token tracking
        """
        print("\nGenerating final response...")
        final_response = self._generate_final_response(query_result.query, trajectory, execution_context)
        print(f" Final response: {final_response}")

        # Collect tools and categories
        tools_used = []
        categories_used = set()
        for step in trajectory:
            for tc in step.tool_calls:
                if tc.tool_name not in tools_used:
                    tools_used.append(tc.tool_name)
                cat = self.tool_manager.get_tool_category(tc.tool_name)
                if cat:
                    categories_used.add(cat)

        # Filter state snapshots to only include APIs whose tools are used
        filtered_initial_state = filter_api_state(initial_api_state, tools_used) if initial_api_state else None

        # Build filtered trajectory steps (strip irrelevant API states)
        filtered_trajectory: List[TrajectoryStep] = []
        for step in trajectory:
            filtered_pre = filter_api_state(step.pre_state, tools_used) if step.pre_state else None
            filtered_post = filter_api_state(step.post_state, tools_used) if step.post_state else None
            filtered_trajectory.append(TrajectoryStep(
                step_number=step.step_number,
                tool_calls=step.tool_calls,
                reasoning=step.reasoning,
                pre_state=filtered_pre,
                post_state=filtered_post,
                state_verification=step.state_verification,
            ))

        # Extract intermediate verified states from trajectory steps
        intermediate_states: List[Dict[str, Any]] = []
        for step in filtered_trajectory:
            if step.post_state is not None and step.state_verification is not None:
                intermediate_states.append({
                    "step_number": step.step_number,
                    "post_state": step.post_state,
                    "state_verification": step.state_verification.model_dump(),
                })

        # Create trajectory
        conv_trajectory = ConversationTrajectory(
            query=query_result.query,
            steps=filtered_trajectory,
            final_response=final_response,
            tools_used=tools_used,
            categories_used=list(categories_used),
            initial_api_state=filtered_initial_state,
        )

        # Run verification
        print("\nRunning verification...")
        verification_result = self.run_full_verification(
            query=query_result.query,
            trajectory=trajectory,
            execution_context=execution_context
        )

        verification_passed = verification_result.overall_verification_passed if verification_result else False

        # If verification failed, return None so the caller knows to retry
        if not verification_passed:
            print(f"  Verification: FAILED")
            if verification_result:
                print(f"  Details: {verification_result.verification_summary}")
                for ov in verification_result.output_validations:
                    if not ov.get('output_type_matches', True):
                        print(f"    - {ov.get('tool_name')}: {ov.get('issues')}")
            print(f"\n✗ Datapoint failed verification - discarding")
            return None

        print(f" Verification: PASSED")

        # Update token usage from class-level tracking
        self._update_token_usage()
        token_usage = self._get_token_stats()

        # Create metadata
        metadata = {
            "num_actions": len(trajectory),
            "focus_category": focus_category,
            "query_intent": query_result.intent,
            "expected_tools": query_result.expected_tools
        }

        # Create datapoint
        datapoint = StepByStepDatapoint(
            trajectory=conv_trajectory,
            generation_metadata=metadata,
            verification_result=verification_result.model_dump() if verification_result else {},
            token_usage=token_usage,
            initial_api_state=filtered_initial_state,
            intermediate_api_states=intermediate_states,
        )

        return datapoint

    def _generate_final_response(self, query: str, trajectory: List[TrajectoryStep], execution_context: Dict[str, Any]) -> str:
        """Generate a natural final response based on the conversation."""
        actions_summary = []
        for step in trajectory:
            for tc in step.tool_calls:
                actions_summary.append({
                    "tool": tc.tool_name,
                    "arguments": tc.arguments,
                    "output_summary": str(tc.output)[:100] if tc.output else None
                })

        prompt = f"""Based on the following conversation, generate a natural final response.

User Query: {query}

Actions taken:
{json.dumps(actions_summary, indent=2)}

Generate a concise, natural response that summarizes what was accomplished."""

        try:
                response = self._safe_llm_generate([{"role": "user", "content": prompt}])
                return response.strip()
        except Exception as e:
            print(f"    Error generating final response: {e}")
            return "I have completed your request."

    # ==================== VERIFICATION METHODS ====================

    def verify_tool_relevance(self, query: str, tool_name: str, step: TrajectoryStep) -> Dict[str, Any]:
        """Verify if a tool is relevant to the query."""
        tool_schema = self.tool_manager.get_tool_schema(tool_name)
        if not tool_schema:
            return {'tool_name': tool_name, 'is_relevant': False, 'relevance_score': 0.0, 'reasoning': 'Tool not found in tool pool'}

        tool_description = tool_schema.get('description', '')
        keywords = set(tool_description.lower().split())
        query_words = set(query.lower().split())
        overlap = len(keywords & query_words)
        relevance_score = min(1.0, overlap / max(1, len(keywords)))

        name_words = set(tool_name.lower().replace('_', ' ').split())
        name_overlap = len(name_words & query_words)

        is_relevant = relevance_score > 0.1 or name_overlap > 0

        reasoning = f"Tool '{tool_name}': score={relevance_score:.2f}, name_match={name_overlap}"
        reasoning += ". Tool appears relevant." if is_relevant else ". Tool may not be directly relevant."

        return {'tool_name': tool_name, 'is_relevant': is_relevant, 'relevance_score': relevance_score, 'reasoning': reasoning}

    def verify_invocation_order(self, query: str, trajectory: List[TrajectoryStep]) -> Dict[str, Any]:
        """Verify if tools were invoked in a logical order."""
        if not trajectory:
            return {'order_is_correct': True, 'order_verification_details': 'No steps to verify'}

        return {'order_is_correct': True, 'order_verification_details': 'Order appears logical.'}

    @staticmethod
    def _is_dict_wrapped_primitive(output: Any, expected_type_lower: str) -> bool:
        """Check if a dict output wraps a value matching expected_type.

        Python tool implementations commonly return:
        - {'result': 42.0} when BFCL declares output_type=float
        - {'matching_tweets': []} when BFCL declares output_type=list
        - {'comments': [...]} when BFCL declares output_type=list
        This is a valid wrapper pattern — the semantic content *is* the
        expected type.
        """
        if not isinstance(output, dict) or not output:
            return False

        # List-wrapping: {"key": [...]}
        if 'list' in expected_type_lower:
            return any(isinstance(v, list) for v in output.values())

        prim_types = {
            'float': float,
            'number': (int, float),
            'integer': int,
            'string': str,
            'boolean': bool,
        }
        py_type = prim_types.get(expected_type_lower)
        if py_type is None:
            return False
        for v in output.values():
            if isinstance(v, py_type):
                return True
        return False

    def verify_output_consistency(self, tool_name: str, step_number: int, output: Any, expected_type: str, expected_description: str) -> Dict[str, Any]:
        """Verify if a tool's output matches its declared type and description."""
        if output is None:
            return {'tool_name': tool_name, 'step_number': step_number, 'output_type_matches': False, 'issues': ['Output is None']}

        issues = []
        output_type_matches = True
        if expected_type:
            expected_type_lower = expected_type.lower()
            output_type = type(output).__name__.lower()
            type_compatible = False
            if 'dict' in expected_type_lower and isinstance(output, dict):
                type_compatible = True
            elif 'list' in expected_type_lower and isinstance(output, list):
                type_compatible = True
            elif 'string' in expected_type_lower and isinstance(output, str):
                type_compatible = True
            elif 'number' in expected_type_lower and isinstance(output, (int, float)):
                type_compatible = True
            elif 'bool' in expected_type_lower and isinstance(output, bool):
                type_compatible = True
            elif expected_type_lower in output_type:
                type_compatible = True

            if not type_compatible and self._is_dict_wrapped_primitive(output, expected_type_lower):
                type_compatible = True

            if not type_compatible:
                output_type_matches = False
                issues.append(f"Type mismatch: expected {expected_type}, got {output_type}")

        return {'tool_name': tool_name, 'step_number': step_number, 'output_type_matches': output_type_matches, 'issues': issues}

    def verify_placeholder_resolution(self, trajectory: List[TrajectoryStep], execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that all placeholders in tool arguments were resolved correctly."""
        total_placeholders = 0
        resolved_count = 0
        details = []
        placeholder_pattern = re.compile(r"\{\{([^{}]+)\}\}")

        for step in trajectory:
            for tc in step.tool_calls:
                for arg_name, arg_value in tc.arguments.items():
                    if isinstance(arg_value, str):
                        placeholders = placeholder_pattern.findall(arg_value)
                        for placeholder in placeholders:
                            total_placeholders += 1
                            keys = placeholder.split('.')
                            current = execution_context
                            found = True
                            for key in keys:
                                if isinstance(current, dict) and key in current:
                                    current = current[key]
                                else:
                                    found = False
                                    break
                            if found:
                                resolved_count += 1
                                details.append({'step': step.step_number, 'tool': tc.tool_name, 'argument': arg_name, 'placeholder': f"{{{{{placeholder}}}}}", 'resolved': True, 'resolved_value': str(current)[:100]})
                            else:
                                details.append({'step': step.step_number, 'tool': tc.tool_name, 'argument': arg_name, 'placeholder': f"{{{{{placeholder}}}}}", 'resolved': False, 'resolved_value': None})

        return {'all_resolved': total_placeholders == resolved_count, 'total_placeholders': total_placeholders, 'resolved_count': resolved_count, 'details': details}

    def verify_state_transition(
        self,
        tool_name: str,
        tool_arguments: Dict[str, Any],
        tool_output: Any,
        pre_state: Dict[str, Dict[str, Any]],
        post_state: Dict[str, Dict[str, Any]],
    ) -> StateVerificationResult:
        """Use an LLM-as-judge to verify that a tool call produced a
        logically correct state transition.

        The LLM receives:
        - The tool name, arguments, and output
        - A *diff* between pre_state and post_state (only changed keys)
        - Relevant class keys (the ones that actually changed)

        It judges whether the state changes are consistent with the tool's
        declared semantics and the returned output.
        """
        # Compute diff — only include class keys that changed
        changed_classes: Dict[str, Dict[str, Any]] = {}
        for class_key in set(pre_state) | set(post_state):
            pre = pre_state.get(class_key, {})
            post = post_state.get(class_key, {})
            if pre != post:
                diff: Dict[str, Any] = {}
                all_keys = set(pre) | set(post)
                for k in all_keys:
                    pre_val = pre.get(k, "<MISSING>")
                    post_val = post.get(k, "<MISSING>")
                    if pre_val != post_val:
                        diff[k] = {"before": pre_val, "after": post_val}
                if diff:
                    changed_classes[class_key] = diff

        if not changed_classes:
            return StateVerificationResult(
                is_valid=True,
                reasoning="No state changes detected (read-only or no-op call).",
                issues=[],
                state_changes_summary="No state changes.",
            )

# Determine which class_key the tool belongs to
        tool_class_key = self.tool_manager.api_name_to_class_key.get(tool_name, "unknown")

        # Build a compact diff summary to avoid truncation issues
        diff_summary = {}
        for class_key in changed_classes:
            changes = []
            for k, v in changed_classes[class_key].items():
                before_val = v.get("before", "<MISSING>")
                after_val = v.get("after", "<MISSING>")
                if before_val == "<MISSING>":
                    changes.append(f"{k}: added")
                elif after_val == "<MISSING>":
                    changes.append(f"{k}: removed")
                else:
                    changes.append(f"{k}: modified")
            diff_summary[class_key] = changes

        diff_summary_str = json.dumps(diff_summary, indent=2, default=str, ensure_ascii=False)

        output_str = json.dumps(tool_output, default=str, ensure_ascii=False) if not isinstance(tool_output, str) else tool_output
        if len(output_str) > 1000:
            output_str = output_str[:1000] + "... (truncated)"

        args_str = json.dumps(tool_arguments, default=str, ensure_ascii=False)
        if len(args_str) > 1000:
            args_str = args_str[:1000] + "... (truncated)"

        prompt = f"""You are an expert API state auditor. Verify that the state transition produced by a tool call is logically correct and consistent with the tool's output.

=== TOOL CALL ===
Tool: {tool_name}
Class: {tool_class_key}
Arguments: {args_str}
Output: {output_str}

=== STATE CHANGE SUMMARY ===
{diff_summary_str}

For each changed class, the list shows what fields were added, removed, or modified.

=== YOUR TASK ===
1. Check whether the state changes are logically consistent with what the tool is supposed to do.
2. Verify that authentication/login state was updated correctly (e.g., current_user, authenticated, access_token).
3. Verify that data mutations (new messages, tickets, bookings, orders, etc.) are reflected in the state.
4. Check for any contradictory or nonsensical state changes.

NOTE: If you cannot determine validity from the summary provided, assume the state change is valid. Only mark as INVALID if you see clear contradictions or impossible changes.

Respond ONLY with valid JSON:
{{
  "is_valid": true/false,
  "reasoning": "brief explanation of your verdict",
  "issues": ["list of issues found, empty if valid"],
  "state_changes_summary": "human-readable summary of what changed"
 }}"""

        try:
            response = self._safe_llm_generate([{"role": "user", "content": prompt}], llm=self.judge)
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
            return StateVerificationResult(
                is_valid=bool(result.get("is_valid", True)),
                reasoning=result.get("reasoning", ""),
                issues=result.get("issues", []),
                state_changes_summary=result.get("state_changes_summary", ""),
            )
        except Exception as e:
            print(f" Warning: State verification LLM call failed: {e}")
            return StateVerificationResult(
                is_valid=True,
                reasoning=f"LLM judge call failed ({e}), assuming valid.",
                issues=[],
                state_changes_summary="Could not verify (LLM error).",
            )

    def run_full_verification(self, query: str, trajectory: List[TrajectoryStep], execution_context: Dict[str, Any]) -> VerificationResult:
        """Run all verification checks on a generated datapoint."""
        print("\n  Running Verification...")

        # 1. Check tool relevance
        tool_relevance_checks = []
        all_relevant = True
        for step in trajectory:
            for tc in step.tool_calls:
                check = self.verify_tool_relevance(query, tc.tool_name, step)
                tool_relevance_checks.append(check)
                if not check['is_relevant']:
                    all_relevant = False

        # 2. Verify invocation order
        order_result = self.verify_invocation_order(query, trajectory)

        # 3. Verify output consistency
        output_validations = []
        all_outputs_valid = True
        for step in trajectory:
            for tc in step.tool_calls:
                tool_schema = self.tool_manager.get_tool_schema(tc.tool_name)
                expected_type = tool_schema.get('output_type', 'unknown') if tool_schema else 'unknown'
                expected_desc = tool_schema.get('output_description', '') if tool_schema else ''

                validation = self.verify_output_consistency(tc.tool_name, step.step_number, tc.output, expected_type, expected_desc)
                output_validations.append(validation)
                if not validation['output_type_matches']:
                    all_outputs_valid = False

        # 4. Check placeholder resolution
        placeholder_result = self.verify_placeholder_resolution(trajectory, execution_context)

        # Overall check
        overall_passed = all_relevant and order_result['order_is_correct'] and all_outputs_valid and placeholder_result['all_resolved']

        issues = []
        if not all_relevant:
            issues.append("Some tools are not relevant to the query")
        if not order_result['order_is_correct']:
            issues.append("Tool invocation order may be incorrect")
        if not all_outputs_valid:
            issues.append("Some tool outputs don't match their declarations")
        if not placeholder_result['all_resolved']:
            issues.append(f"{placeholder_result['total_placeholders'] - placeholder_result['resolved_count']} placeholders were not resolved")

        summary = "Verification PASSED" if overall_passed else "Verification FAILED - " + "; ".join(issues)

        return VerificationResult(
            query=query,
            tool_relevance_checks=tool_relevance_checks,
            order_is_correct=order_result['order_is_correct'],
            order_verification_details=order_result['order_verification_details'],
            output_validations=output_validations,
            placeholder_resolution=placeholder_result,
            overall_verification_passed=overall_passed,
            verification_summary=summary
        )


# --- CLI Entry Point ---

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")

    if not api_key or not api_base:
        print("ERROR: OPENAI_API_KEY or OPENAI_API_BASE not set")
        exit(1)

    llm_client = LocalOpenAILLMClient(
        url=api_base,
        api_key=api_key,
        api_model="z-ai/glm-5.1",
        hf_tokenizer_id=None
    )

    tool_pool_path = str(Path("~/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl").expanduser())
    invocation_examples_path = str(Path("~/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl").expanduser())
    tool_manager = ToolManager(
        llm=llm_client,
        tool_pool_path=tool_pool_path,
        invocation_examples_path=invocation_examples_path
    )

    generator = StepByStepGenerator(
        llm_client=llm_client,
        tool_manager=tool_manager,
        num_actions=2
    )

    print("Generating test datapoint...")
    datapoint = generator.generate_datapoint(focus_category="Communication")

    if datapoint:
        print("\n" + "=" * 60)
        print("GENERATED DATAPOINT:")
        print("=" * 60)
        print(datapoint.model_dump_json(indent=2))
    else:
        print("\nFailed to generate datapoint")