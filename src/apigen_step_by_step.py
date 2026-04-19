#!/usr/bin/env python3
"""
Step-by-step blueprint generator for APIGen-MT.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import re
import copy
import os
from llm_client import LLMClient, LocalOpenAILLMClient
from tool_manager import ToolManager


class ToolCallWithOutput(BaseModel):
    """A single tool call with its simulated output."""
    tool_name: str
    arguments: Dict[str, Any] = {}
    output: Any = None


class TrajectoryStep(BaseModel):
    """A single step in the conversation trajectory."""
    step_number: int
    tool_calls: List[ToolCallWithOutput] = []
    reasoning: Optional[str] = None


class ConversationTrajectory(BaseModel):
    """Complete conversation trajectory for a datapoint."""
    query: str
    steps: List[TrajectoryStep] = []
    final_response: str
    tools_used: List[str] = []
    categories_used: List[str] = []


class StepByStepDatapoint(BaseModel):
    """Complete datapoint generated step-by-step."""
    trajectory: ConversationTrajectory
    generation_metadata: Dict[str, Any] = {}
    verification_result: Optional[Dict[str, Any]] = None


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

    def __init__(self, llm_client: LLMClient, tool_manager: ToolManager, num_actions: int = 2, validate_outputs: bool = True):
        self.llm = llm_client
        self.tool_manager = tool_manager
        self.num_actions = num_actions
        self.validate_outputs = validate_outputs

    def _get_tool_schemas_str(self, tools_subset: Optional[List[str]] = None) -> str:
        schemas = self.tool_manager.get_tools_json_schema()
        if tools_subset:
            schemas = [s for s in schemas if s['name'] in tools_subset]
        return json.dumps(schemas, indent=2, ensure_ascii=False)

    def _get_tools_with_descriptions_str(self, category: Optional[str] = None) -> str:
        """Get a formatted string of tools with their full descriptions, organized by category."""
        tools = self.tool_manager.get_tools_json_schema()
        
        if category:
            tools = [t for t in tools if t.get('category') == category]
        
        # Group by category
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
                # Truncate very long descriptions
                if len(desc) > 200:
                    desc = desc[:197] + "..."
                result.append(f"  - {name}: {desc}")
        
        return "\n".join(result)

    def _process_placeholders(self, arguments: Dict[str, Any], execution_context: Dict[str, Any]) -> Dict[str, Any]:
        processed_args = copy.deepcopy(arguments)
        for arg_name, arg_value in processed_args.items():
            if isinstance(arg_value, str):
                placeholders = re.findall(r"\{\{([^{}]+)\}\}", arg_value)
                for placeholder_full_key in placeholders:
                    keys = placeholder_full_key.split('.')
                    current_value = execution_context
                    found = True
                    for key_part in keys:
                        if isinstance(current_value, dict) and key_part in current_value:
                            current_value = current_value[key_part]
                        else:
                            found = False
                            break
                    if found:
                        placeholder_tag = "{{" + placeholder_full_key + "}}"
                        if arg_value == placeholder_tag:
                            processed_args[arg_name] = current_value
                        else:
                            processed_args[arg_name] = arg_value.replace(placeholder_tag, str(current_value))
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

        response = self.llm.generate([{"role": "user", "content": prompt}])
        response_text = response.strip()

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif response_text.startswith("{"):
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            response_text = response_text[start:end]

        result = json.loads(response_text)
        is_valid = result.get("is_valid", False)
        issues = result.get("issues", [])

        if not is_valid:
            return False, f"Tool sequence validation failed: {'; '.join(issues)}"
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
                "query": "Buy 50 shares of Apple stock at market price, then add AAPL to my watchlist and notify me when the price changes by more than 5%.",
                "intent": "User wants to purchase Apple stock, monitor it, and receive price alerts",
                "expected_tools": ["get_symbol_by_name", "place_order", "add_to_watchlist"]
            },
            {
                "category": "Events",
                "num_tools": 4,
                "query": "My order #12345 never arrived. Create a ticket about this issue, get the ticket details, and update it with high priority since it's been 2 weeks.",
                "intent": "User has a delivery issue and wants to create and manage a support ticket",
                "expected_tools": ["create_ticket", "get_ticket", "edit_ticket", "get_user_tickets"]
            },
            {
                "category": "Storage",
                "num_tools": 3,
                "query": "Navigate to the project directory, check the disk usage of all files, and then display the contents of config.json.",
                "intent": "User wants to browse a directory, check file sizes, and view a configuration file",
                "expected_tools": ["cd", "du", "cat"]
            },
            {
                "category": "Communication",
                "num_tools": 2,
                "query": "Send a message to user john_doe saying 'Meeting at 3pm', but first get their user ID from their username.",
                "intent": "User wants to send a message to another user, requiring ID lookup first",
                "expected_tools": ["get_user_id", "send_message"]
            }
        ]
        
        # Filter examples by num_actions if possible
        filtered_examples = [ex for ex in examples if ex["num_tools"] <= self.num_actions + 1 and ex["num_tools"] >= self.num_actions - 1]
        if not filtered_examples:
            filtered_examples = examples[:3]  # Just take first 3
        
        result = []
        for i, ex in enumerate(filtered_examples, 1):
            result.append(f"\n=== EXAMPLE {i} ({ex['category']}, {ex['num_tools']} tools) ===")
            result.append(f"Query: \"{ex['query']}\"")
            result.append(f"Intent: {ex['intent']}")
            result.append(f"Expected tools: {ex['expected_tools']}")
        
        return "\n".join(result)

    def generate_user_query(self, focus_category: Optional[str] = None, context_hint: Optional[str] = None, validation_feedback: Optional[str] = None, max_retries: int = 3) -> QueryGenerationResult:
        # Get tools with full descriptions
        tools_with_descriptions = self._get_tools_with_descriptions_str(category=focus_category)
        
        accumulated_feedback = validation_feedback or ""
        example_queries = self._get_example_queries()

        for attempt in range(max_retries + 1):
            prompt = f"""You are generating a realistic user query for testing a tool-calling system.

Generate a natural, realistic user query that would require using EXACTLY {self.num_actions} tools to fulfill.

=== REQUIREMENTS ===
1. The query should be specific and actionable
2. It should mention concrete entities (names, IDs, dates, locations, etc.)
3. It should require EXACTLY {self.num_actions} tool calls to complete - not more, not less
4. The expected_tools list must contain EXACTLY {self.num_actions} tool names
5. CRITICAL: Use ONLY the exact tool names from the AVAILABLE TOOLS section below
6. CRITICAL: Do NOT invent tool names - only use tools that exist in the list
7. The tools should logically fit together to accomplish the query

=== AVAILABLE TOOLS WITH DESCRIPTIONS ===
{tools_with_descriptions}
{example_queries}
"""
            if focus_category:
                prompt += f"\n=== FOCUS CATEGORY ===\nPrimary category: {focus_category} (select tools primarily from this category)\n"

            if accumulated_feedback:
                prompt += f"\n=== PREVIOUS ATTEMPT FEEDBACK ===\n{accumulated_feedback}\n=== END FEEDBACK ===\n"

            prompt += f"""
=== YOUR TASK ===
Generate a query for category: {focus_category or 'any'} that requires EXACTLY {self.num_actions} tools from the AVAILABLE TOOLS list above.

The query should be realistic and the expected_tools must be EXACT names from the available tools list.

Respond ONLY with valid JSON in this exact format:
{{
    "query": "the generated user query - be specific with names, dates, IDs",
    "intent": "brief description of what the user wants to accomplish",
    "expected_tools": ["tool_name_1", "tool_name_2", ...] // EXACTLY {self.num_actions} tools from AVAILABLE TOOLS
}}"""

            try:
                response = self.llm.generate([{"role": "user", "content": prompt}])
                response_text = response.strip()

                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]
                elif response_text.startswith("{"):
                    start = response_text.find("{")
                    end = response_text.rfind("}") + 1
                    response_text = response_text[start:end]

                result = json.loads(response_text)
                expected_tools = result.get("expected_tools", [])

                # Validate tool count
                if len(expected_tools) != self.num_actions:
                    accumulated_feedback += f"\nAttempt {attempt + 1}: Expected {self.num_actions} tools, but got {len(expected_tools)}: {expected_tools}. Please generate EXACTLY {self.num_actions} tools."
                    print(f" Query generation attempt {attempt + 1}: Wrong tool count ({len(expected_tools)}/{self.num_actions})")
                    continue

                # Validate that tools exist
                all_tools_valid = True
                invalid_tools = []
                for tool in expected_tools:
                    if not self.tool_manager.tool_exists(tool):
                        all_tools_valid = False
                        invalid_tools.append(tool)

                if not all_tools_valid:
                # Get available tools in the focus category or all categories
                    available_tools = []
                    if focus_category:
                        cat_tools = self.tool_manager.get_tools_by_category(focus_category)
                        available_tools = [t['name'] for t in cat_tools[:20]]
                    else:
                        for cat in self.tool_manager.get_categories():
                            cat_tools = self.tool_manager.get_tools_by_category(cat)
                            available_tools.extend([t['name'] for t in cat_tools[:5]])

                # Build helpful feedback with suggestions
                    feedback_msg = f"\nAttempt {attempt + 1}: INVALID TOOLS: {invalid_tools}\n"
                    feedback_msg += f"\nThese tools do NOT exist. You MUST choose from the available tools.\n"
                    feedback_msg += f"\nAvailable tools (sample): {available_tools[:15]}\n"
                    feedback_msg += f"\nPlease select ONLY tools from the AVAILABLE TOOLS list with exact names.\n"

                    accumulated_feedback += feedback_msg
                    print(f" Query generation attempt {attempt + 1}: Invalid tools {invalid_tools}")
                    continue

                # Validate tool sequence makes sense for the query
                is_valid, validation_msg = self.validate_expected_tools(
                    result.get("query", ""),
                    expected_tools,
                    result.get("intent", "")
                )

                if not is_valid:
                    accumulated_feedback += f"\nAttempt {attempt + 1}: Tool sequence validation failed: {validation_msg}"
                    print(f" Query generation attempt {attempt + 1}: Tool sequence invalid")
                    continue

                print(f" Query generation successful after {attempt + 1} attempt(s)")
                return QueryGenerationResult(
                    query=result.get("query", ""),
                    intent=result.get("intent", ""),
                    expected_tools=expected_tools
                )

            except json.JSONDecodeError as e:
                print(f"JSON decode error on attempt {attempt + 1}: {e}")
                accumulated_feedback += f"\nAttempt {attempt + 1}: JSON parsing error. Please ensure valid JSON output."
            except Exception as e:
                print(f"Error generating query on attempt {attempt + 1}: {e}")
                accumulated_feedback += f"\nAttempt {attempt + 1}: Error occurred: {str(e)}"

        print(f"Failed to generate valid query after {max_retries + 1} attempts")
        return QueryGenerationResult(query="", intent="", expected_tools=[])

    def _generate_next_step(self, query: str, trajectory: List[TrajectoryStep], execution_context: Dict[str, Any], expected_tools: List[str], step_num: int = 1) -> StepSelectionResult:
        trajectory_str = ""
        for i, step in enumerate(trajectory):
            trajectory_str += f"\nStep {i+1}:"
            for tc in step.tool_calls:
                trajectory_str += f"\n  - {tc.tool_name}({json.dumps(tc.arguments)})"
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
                desc = schema.get('description', 'No description available.')[:150]
                tool_descriptions_str += f"  - {tool_name}: {desc}\n"
            except:
                tool_descriptions_str += f"  - {tool_name}: (no description)\n"

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
            response = self.llm.generate([{"role": "user", "content": prompt}])
            response_text = response.strip()

            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            elif response_text.startswith("{"):
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]

            result = json.loads(response_text)
            return StepSelectionResult(
                tool_name=result.get("tool_name", ""),
                arguments=result.get("arguments", {}),
                reasoning=result.get("reasoning", "")
            )
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return StepSelectionResult(tool_name="", arguments={}, reasoning="")
        except Exception as e:
            print(f"Error generating step: {e}")
            return StepSelectionResult(tool_name="", arguments={}, reasoning="")

    def _simulate_tool_execution(self, tool_name: str, arguments: Dict[str, Any], execution_context: Dict[str, Any]) -> Any:
        processed_args = self._process_placeholders(arguments, execution_context)
        return self.tool_manager.invoke_tool(tool_name=tool_name, params=processed_args)

    def generate_datapoint(self, focus_category: Optional[str] = None, context_hint: Optional[str] = None, max_retries: int = 3) -> Optional[StepByStepDatapoint]:
        print("\n" + "=" * 60)
        print("STEP-BY-STEP DATAPOINT GENERATION")
        print("=" * 60)

        accumulated_feedback = context_hint or ""

        for attempt in range(max_retries + 1):
            print(f"\n[Attempt {attempt + 1}/{max_retries + 1}] Generating user query...")
            query_result = self.generate_user_query(focus_category, accumulated_feedback)
            if not query_result.query:
                print("Failed to generate query")
                accumulated_feedback += f"\nAttempt {attempt + 1}: Failed to generate a valid query. Please try again with a clear, actionable request."
                continue

            print(f"Query: {query_result.query}")
            print(f"Intent: {query_result.intent}")
            print(f"Expected tools: {query_result.expected_tools}")

        trajectory = []
        execution_context = {}
        steps_completed = 0

        for step_num in range(1, self.num_actions + 1):
            print(f"\n[Step {step_num + 1}/{self.num_actions + 2}] Selecting tool for step {step_num}...")

            step_result = self._generate_next_step(
                query=query_result.query,
                trajectory=trajectory,
                execution_context=execution_context,
                expected_tools=query_result.expected_tools
            )

            if step_result.tool_name == "__FINAL_RESPONSE__":
                print("All expected tools have been used")
                break

            if not step_result.tool_name:
                print("Failed to generate step")
                break

            print(f"Selected tool: {step_result.tool_name}")
            print(f"Arguments: {json.dumps(step_result.arguments)}")
            print(f"Reasoning: {step_result.reasoning[:100]}...")

            print(f"Simulating {step_result.tool_name}...")
            output = self._simulate_tool_execution(
                tool_name=step_result.tool_name,
                arguments=step_result.arguments,
                execution_context=execution_context
            )

            print(f"Output: {str(output)[:200]}...")

            context_key = f"{step_result.tool_name}_output"
            if output and isinstance(output, dict):
                for k, v in output.items():
                    execution_context[f"{step_result.tool_name}_{k}"] = v
            execution_context[context_key] = output

            tool_call = ToolCallWithOutput(tool_name=step_result.tool_name, arguments=step_result.arguments, output=output)
            trajectory_step = TrajectoryStep(step_number=step_num, tool_calls=[tool_call], reasoning=step_result.reasoning)
            trajectory.append(trajectory_step)
            steps_completed += 1

            print(f"\n[Step {steps_completed + 2}/{self.num_actions + 2}] Generating final response...")
            final_response = self._generate_final_response(query_result.query, trajectory, execution_context)
            print(f"Final response: {final_response[:200]}...")

            tools_used = []
            categories_used = set()
            for step in trajectory:
                for tc in step.tool_calls:
                    if tc.tool_name not in tools_used:
                        tools_used.append(tc.tool_name)
                    cat = self.tool_manager.get_tool_category(tc.tool_name)
                    if cat:
                        categories_used.add(cat)

            conv_trajectory = ConversationTrajectory(
                query=query_result.query,
                steps=trajectory,
                final_response=final_response,
                tools_used=tools_used,
                categories_used=list(categories_used)
            )

            metadata = {
                "num_actions": steps_completed,
                "focus_category": focus_category,
                "query_intent": query_result.intent,
                "expected_tools": query_result.expected_tools
            }

            print("\n" + "=" * 60)
            print("RUNNING VERIFICATION")
            print("=" * 60)

            # Run full verification
            verification_result = self.run_full_verification(
                query=query_result.query,
                trajectory=trajectory,
                execution_context=execution_context
            )

            # Convert VerificationResult to dict for storage
            verification_dict = verification_result.model_dump()

            # Create the datapoint with verification results
            datapoint = StepByStepDatapoint(
                trajectory=conv_trajectory,
                generation_metadata=metadata,
                verification_result=verification_dict
            )

            print("\n" + "=" * 60)
            print("DATAPOINT GENERATION COMPLETE")
            print("=" * 60)
            print(f"Tools used: {tools_used}")
            print(f"Categories: {categories_used}")
            print(f"Verification: {'PASSED' if verification_result.overall_verification_passed else 'FAILED'}")

            # Only return the datapoint if all steps were completed
            if steps_completed == self.num_actions:
                return datapoint
            else:
                accumulated_feedback += f"\nAttempt {attempt + 1}: Only completed {steps_completed} steps instead of {self.num_actions}. Generate a query that clearly requires all {self.num_actions} tools."
                continue

        print(f"\nFailed to generate valid datapoint after {max_retries + 1} attempts")
        return None

    def _generate_final_response(self, query: str, trajectory: List[TrajectoryStep], execution_context: Dict[str, Any]) -> str:
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
            response = self.llm.generate([{"role": "user", "content": prompt}])
            return response.strip()
        except Exception as e:
            print(f"Error generating final response: {e}")
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

        issues = []
        for i, step in enumerate(trajectory):
            for tc in step.tool_calls:
                tool_name = tc.tool_name.lower()
                if 'create' in tool_name or 'update' in tool_name or 'send' in tool_name:
                    if i == 0 and not any(k in tool_name for k in ['create_new', 'send_notification']):
                        issues.append(f"Step {i+1}: {tc.tool_name} might need prior context")

        order_is_correct = len(issues) == 0
        details = "Order appears logical. " + "; ".join(issues) if issues else "No order issues detected."

        return {'order_is_correct': order_is_correct, 'order_verification_details': details}

    def verify_output_consistency(self, tool_name: str, step_number: int, output: Any, expected_type: str, expected_description: str) -> Dict[str, Any]:
        """Verify if a tool's output matches its declared type and description."""
        if output is None:
            return {'tool_name': tool_name, 'step_number': step_number, 'output_type_matches': False, 'output_description_matches': False, 'issues': ['Output is None']}

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
            if not type_compatible:
                output_type_matches = False
                issues.append(f"Type mismatch: expected {expected_type}, got {output_type}")

        output_description_matches = True
        if expected_description and output:
            output_str = str(output).lower()
            desc_words = set(expected_description.lower().split())
            output_words = set(output_str.split())
            overlap = len(desc_words & output_words)
            if overlap < 2 and len(desc_words) > 5:
                output_description_matches = False
                issues.append("Output may not match description")

        return {'tool_name': tool_name, 'step_number': step_number, 'output_type_matches': output_type_matches, 'output_description_matches': output_description_matches, 'issues': issues}

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

    def run_full_verification(self, query: str, trajectory: List[TrajectoryStep], execution_context: Dict[str, Any]) -> VerificationResult:
        """
        Run all verification checks on a generated datapoint.

        Verification checks:
        1. Tool relevance - are tools relevant for the query?
        2. Invocation order - are tools invoked in the right order?
        3. Output consistency - do outputs match tool declarations?
        4. Placeholder resolution - are all placeholders resolved?
        """
        print("\n=== Running Verification ===")

        # 1. Check tool relevance
        tool_relevance_checks = []
        all_relevant = True
        for step in trajectory:
            for tc in step.tool_calls:
                check = self.verify_tool_relevance(query, tc.tool_name, step)
                tool_relevance_checks.append(check)
                if not check['is_relevant']:
                    all_relevant = False
                print(f"  {tc.tool_name}: relevance={check['relevance_score']:.2f}, relevant={check['is_relevant']}")

        # 2. Verify invocation order
        print("Verifying invocation order...")
        order_result = self.verify_invocation_order(query, trajectory)
        print(f"  Order correct: {order_result['order_is_correct']}")
        print(f"  Details: {order_result['order_verification_details']}")

        # 3. Verify output consistency
        print("Verifying output consistency...")
        output_validations = []
        all_outputs_valid = True
        for step in trajectory:
            for tc in step.tool_calls:
                tool_schema = self.tool_manager.get_tool_schema(tc.tool_name)
                expected_type = tool_schema.get('output_type', 'unknown') if tool_schema else 'unknown'
                expected_desc = tool_schema.get('output_description', '') if tool_schema else ''

                validation = self.verify_output_consistency(tc.tool_name, step.step_number, tc.output, expected_type, expected_desc)
                output_validations.append(validation)
                if not validation['output_type_matches'] or not validation['output_description_matches']:
                    all_outputs_valid = False
                print(f"  {tc.tool_name}: type_match={validation['output_type_matches']}, desc_match={validation['output_description_matches']}")
                if validation['issues']:
                    for issue in validation['issues']:
                        print(f"    - {issue}")

        # 4. Check placeholder resolution
        print("Checking placeholder resolution...")
        placeholder_result = self.verify_placeholder_resolution(trajectory, execution_context)
        print(f"  Resolved: {placeholder_result['resolved_count']}/{placeholder_result['total_placeholders']}")

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

        summary = "Verification PASSED - all checks successful" if overall_passed else "Verification FAILED - issues found"
        print(f"\n=== Verification Result: {'PASSED' if overall_passed else 'FAILED'} ===")
        if issues:
            for issue in issues:
                print(f"  - {issue}")

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
        api_model="nvidia/nemotron-3-super-120b-a12b",
        hf_tokenizer_id=None
    )

    tool_pool_path = "/home/ishalyminov/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl"
    tool_manager = ToolManager(llm=llm_client, tool_pool_path=tool_pool_path)

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