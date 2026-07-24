"""User-query generation and stage-one query orchestration."""

import json
from typing import Protocol, override

from step_by_step_models import ObjectMap, QueryGenerationResult, StateSnapshot
from step_by_step_protocols import StepByStepMixinBase, is_object_map
from tool_manager import filter_api_state


class QueryGenerationMixin(StepByStepMixinBase, Protocol):
    def generate_user_query(
        self,
        focus_category: str | None = None,
        validation_feedback: str | None = None,
        max_retries: int = 3,
        query_seed: ObjectMap | None = None,
        initial_api_state: StateSnapshot | None = None,
    ) -> QueryGenerationResult:
        # Query generation needs complete parameter and output schemas. Compact
        # descriptions omit the information needed to plan argument dependencies.
        tools_for_prompt = self.tool_manager.get_tools_json_schema()
        if focus_category:
            tools_for_prompt = [
                tool for tool in tools_for_prompt
                if tool.get("category") == focus_category
            ]
        tools_with_descriptions = json.dumps(
            tools_for_prompt,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        # The initial state is generator-only. The policy will not see it, so any
        # state value needed by a target call must be surfaced in the user query
        # unless an earlier tool call returns it.
        state_for_prompt = initial_api_state
        if initial_api_state and tools_for_prompt:
            state_for_prompt = filter_api_state(
                initial_api_state,
                [
                    name
                    for tool in tools_for_prompt
                    if isinstance(name := tool.get("name", ""), str)
                ],
            )
        generator_state_section = ""
        if state_for_prompt:
            generator_state_section = f"""
=== GENERATOR-ONLY API STATE ===
This state is visible only while constructing the synthetic task. It will NOT be
shown to the assistant that solves the task. Use it to choose valid values, but
copy every required value into the generated user query unless that value is
returned by an earlier expected tool call.

{json.dumps(state_for_prompt, indent=2, ensure_ascii=False, default=str)}
"""

        accumulated_feedback = validation_feedback or ""
        example_queries = self._get_example_queries()

        persona_section = ""
        if query_seed:
            persona = query_seed.get("persona")
            city = query_seed.get("city")
            if not is_object_map(persona) or not is_object_map(city):
                raise ValueError("query seed must contain persona and city mappings")
            name = persona.get("name", "")
            home_city = persona.get("city", "")
            country = persona.get("country", "")
            travel_city = city.get("city", "")
            persona_section = f"""
=== USER PERSONA (MANDATORY) ===
The user's name is {name}, they are based in {home_city}, {country}.
You MUST use this person's name and city in the query when mentioning people or locations.
For example, if the query involves booking a flight, use {travel_city} as origin/destination.
If the query involves a credit card, use the name {name} as cardholder.
Do NOT use "Michael Smith", "John", or generic American names - use {name} exclusively.
"""

        for attempt in range(max_retries):
            prompt = f"""Generate a realistic user query requiring one or more tool calls to fulfill.

=== REQUIREMENTS ===
1. Specific with concrete entities (names, IDs, dates, locations)
2. Use one or more tool calls as needed to complete the task
3. expected_tools: List all tools from AVAILABLE TOOLS that would be needed
   Target about {self.target_num_actions} tools (prefer this count when realistic)
4. CRITICAL: Match query to tool capabilities - if a tool (like displayCarStatus) only accepts ONE parameter value, ask for ONE thing per query, not multiple
5. CRITICAL: Use ONLY tools from AVAILABLE TOOLS - no invented names
6. Auth-dependent tools need authentication FIRST - check which tools require prior authentication
7. POLICY-CONTEXT CLOSURE: Every required argument for every expected tool must
   be available from the generated user query, an earlier expected tool output,
   or a default declared in that tool's schema.
8. The solving assistant receives only the user query, the complete tool schemas,
   and prior tool outputs. It does NOT receive the API state below.
9. If a required value exists only in API state, write that exact value naturally
   into the user query. If an earlier tool produces it, place that tool before the
   dependent tool. Never require the assistant to guess a value.
10. Match the exact semantic representation required by the tool definition. A
    human-readable label is not interchangeable with an opaque identifier, code,
    token, symbol, handle, coordinate, path, or credential.
11. General/model knowledge is not an argument source. If an opaque value is not
    written in the user query and is not returned by an earlier expected tool,
    choose a different task or include the required lookup within the call budget.
{persona_section}
=== AVAILABLE TOOLS ===
{tools_with_descriptions}
{generator_state_section}
{example_queries}"""
            if focus_category:
                prompt += f"\n=== FOCUS CATEGORY ===\nPrimary: {focus_category}\n"
            if accumulated_feedback:
                prompt += f"\n=== FEEDBACK ===\n{accumulated_feedback}\n"
            prompt += """
=== TASK ===
Generate query that realistically requires multiple tools. Respond JSON:
{"query": "specific with names/IDs", "intent": "what user wants", "expected_tools": ["tool1", ...]}"""

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

                result = QueryGenerationResult.model_validate_json(response_text)
                query = result.query
                intent = result.intent
                expected_tools = result.expected_tools

                print(f" Generated Query: {query}")
                print(f" Intent: {intent}")
                print(f" Expected tools: {expected_tools}")

                generated_summary = f"""--- ATTEMPT {attempt + 1} OUTPUT ---
Query: {query}
Intent: {intent}
Expected tools: {expected_tools}"""

                all_tools_valid = True
                invalid_tools: list[str] = []
                for tool in expected_tools:
                    if not self.tool_manager.tool_exists(tool):
                        all_tools_valid = False
                        invalid_tools.append(tool)

                if not all_tools_valid:
                    available_tools: list[str] = []
                    categories = (
                        [focus_category]
                        if focus_category
                        else self.tool_manager.get_categories()
                    )
                    per_category = 20 if focus_category else 5
                    for category in categories:
                        schemas = self.tool_manager.get_tools_by_category(category)
                        available_tools.extend(
                            name
                            for schema in schemas[:per_category]
                            if isinstance(name := schema.get("name"), str)
                        )

                    print(f" ✗ Invalid tools: {invalid_tools}")
                    accumulated_feedback += f"""\n{generated_summary}
FAILURE: Tools not found: {invalid_tools}
These tools do NOT exist. Choose from available tools.
Available tools (sample): {available_tools[:15]}
--- END ATTEMPT {attempt + 1} ---"""
                    continue

                is_valid, validation_msg = self.validate_expected_tools(query, expected_tools, intent)

                if not is_valid:
                    print(f" ✗ Tool sequence validation failed: {validation_msg}")
                    accumulated_feedback += f"\n{generated_summary}\nFAILURE: Tool sequence validation - {validation_msg}\n--- END ATTEMPT {attempt + 1} ---"
                    continue

                print(" ✓ Query generation successful")
                return result

            except ValueError as exc:
                print(f" ✗ JSON decode error: {exc}")
                accumulated_feedback += f"\n--- ATTEMPT {attempt + 1} FAILED ---\nJSON parsing error: {exc}\n--- END ATTEMPT {attempt + 1} ---"
                continue

        print(f" Failed to generate valid query after {max_retries} attempts")
        return QueryGenerationResult(query="", intent="", expected_tools=[])

    @override
    def _stage1_generate_query(
        self,
        focus_category: str | None,
        context_hint: str | None,
        max_retries: int,
        query_seed: ObjectMap | None = None,
        initial_api_state: StateSnapshot | None = None,
    ) -> QueryGenerationResult | None:
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
            query_result = self.generate_user_query(
                focus_category,
                accumulated_feedback if accumulated_feedback else None,
                query_seed=query_seed,
                initial_api_state=initial_api_state,
            )

            if not query_result.query:
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
            print("  Verifying expected tools...")

            if not query_result.expected_tools:
                print("  ✗ ERROR: expected_tools is empty")
                accumulated_feedback += f"\n{generated_summary}\nFAILURE: expected_tools is empty.\n--- END ATTEMPT {attempt + 1} ---"
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
