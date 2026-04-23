"""Prompt templates for step-by-step datapoint generation."""

from typing import Optional, List


class StepByStepPrompts:
    """Collection of prompts for the three-stage generation process."""

    @staticmethod
    def get_query_generation_prompt(
        num_actions: int,
        tools_with_descriptions: str,
        example_queries: str,
        focus_category: Optional[str],
        accumulated_feedback: str
    ) -> str:
        """Generate the prompt for Stage 1: Query Generation."""
        
        prompt_parts = [
            "You are generating a realistic user query for testing a tool-calling system.",
            "",
            f"Generate a natural, realistic user query that would require using EXACTLY {num_actions} tools to fulfill.",
            "",
            "=== REQUIREMENTS ===",
            "1. The query should be specific and actionable",
            "2. It should mention concrete entities (names, IDs, dates, locations, etc.)",
            f"3. It should require EXACTLY {num_actions} tool calls to complete - not more, not less",
            f"4. The expected_tools list must contain EXACTLY {num_actions} tool names",
            "5. CRITICAL: Use ONLY the exact tool names from the AVAILABLE TOOLS section below",
            "6. CRITICAL: Do NOT invent tool names - only use tools that exist in the list",
            "7. The tools should logically fit together to accomplish the query",
            "",
            "=== AVAILABLE TOOLS WITH DESCRIPTIONS ===",
            tools_with_descriptions,
            example_queries,
        ]
        
        if focus_category:
            prompt_parts.extend([
                "",
                f"=== FOCUS CATEGORY ===",
                f"Primary category: {focus_category} (select tools primarily from this category)",
            ])
        
        if accumulated_feedback:
            prompt_parts.extend([
                "",
                "=== PREVIOUS ATTEMPT FEEDBACK ===",
                accumulated_feedback,
                "=== END FEEDBACK ===",
            ])
        
        prompt_parts.extend([
            "",
            "=== YOUR TASK ===",
            f"Generate a query for category: {focus_category or 'any'} that requires EXACTLY {num_actions} tools from the AVAILABLE TOOLS list above.",
            "",
            "The query should be realistic and the expected_tools must be EXACT names from the available tools list.",
            "",
            "Respond ONLY with valid JSON in this exact format:",
            "{",
            '    "query": "the generated user query - be specific with names, dates, IDs",',
            '    "intent": "brief description of what the user wants to accomplish",',
            f'    "expected_tools": ["tool_name_1", "tool_name_2", ...] // EXACTLY {num_actions} tools from AVAILABLE TOOLS',
            "}",
        ])
        
        return "\n".join(prompt_parts)

    @staticmethod
    def get_tool_sequence_validation_prompt(
        query: str,
        expected_tools: List[str],
        intent: str,
        tool_schemas: str
    ) -> str:
        """Generate the prompt for validating tool sequence."""
        
        return f"""You are validating a tool sequence plan for a user query.

User Query: {query}
Intent: {intent}

Planned Tool Sequence: {expected_tools}

Tool Schemas:
{tool_schemas}

Evaluate if the sequence logically fits the query intent.

Respond with JSON:
{{
    "is_valid": true/false,
    "issues": ["list of issues if any"]
}}"""

    @staticmethod
    def get_tool_arguments_prompt(
        tool_name: str,
        tool_schema: dict,
        query: str,
        trajectory_str: str,
        execution_context: dict
    ) -> str:
        """Generate the prompt for generating tool arguments."""
        
        context_str = str(execution_context)[:500] if execution_context else "{}"
        
        return f"""Generate arguments for the tool '{tool_name}' based on the user query and previous steps.

=== USER QUERY ===
{query}

=== PREVIOUS STEPS ===
{trajectory_str if trajectory_str else "None"}

=== EXECUTION CONTEXT ===
{context_str}

=== TOOL SCHEMA ===
{tool_schema}

=== YOUR TASK ===
Generate arguments for '{tool_name}' that:
1. Match the schema above
2. Fulfill the user query
3. Use values from Execution Context when available (e.g., user_id from previous step)
4. Are specific and realistic

Respond with JSON containing only the arguments:
{{
    "arg1": "value1",
    "arg2": "value2"
}}"""

    @staticmethod
    def get_final_response_prompt(query: str, actions_summary: List[dict]) -> str:
        """Generate the prompt for final response generation."""
        
        import json
        actions_json = json.dumps(actions_summary, indent=2)
        
        return f"""Based on the following conversation, generate a natural final response.

User Query: {query}

Actions taken:
{actions_json}

Generate a concise, natural response that summarizes what was accomplished."""
