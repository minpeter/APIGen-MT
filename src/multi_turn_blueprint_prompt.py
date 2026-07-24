"""Prompt construction for multi-turn dialog blueprints."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.multi_turn_protocols import GeneratorMixinBase


def build_blueprint_prompt(
    generator: GeneratorMixinBase,
    focus_category: str | None,
    tools_str: str,
    output_fields_str: str,
    initial_state_context: str,
    credential_context: str,
    domain_hints_getter: Callable[[str], str | None],
) -> str:
    """Build the Stage 0 prompt without changing its public wording."""
    prompt = f"""Design a {generator.num_turns}-turn user-agent conversation. Each turn: USER request → AGENT calls 1-3 tools → AGENT responds.

=== AVAILABLE TOOLS ===
{tools_str}

=== OUTPUT SCHEMAS (use these exact field names in placeholders) ===
{output_fields_str}

=== REQUIREMENTS ===
 1. Each turn: specific entities (IDs, names, dates, prices) + 1-3 tools (vary naturally based on query complexity)
 2. Conversation flows naturally, each turn builds on previous
 3. Auth persists across turns - login only in FIRST turn needing auth (don't re-login)
 4. expected_tools: aim for about {generator.target_num_actions} tools per turn (allow small variation; at least 1)
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
        domain_hints = domain_hints_getter(focus_category)
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

    return prompt
