"""Structural validation for generated dialog blueprints."""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.multi_turn_protocols import BlueprintTurn, GeneratorMixinBase

from multi_turn_protocols import string_list, string_value

CROSS_TURN_ENTITY_TOOLS: dict[str, tuple[str, str]] = {
    "comment": ("tweet_id", "post_tweet"),
    "retweet": ("tweet_id", "post_tweet"),
    "mention": ("tweet_id", "post_tweet"),
    "edit_ticket": ("ticket_id", "create_ticket"),
    "resolve_ticket": ("ticket_id", "create_ticket"),
    "close_ticket": ("ticket_id", "create_ticket"),
    "delete_message": ("message_id", "send_message"),
    "purchase_insurance": ("booking_id", "book_flight"),
}


def validate_blueprint_turns(
    generator: GeneratorMixinBase,
    turns: list[BlueprintTurn],
    focus_category: str | None,
    output_fields_validation_map: dict[str, list[str]],
) -> tuple[bool, list[str]]:
    """Return the original Stage 0 structural-validity verdict and errors."""
    validation_errors: list[str] = []
    all_tools_valid = True
    for turn_index, turn in enumerate(turns):
        expected = string_list(turn, "expected_tools")
        if not 1 <= len(expected) <= 3:
            validation_errors.append(
                f"Turn {turn_index + 1} has {len(expected)} tools, need 1-3: "
                + f"{expected}"
            )
            all_tools_valid = False
            break

        if focus_category:
            for tool_name in expected:
                tool_category = generator.tool_manager.get_tool_category(tool_name)
                if tool_category != focus_category:
                    validation_errors.append(
                        f"Turn {turn_index + 1} tool '{tool_name}' is from category "
                        + f"'{tool_category}', not '{focus_category}'. Use only "
                        + f"{focus_category} tools."
                    )
                    all_tools_valid = False
                    break
            if not all_tools_valid:
                break

        duplicate_tools = [
            tool for tool, count in Counter(expected).items() if count > 1
        ]
        if duplicate_tools:
            validation_errors.append(
                f"Turn {turn_index + 1} has duplicate tools that can't share "
                + f"arguments: {duplicate_tools}. A single LLM call can't generate "
                + "distinct args for the same tool called twice. Use different "
                + "tools instead."
            )
            all_tools_valid = False
            break

        query = string_value(turn, "user_query")
        placeholders = re.finditer(
            r"\{\{TURN(\d+)\.(\w+)\.(\w+)\}\}",
            query,
        )
        for placeholder_match in placeholders:
            referenced_turn = placeholder_match.group(1)
            referenced_tool = placeholder_match.group(2)
            referenced_field = placeholder_match.group(3)
            referenced_turn_index = int(referenced_turn) - 1
            if referenced_turn_index >= turn_index:
                validation_errors.append(
                    f"Turn {turn_index + 1} placeholder references future turn "
                    + referenced_turn
                )
                all_tools_valid = False
                break
            if referenced_turn_index < len(turns):
                referenced_tools = string_list(
                    turns[referenced_turn_index], "expected_tools"
                )
                if referenced_tool not in referenced_tools:
                    validation_errors.append(
                        f"Turn {turn_index + 1} references {referenced_tool} from "
                        + f"turn {referenced_turn}, but that turn uses "
                        + f"{referenced_tools}"
                    )
                    all_tools_valid = False
                    break
                known_fields = output_fields_validation_map.get(
                    referenced_tool,
                    ["success", "message", "id", "result"],
                )
                if referenced_field not in known_fields:
                    validation_errors.append(
                        f"Turn {turn_index + 1} placeholder {{TURN{referenced_turn}."
                        + f"{referenced_tool}.{referenced_field}}}: "
                        + f"'{referenced_field}' not in {referenced_tool} output. "
                        + f"Use: {known_fields}"
                    )
                    all_tools_valid = False
                    break
        if not all_tools_valid:
            break

    for turn_index, turn in enumerate(turns):
        if turn_index == 0:
            continue
        expected = string_list(turn, "expected_tools")
        query = string_value(turn, "user_query")
        for tool_name in expected:
            entity_dependency = CROSS_TURN_ENTITY_TOOLS.get(tool_name)
            if entity_dependency is None:
                continue
            id_field, create_tool = entity_dependency
            prior_tools = string_list(turns[turn_index - 1], "expected_tools")
            if create_tool in prior_tools:
                placeholder = f"{{{{TURN{turn_index}.{create_tool}.{id_field}}}}}"
                if placeholder not in query:
                    validation_errors.append(
                        f"Turn {turn_index + 1} uses '{tool_name}' to operate on "
                        + f"{create_tool} result but query lacks placeholder "
                        + f"'{id_field}'"
                    )
                    all_tools_valid = False
                    break
        if not all_tools_valid:
            break

    return all_tools_valid, validation_errors
