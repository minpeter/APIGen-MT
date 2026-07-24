"""State-backed blueprint entity validation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from src.multi_turn_protocols import ApiState, BlueprintTurn, GeneratorMixinBase
else:
    from multi_turn_protocols import GeneratorMixinBase

from multi_turn_protocols import (
    is_object_dict,
    is_object_list,
    string_list,
    string_value,
)

_POSTING_TOOLS = {
    "get_user_tweets",
    "get_user_stats",
    "follow_user",
    "unfollow_user",
    "authenticate_twitter",
    "mention",
    "comment",
    "retweet",
    "search_tweets",
    "get_tweet",
    "get_tweet_comments",
}
_USERNAME_PATTERNS = (
    r"from\s+user\s+(\w+)",
    r"follow\s+(\w+)",
    r"following\s+(\w+)",
    r"tweets?\s+from\s+(\w+)",
    r"by\s+user\s+(\w+)",
    r"username\s+(\w+)",
    r"@(\w+)",
)


def _add_username(container: set[str], value: object) -> None:
    if isinstance(value, str) and value:
        container.add(value)


class EntityValidationMixin(GeneratorMixinBase):
    """Validate domain entities referenced by generated blueprint turns."""

    @override
    def _validate_posting_api_entities(
        self,
        turns: list[BlueprintTurn],
        initial_api_state: ApiState | None = None,
    ) -> list[str]:
        """Validate that PostingApi query usernames exist in state."""
        if not initial_api_state:
            return []

        posting_state = initial_api_state.get(
            "posting_api"
        ) or initial_api_state.get("PostingAPI")
        if not posting_state:
            return []

        valid_usernames: set[str] = set()
        users = posting_state.get("users")
        if is_object_dict(users):
            for username in users:
                _add_username(valid_usernames, username)
        following = posting_state.get("following_list")
        if is_object_list(following):
            for username in following:
                _add_username(valid_usernames, username)
        tweets = posting_state.get("tweets")
        if is_object_dict(tweets):
            for tweet in tweets.values():
                if is_object_dict(tweet):
                    _add_username(valid_usernames, tweet.get("username"))
        _add_username(valid_usernames, posting_state.get("username"))
        comments = posting_state.get("comments")
        if is_object_dict(comments):
            for comment_list in comments.values():
                if is_object_list(comment_list):
                    for comment in comment_list:
                        if is_object_dict(comment):
                            _add_username(
                                valid_usernames,
                                comment.get("username"),
                            )
        retweets = posting_state.get("retweets")
        if is_object_list(retweets):
            for retweet in retweets:
                if is_object_dict(retweet):
                    _add_username(valid_usernames, retweet.get("username"))

        normalized_usernames = {username.lower() for username in valid_usernames}
        issues: list[str] = []
        for turn_index, turn in enumerate(turns, 1):
            query = string_value(turn, "user_query")
            expected_tools = string_list(turn, "expected_tools")
            if not any(tool in _POSTING_TOOLS for tool in expected_tools):
                continue

            found_usernames: set[str] = set()
            for pattern in _USERNAME_PATTERNS:
                found_usernames.update(re.findall(pattern, query, re.IGNORECASE))

            for username in found_usernames:
                if username.lower() not in normalized_usernames:
                    issues.append(
                        f"Turn {turn_index}: query references username '{username}' but "
                        + f"'{username}' does not exist in state. Valid users: "
                        + f"{', '.join(sorted(valid_usernames))[:100]}"
                    )

        return issues

    @override
    def _validate_vehicle_control_queries(
        self,
        turns: list[BlueprintTurn],
        initial_api_state: ApiState | None = None,
    ) -> list[str]:
        """Validate fuel-related queries against initial vehicle state."""
        if not initial_api_state:
            return []

        vehicle_state = initial_api_state.get(
            "vehicle_control"
        ) or initial_api_state.get("VehicleControlAPI")
        if not vehicle_state:
            return []

        initial_fuel = vehicle_state.get("fuelLevel")
        if not isinstance(initial_fuel, int | float):
            return []

        fuel_fill_patterns = (
            r"fill.*tank",
            r"add.*fuel",
            r"top.*up",
            r"refuel",
            r"fuel.*fill",
        )
        issues: list[str] = []
        for turn_index, turn in enumerate(turns, 1):
            query = string_value(turn, "user_query")
            expected_tools = string_list(turn, "expected_tools")
            if not any(
                tool in {"fillFuelTank", "addFuel"}
                for tool in expected_tools
            ):
                continue

            for pattern in fuel_fill_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    if initial_fuel >= 50.0:
                        issues.append(
                            f"Turn {turn_index}: query asks to 'fill/add fuel' but "
                            + f"initial fuelLevel is {initial_fuel} (tank is at or "
                            + "above max capacity of 50.0). This scenario is "
                            + "incoherent - tank cannot be filled when full. Use a "
                            + "config with fuelLevel < 50 or change the query to match state."
                        )
                    break

        return issues
