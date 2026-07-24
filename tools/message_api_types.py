"""Typed public results returned by MessageAPI operations."""

from typing import TypedDict


class AddContactResult(TypedDict):
    """Result from adding a workspace contact."""

    added_status: bool
    user_id: str
    message: str


class DeleteMessageResult(TypedDict):
    """Result from deleting a message."""

    deleted_status: bool
    message: str


class UserIdResult(TypedDict):
    """Result from looking up a workspace user."""

    user_id: str


class MessageLoginResult(TypedDict):
    """Result from selecting the current messaging user."""

    login_status: bool
    message: str


class MessageSearchMatch(TypedDict, total=False):
    """One matching sent or received message."""

    receiver_id: str
    sender_id: str
    message: str
    direction: str


class MessageSearchResult(TypedDict):
    """Result from searching the current user's messages."""

    results: list[MessageSearchMatch]


class SendMessageResult(TypedDict):
    """Result from sending a message."""

    sent_status: bool
    message_id: str
    message: str
