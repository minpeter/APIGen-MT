"""Stateful workspace messaging API."""

from typing import TypeIs

from .message_api_types import (
    AddContactResult,
    DeleteMessageResult,
    MessageLoginResult,
    MessageSearchMatch,
    MessageSearchResult,
    SendMessageResult,
    UserIdResult,
)
from .type_utils import (
    Config,
    Record,
    get_int,
    get_str,
    get_string_map,
    is_object_list,
    is_record,
)

type MessageEntry = str | Record
type ConversationMap = dict[str, list[MessageEntry]]
type MessageMap = dict[str, ConversationMap]


def _is_message_list(value: object) -> TypeIs[list[MessageEntry]]:
    return is_object_list(value) and all(
        isinstance(item, str) or is_record(item) for item in value
    )


def _is_conversation_map(value: object) -> TypeIs[ConversationMap]:
    return is_record(value) and all(
        _is_message_list(messages) for messages in value.values()
    )


def _is_message_map(value: object) -> TypeIs[MessageMap]:
    return is_record(value) and all(
        _is_conversation_map(conversations) for conversations in value.values()
    )


def _get_message_map(config: Config, key: str) -> MessageMap:
    value = config.get(key)
    return value if _is_message_map(value) else {}


def _message_text(entry: MessageEntry) -> str:
    if isinstance(entry, str):
        return entry
    return get_str(entry, "message")


class MessageAPI:
    """Manage user interactions and messages in a workspace."""

    def __init__(self, initial_config: Config) -> None:
        """Initialize the API with the given configuration."""
        self.workspace_id: str = get_str(initial_config, "workspace_id")
        self.user_count: int = get_int(initial_config, "user_count")
        self.user_map: dict[str, str] = get_string_map(
            initial_config, "user_map"
        )
        self.messages_sent_map: MessageMap = _get_message_map(
            initial_config, "messages_sent_map"
        )
        self.messages_inbox_map: MessageMap = _get_message_map(
            initial_config, "messages_inbox_map"
        )
        self.message_count: int = get_int(initial_config, "message_count")
        self.current_user: str = get_str(initial_config, "current_user")

    def add_contact(self, user_name: str) -> AddContactResult:
        """Add a contact to the workspace."""
        if not user_name:
            return {
                "added_status": False,
                "user_id": "",
                "message": "User name cannot be empty.",
            }
        existing_name = next(
            (
                name
                for name in self.user_map
                if name.lower() == user_name.lower()
            ),
            None,
        )
        if existing_name:
            return {
                "added_status": False,
                "user_id": self.user_map[existing_name],
                "message": f"User '{user_name}' already exists in the workspace.",
            }

        existing_ids = (
            set(self.user_map.values())
            | self.messages_sent_map.keys()
            | self.messages_inbox_map.keys()
        )
        max_number = 0
        for user_id in existing_ids:
            try:
                max_number = max(max_number, int(user_id.replace("USR", "")))
            except ValueError:
                continue

        new_user_id = f"USR{max_number + 1:03d}"
        self.user_count += 1
        self.user_map[user_name] = new_user_id
        self.messages_sent_map[new_user_id] = {}
        self.messages_inbox_map[new_user_id] = {}
        return {
            "added_status": True,
            "user_id": new_user_id,
            "message": f"Contact '{user_name}' added successfully.",
        }

    def delete_message(
        self, receiver_id: str, message_id: int | None = None
    ) -> DeleteMessageResult:
        """Delete one message sent by the current user to a receiver."""
        if not self.current_user:
            return {
                "deleted_status": False,
                "message": "User is not authenticated. Please log in first.",
            }
        conversation = self.messages_sent_map.get(
            self.current_user, {}
        ).get(receiver_id)
        if not conversation:
            return {
                "deleted_status": False,
                "message": "No messages found for receiver.",
            }

        target_index: int | None = None
        if message_id is None:
            target_index = len(conversation) - 1
        else:
            for index, entry in enumerate(conversation):
                if isinstance(entry, str):
                    stored_id = index + 1
                else:
                    raw_id = entry.get("message_id")
                    if not isinstance(raw_id, int | str):
                        continue
                    try:
                        stored_id = int(raw_id)
                    except ValueError:
                        continue
                if stored_id == message_id:
                    target_index = index
                    break

        if target_index is None or target_index < 0:
            return {"deleted_status": False, "message": "Message not found."}
        del conversation[target_index]
        return {
            "deleted_status": True,
            "message": "Message deleted successfully.",
        }

    def get_user_id(self, user: str) -> UserIdResult:
        """Get a user ID from a user name."""
        for name, user_id in self.user_map.items():
            if user and name.lower() == user.lower():
                return {"user_id": user_id}
        return {"user_id": ""}

    def message_login(self, user_id: str) -> MessageLoginResult:
        """Select a workspace user as the current messaging user."""
        if not user_id:
            return {
                "login_status": False,
                "message": "User ID cannot be empty.",
            }
        user_exists = (
            user_id in self.user_map.values()
            or user_id in self.messages_sent_map
            or user_id in self.messages_inbox_map
        )
        if not user_exists:
            return {
                "login_status": False,
                "message": f"User ID '{user_id}' not found in the workspace.",
            }
        self.current_user = user_id
        return {
            "login_status": True,
            "message": f"User '{user_id}' logged in successfully.",
        }

    def search_messages(self, keyword: str) -> MessageSearchResult:
        """Search the current user's sent and received messages."""
        results: list[MessageSearchMatch] = []
        if not keyword or not self.current_user:
            return {"results": results}
        keyword = keyword.lower()
        for receiver_id, messages in self.messages_sent_map.get(
            self.current_user, {}
        ).items():
            for entry in messages:
                message = _message_text(entry)
                if keyword in message.lower():
                    results.append(
                        {
                            "receiver_id": receiver_id,
                            "message": message,
                            "direction": "sent",
                        }
                    )
        for sender_id, messages in self.messages_inbox_map.get(
            self.current_user, {}
        ).items():
            for entry in messages:
                message = _message_text(entry)
                if keyword in message.lower():
                    results.append(
                        {
                            "sender_id": sender_id,
                            "message": message,
                            "direction": "received",
                        }
                    )
        return {"results": results}

    def send_message(self, receiver_id: str, message: str) -> SendMessageResult:
        """Send a message from the current user to a receiver."""
        if not self.current_user:
            return {
                "sent_status": False,
                "message_id": "0",
                "message": "No user currently logged in.",
            }
        if not receiver_id:
            return {
                "sent_status": False,
                "message_id": "0",
                "message": "Receiver ID cannot be empty.",
            }
        if not message:
            return {
                "sent_status": False,
                "message_id": "0",
                "message": "Message cannot be empty.",
            }

        self.message_count += 1
        sender_id = self.current_user
        message_record: Record = {
            "message_id": self.message_count,
            "message": message,
        }
        sent = self.messages_sent_map.setdefault(sender_id, {})
        sent.setdefault(receiver_id, []).append(message_record)
        inbox = self.messages_inbox_map.setdefault(receiver_id, {})
        inbox.setdefault(sender_id, []).append(message_record)
        return {
            "sent_status": True,
            "message_id": str(self.message_count),
            "message": "Message sent successfully.",
        }
