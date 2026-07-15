"""Auto-generated MessageAPI implementation."""

import json
import math
import re
import copy
from datetime import datetime
from typing import List, Dict, Any, Optional


class MessageAPI:
    """API to manage user interactions and messaging in a workspace."""

    def __init__(self, initial_config: dict) -> None:
        """Initialize the MessageAPI with the given configuration."""
        self.workspace_id: str = initial_config.get("workspace_id", "")
        self.user_count: int = initial_config.get("user_count", 0)
        self.user_map: Dict[str, str] = initial_config.get("user_map", {})
        self.messages_sent_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = initial_config.get("messages_sent_map", {})
        self.messages_inbox_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = initial_config.get("messages_inbox_map", {})
        self.message_count: int = initial_config.get("message_count", 0)
        self.current_user: str = initial_config.get("current_user", "")

    def add_contact(self, user_name: str) -> dict:
        """Add a contact to the workspace."""
        if not user_name:
            return {
                "added_status": False,
                "user_id": "",
                "message": "User name cannot be empty."
            }

        # Case-insensitive check
        existing_name = None
        for name in self.user_map:
            if name.lower() == user_name.lower():
                existing_name = name
                break

        if existing_name:
            return {
                "added_status": False,
                "user_id": self.user_map[existing_name],
                "message": f"User '{user_name}' already exists in the workspace."
            }

        existing_ids = {
            uid for uid in self.user_map.values()
        } | set(self.messages_sent_map.keys()) | set(self.messages_inbox_map.keys())
        max_num = 0
        for uid in existing_ids:
            try:
                num = int(uid.replace("USR", ""))
                if num > max_num:
                    max_num = num
            except (ValueError, AttributeError):
                pass

        new_num = max_num + 1
        new_user_id = f"USR{new_num:03d}"
        self.user_count += 1
        self.user_map[user_name] = new_user_id

        # Initialize message maps for the new user
        self.messages_sent_map[new_user_id] = {}
        self.messages_inbox_map[new_user_id] = {}

        return {
            "added_status": True,
            "user_id": new_user_id,
            "message": f"Contact '{user_name}' added successfully."
        }

    def delete_message(self, receiver_id: str, message_id: Optional[int] = None) -> dict:
        """Delete a message sent to a receiver by its message_id."""
        msg_id_int = int(message_id) if message_id is not None else 0

        if not self.current_user:
            return {
                "deleted_status": False,
                "message_id": "0",
                "message": "No user currently logged in."
            }

        if not receiver_id:
            return {
                "deleted_status": False,
                "message_id": "0",
                "message": "Receiver ID cannot be empty."
            }

        sender_id = self.current_user

        # Check if sender has sent any messages to the receiver
        if sender_id not in self.messages_sent_map or receiver_id not in self.messages_sent_map[sender_id]:
            return {
                "deleted_status": False,
                "message_id": "0",
                "message": "No messages found to delete."
            }

        sent_messages = self.messages_sent_map[sender_id][receiver_id]

        if not sent_messages:
            return {
                "deleted_status": False,
                "message_id": "0",
                "message": "No messages found to delete."
            }

        # Find the message dict with matching message_id
        found_idx = None
        for idx, msg_dict in enumerate(sent_messages):
            if msg_dict.get("message_id") == msg_id_int:
                found_idx = idx
                break

        if found_idx is None:
            return {
                "deleted_status": False,
                "message_id": "0",
                "message": f"Message with ID {message_id} not found."
            }

        sent_messages.pop(found_idx)

        # Remove the corresponding message from the receiver's inbox
        if receiver_id in self.messages_inbox_map and sender_id in self.messages_inbox_map[receiver_id]:
            inbox_messages = self.messages_inbox_map[receiver_id][sender_id]
            for idx, msg_dict in enumerate(inbox_messages):
                if msg_dict.get("message_id") == msg_id_int:
                    inbox_messages.pop(idx)
                    break

        return {
            "deleted_status": True,
            "message_id": str(msg_id_int),
            "message": "Message deleted successfully."
        }

    def get_user_id(self, user: str) -> dict:
        """Get user ID from user name."""
        if not user:
            return {
                "user_id": ""
            }

        # Case-insensitive lookup
        for name, uid in self.user_map.items():
            if name.lower() == user.lower():
                return {"user_id": uid}
        return {
            "user_id": ""
        }

    def message_login(self, user_id: str) -> dict:
        """Log in a user with the given user ID to message application."""
        if not user_id:
            return {
                "login_status": False,
                "message": "User ID cannot be empty."
            }

        user_exists = user_id in self.user_map.values() or user_id in self.messages_sent_map or user_id in self.messages_inbox_map

        if not user_exists:
            return {
                "login_status": False,
                "message": f"User ID '{user_id}' not found in the workspace."
            }

        self.current_user = user_id
        return {
            "login_status": True,
            "message": f"User '{user_id}' logged in successfully."
        }

    def search_messages(self, keyword: str) -> dict:
        """Search for messages containing a specific keyword."""
        results: List[Dict[str, Any]] = []

        if not keyword:
            return {
                "results": "[]"
            }

        if not self.current_user:
            return {
                "results": "[]"
            }

        sender_id = self.current_user

        # Search in sent messages
        if sender_id in self.messages_sent_map:
            for receiver_id, messages in self.messages_sent_map[sender_id].items():
                for msg in messages:
                    msg_text = msg.get("message", "") if isinstance(msg, dict) else str(msg)
                    if keyword.lower() in msg_text.lower():
                        results.append({
                            "receiver_id": receiver_id,
                            "message": msg_text,
                            "direction": "sent"
                        })

        # Search in received messages (inbox)
        if sender_id in self.messages_inbox_map:
            for sender, messages in self.messages_inbox_map[sender_id].items():
                for msg in messages:
                    msg_text = msg.get("message", "") if isinstance(msg, dict) else str(msg)
                    if keyword.lower() in msg_text.lower():
                        results.append({
                            "sender_id": sender,
                            "message": msg_text,
                            "direction": "received"
                        })

        return {
            "results": json.dumps(results)
        }

    def send_message(self, receiver_id: str, message: str) -> dict:
        """Send a message to a user."""
        if not self.current_user:
            return {
                "sent_status": False,
                "message_id": "0",
                "message": "No user currently logged in."
            }

        if not receiver_id:
            return {
                "sent_status": False,
                "message_id": "0",
                "message": "Receiver ID cannot be empty."
            }

        if not message:
            return {
                "sent_status": False,
                "message_id": "0",
                "message": "Message cannot be empty."
            }

        sender_id = self.current_user

        self.message_count += 1
        msg_id = self.message_count

        # Create message dict with ID (both stored as int for lookup, returned as string per schema)
        msg_dict = {"message_id": msg_id, "message": message}

        # Initialize sender's sent map if needed
        if sender_id not in self.messages_sent_map:
            self.messages_sent_map[sender_id] = {}

        # Initialize receiver's entry in sender's sent map if needed
        if receiver_id not in self.messages_sent_map[sender_id]:
            self.messages_sent_map[sender_id][receiver_id] = []

        # Add message dict to sender's sent map
        self.messages_sent_map[sender_id][receiver_id].append(msg_dict)

        # Initialize receiver's inbox map if needed
        if receiver_id not in self.messages_inbox_map:
            self.messages_inbox_map[receiver_id] = {}

        # Initialize sender's entry in receiver's inbox map if needed
        if sender_id not in self.messages_inbox_map[receiver_id]:
            self.messages_inbox_map[receiver_id][sender_id] = []

        # Add message dict to receiver's inbox map
        self.messages_inbox_map[receiver_id][sender_id].append(msg_dict)

        return {
            "sent_status": True,
            "message_id": str(msg_id),
            "message": "Message sent successfully."
        }