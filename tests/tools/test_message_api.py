import pytest
import json
from tools.message_api import MessageAPI


INITIAL_CONFIG = {
    "workspace_id": "WS123456",
    "user_count": 4,
    "user_map": {
        "Michael": "USR005",
        "Sarah": "USR006",
        "David": "USR007",
        "Emma": "USR008"
    },
    "messages_sent_map": {
        "USR005": {
            "USR006": [
                "Please review the attached document."
            ],
            "USR007": [
                "Meeting at 3 PM."
            ],
            "USR008": [
                "Lunch tomorrow?"
            ]
        },
        "USR006": {
            "USR005": [
                "Got it, thanks!"
            ],
            "USR007": [
                "Can we reschedule?"
            ]
        },
        "USR007": {
            "USR005": [
                "Sure, see you then."
            ],
            "USR008": [
                "Let's catch up soon."
            ]
        },
        "USR008": {
            "USR006": [
                "I'll be there."
            ],
            "USR007": [
                "Sounds good."
            ]
        }
    },
    "messages_inbox_map": {
        "USR005": {
            "USR006": [
                "Got it, thanks!"
            ],
            "USR007": [
                "Sure, see you then."
            ]
        },
        "USR006": {
            "USR005": [
                "Please review the attached document."
            ],
            "USR008": [
                "I'll be there."
            ]
        },
        "USR007": {
            "USR005": [
                "Meeting at 3 PM."
            ],
            "USR006": [
                "Can we reschedule?"
            ]
        },
        "USR008": {
            "USR005": [
                "Lunch tomorrow?"
            ],
            "USR007": [
                "Let's catch up soon."
            ]
        }
    },
    "message_count": 0,
    "current_user": "USR005"
}


@pytest.fixture
def message_api():
    config_copy = json.loads(json.dumps(INITIAL_CONFIG))
    return MessageAPI(initial_config=config_copy)


class TestAddContact:
    def test_add_new_contact_successfully(self, message_api):
        result = message_api.add_contact(user_name="John Levy")
        assert result.get("added_status") is True
        assert result.get("user_id") != ""
        lookup = message_api.get_user_id(user="John Levy")
        assert lookup.get("user_id") != ""

    def test_add_existing_contact(self, message_api):
        result = message_api.add_contact(user_name="Sarah")
        assert result.get("added_status") is False
        assert result.get("user_id") == "USR006"

    def test_add_contact_empty_name(self, message_api):
        result = message_api.add_contact(user_name="")
        assert result.get("added_status") is False
        assert "cannot be empty" in result.get("message", "").lower()


class TestDeleteMessage:
    def test_delete_latest_message_successfully(self, message_api):
        result = message_api.delete_message(receiver_id="USR006")
        assert result.get("deleted_status") is True

    def test_delete_message_nonexistent_receiver(self, message_api):
        result = message_api.delete_message(receiver_id="USR999")
        assert result.get("deleted_status") is False

    def test_delete_message_no_messages_sent(self, message_api):
        result = message_api.delete_message(receiver_id="USR005")
        assert result.get("deleted_status") is False
        assert "no messages" in result.get("message", "").lower()


class TestGetUserId:
    def test_get_existing_user_id(self, message_api):
        result = message_api.get_user_id(user="Michael")
        assert result.get("user_id") == "USR005"

    def test_get_nonexistent_user_id(self, message_api):
        result = message_api.get_user_id(user="Bob")
        assert result.get("user_id") == ""

    def test_get_user_id_empty_string(self, message_api):
        result = message_api.get_user_id(user="")
        assert result.get("user_id") == ""


class TestMessageLogin:
    def test_login_with_valid_user_id(self, message_api):
        result = message_api.message_login(user_id="USR006")
        assert result.get("login_status") is True

    def test_login_with_existing_workspace_user_id(self, message_api):
        result = message_api.message_login(user_id="USR005")
        assert result.get("login_status") is True
        assert message_api.current_user == "USR005"

    def test_login_with_empty_user_id(self, message_api):
        result = message_api.message_login(user_id="")
        assert result.get("login_status") is False

    def test_login_with_nonexistent_user_id(self, message_api):
        result = message_api.message_login(user_id="USR999")
        assert result.get("login_status") is False


class TestSearchMessages:
    def test_search_with_matching_keyword(self, message_api):
        result = message_api.search_messages(keyword="Meeting")
        assert len(result.get("results", [])) > 0

    def test_search_with_no_matching_keyword(self, message_api):
        result = message_api.search_messages(keyword="xylophone_unlikely_word")
        assert len(result.get("results", [])) == 0

    def test_search_with_empty_keyword(self, message_api):
        result = message_api.search_messages(keyword="")
        assert result.get("results") == []


class TestSendMessage:
    def test_send_message_to_valid_receiver(self, message_api):
        result = message_api.send_message(receiver_id="USR006", message="Latest Quarter Performance has been well.")
        assert result.get("sent_status") is True
        assert result.get("message_id") is not None

    def test_send_message_to_another_user(self, message_api):
        result = message_api.send_message(receiver_id="USR007", message="Kelly Total Score: 96")
        assert result.get("sent_status") is True

    def test_send_message_no_user_logged_in(self):
        config = json.loads(json.dumps(INITIAL_CONFIG))
        config["current_user"] = ""
        api = MessageAPI(initial_config=config)
        result = api.send_message(receiver_id="USR006", message="Hello?")
        assert result.get("sent_status") is False

    def test_send_empty_message(self, message_api):
        result = message_api.send_message(receiver_id="USR006", message="")
        assert result.get("sent_status") is False
