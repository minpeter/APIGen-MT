import pytest
import json
from tools.ticket_api import TicketAPI


@pytest.fixture
def ticket_api():
    initial_config = {
        "tickets_queue": [
            {
                "id": 123456,
                "title": "System Error",
                "description": "There is a critical system error that needs immediate attention.",
                "status": "Open",
                "priority": 3,
                "created_by": "tech_guru"
            },
            {
                "id": 654321,
                "title": "Feature Request",
                "description": "Request for a new feature in the application.",
                "status": "In Progress",
                "priority": 2,
                "created_by": "tech_guru"
            },
        ],
        "ticket_count": 654321,
        "current_user": "tech_guru"
    }
    return TicketAPI(initial_config)


def test_close_ticket_normal(ticket_api):
    result = ticket_api.close_ticket(ticket_id=123456)
    assert result.get("status") == "Ticket closed successfully"
    assert ticket_api.ticket_queue[0].get("status") == "Closed"


def test_close_ticket_not_found(ticket_api):
    result = ticket_api.close_ticket(ticket_id=987654)
    assert "not found" in result.get("status", "").lower()


def test_close_ticket_already_closed(ticket_api):
    ticket_api.close_ticket(ticket_id=123456)
    result = ticket_api.close_ticket(ticket_id=123456)
    assert result.get("status") == "Ticket closed successfully"


def test_create_ticket_normal(ticket_api):
    result = ticket_api.create_ticket(
        title="emergency", description="Initial project plan details.", priority=3
    )
    assert result.get("title") == "emergency"
    assert result.get("description") == "Initial project plan details."
    assert result.get("priority") == 3
    assert result.get("status") == "Open"


def test_create_ticket_defaults(ticket_api):
    result = ticket_api.create_ticket(title="Tire Pressure Issue")
    assert result.get("title") == "Tire Pressure Issue"
    assert result.get("description") == ""
    assert result.get("priority") == 1
    assert result.get("status") == "Open"


def test_create_ticket_clamp_priority_high(ticket_api):
    result = ticket_api.create_ticket(title="Bad Priority", priority=10)
    assert result.get("priority") == 5


def test_create_ticket_clamp_priority_low(ticket_api):
    result = ticket_api.create_ticket(title="Low Priority", priority=-1)
    assert result.get("priority") == 1


def test_edit_ticket_normal(ticket_api):
    result = ticket_api.edit_ticket(ticket_id=654321, updates={"priority": 4})
    assert result.get("status") == "Ticket updated successfully"
    assert ticket_api.ticket_queue[1].get("priority") == 4


def test_edit_ticket_not_found(ticket_api):
    result = ticket_api.edit_ticket(ticket_id=0, updates={"status": "Urgent"})
    assert "not found" in result.get("status", "").lower()


def test_edit_ticket_invalid_field(ticket_api):
    result = ticket_api.edit_ticket(ticket_id=123456, updates={"nonexistent_field": "value"})
    assert result.get("status") == "Ticket updated successfully"


def test_get_ticket_normal(ticket_api):
    result = ticket_api.get_ticket(ticket_id=123456)
    assert result.get("id") == 123456
    assert result.get("title") == "System Error"
    assert result.get("status") == "Open"


def test_get_ticket_not_found(ticket_api):
    result = ticket_api.get_ticket(ticket_id=987654)
    assert result.get("status") == "Not Found"
    assert result.get("id") == 0


def test_get_ticket_invalid_id(ticket_api):
    result = ticket_api.get_ticket(ticket_id=-1)
    assert result.get("status") == "Not Found"
    assert result.get("id") == 0


def test_get_user_tickets_all(ticket_api):
    result = ticket_api.get_user_tickets()
    assert result.get("id") is not None
    assert result.get("created_by") == "tech_guru"


def test_get_user_tickets_by_status(ticket_api):
    result = ticket_api.get_user_tickets(status="Open")
    assert result.get("status") == "Open"
    assert result.get("created_by") == "tech_guru"


def test_get_user_tickets_empty_status(ticket_api):
    result = ticket_api.get_user_tickets(status="Closed")
    assert result.get("id") == 0
    assert result.get("title") == ""


def test_resolve_ticket_normal(ticket_api):
    result = ticket_api.resolve_ticket(
        ticket_id=123456, resolution="Fixed through manual troubleshooting techniques."
    )
    assert result.get("status") == "Ticket resolved successfully"
    assert ticket_api.ticket_queue[0].get("status") == "Resolved"
    assert ticket_api.ticket_queue[0].get("resolution") == "Fixed through manual troubleshooting techniques."


def test_resolve_ticket_not_found(ticket_api):
    result = ticket_api.resolve_ticket(ticket_id=7423, resolution="")
    assert "not found" in result.get("status", "").lower()


def test_resolve_ticket_already_resolved(ticket_api):
    ticket_api.resolve_ticket(ticket_id=123456, resolution="First resolution")
    result = ticket_api.resolve_ticket(ticket_id=123456, resolution="Second resolution")
    assert result.get("status") == "Ticket resolved successfully"


def test_ticket_login_normal(ticket_api):
    result = ticket_api.ticket_login(username="tech_guru", password="testpass")
    assert result.get("success") is True
    assert ticket_api.current_user == "tech_guru"


def test_ticket_login_invalid_credentials(ticket_api):
    result = ticket_api.ticket_login(username="user123", password="12345")
    assert result.get("success") is True


def test_ticket_login_missing_fields(ticket_api):
    result = ticket_api.ticket_login(username="", password="")
    assert result.get("success") is False
