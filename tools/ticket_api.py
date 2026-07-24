"""Stateful support-ticket API."""

from typing import TypedDict

from .type_utils import (
    Config,
    Record,
    get_bool,
    get_int,
    get_str,
    is_object_dict,
    is_record_list,
)


class CreatedTicketResult(TypedDict):
    """Public representation of a newly created support ticket."""

    id: int
    title: str
    description: str
    status: str
    priority: int


class TicketResult(CreatedTicketResult):
    """Stored support-ticket details."""

    created_by: str


class TicketStatusResult(TypedDict):
    """Result from changing a support ticket."""

    status: str


class TicketLoginResult(TypedDict):
    """Result from authenticating with the ticket system."""

    success: bool


def _configured_queue(config: Config) -> list[Record]:
    for key in ("tickets_queue", "ticket_list", "support_tickets", "ticket_queue"):
        value = config.get(key)
        if is_record_list(value):
            return value
    return []


def _empty_created_ticket(status: str = "") -> CreatedTicketResult:
    return {
        "id": 0,
        "title": "",
        "description": "",
        "status": status,
        "priority": 0,
    }


def _empty_ticket(status: str = "") -> TicketResult:
    return {**_empty_created_ticket(status), "created_by": ""}


def _ticket_result(ticket: Config) -> TicketResult:
    return {
        "id": get_int(ticket, "id"),
        "title": get_str(ticket, "title"),
        "description": get_str(ticket, "description"),
        "status": get_str(ticket, "status"),
        "priority": get_int(ticket, "priority"),
        "created_by": get_str(ticket, "created_by"),
    }


class TicketAPI:
    """Create, retrieve, and manage support tickets."""

    def __init__(self, initial_config: Config) -> None:
        """Initialize and normalize ticket-system state."""
        self.ticket_queue: list[Record] = _configured_queue(initial_config)
        self.ticket_counter: int = get_int(
            initial_config,
            "ticket_count",
            get_int(initial_config, "ticket_counter"),
        )
        self.current_user: str = get_str(initial_config, "current_user")
        self.authenticated: bool = get_bool(
            initial_config,
            "authenticated",
            bool(self.current_user),
        )
        self.username: str = get_str(initial_config, "username")
        self.password: str = get_str(initial_config, "password")
        current_ticket_id = initial_config.get("current_ticket_id")
        self.current_ticket_id: int | None = (
            current_ticket_id if isinstance(current_ticket_id, int) else None
        )
        configured_priorities = initial_config.get("priority_levels")
        self.priority_levels: dict[object, object] = (
            configured_priorities
            if is_object_dict(configured_priorities)
            else {
                1: "Low",
                2: "Medium",
                3: "High",
                4: "Urgent",
                5: "Critical",
            }
        )

    def close_ticket(self, ticket_id: int) -> TicketStatusResult:
        """Close a ticket."""
        if not self.authenticated:
            return {"status": "User not authenticated"}
        for ticket in self.ticket_queue:
            if get_int(ticket, "id") == ticket_id:
                ticket["status"] = "Closed"
                return {"status": "Ticket closed successfully"}
        return {"status": f"Ticket with ID {ticket_id} not found"}

    def create_ticket(
        self, title: str, description: str = "", priority: int = 1
    ) -> CreatedTicketResult:
        """Create and enqueue a support ticket."""
        if not self.authenticated:
            return _empty_created_ticket("User not authenticated")
        self.ticket_counter += 1
        priority = min(5, max(1, priority))
        ticket: Record = {
            "id": self.ticket_counter,
            "title": title,
            "description": description,
            "status": "Open",
            "priority": priority,
            "created_by": self.current_user,
        }
        self.ticket_queue.append(ticket)
        return {
            "id": self.ticket_counter,
            "title": title,
            "description": description,
            "status": "Open",
            "priority": priority,
        }

    def edit_ticket(
        self, ticket_id: int, updates: Config
    ) -> TicketStatusResult:
        """Modify fields on an existing ticket."""
        if not self.authenticated:
            return {"status": "User not authenticated"}
        for ticket in self.ticket_queue:
            if get_int(ticket, "id") == ticket_id:
                for key, value in updates.items():
                    if key in ticket:
                        ticket[key] = value
                return {"status": "Ticket updated successfully"}
        return {"status": f"Ticket with ID {ticket_id} not found"}

    def get_ticket(self, ticket_id: int) -> TicketResult:
        """Get a specific ticket by its ID."""
        for ticket in self.ticket_queue:
            if get_int(ticket, "id") == ticket_id:
                return _ticket_result(ticket)
        return _empty_ticket("Not Found")

    def get_user_tickets(self, status: str = "None") -> TicketResult:
        """Get the first current-user ticket with an optional status filter."""
        for ticket in self.ticket_queue:
            if get_str(ticket, "created_by") != self.current_user:
                continue
            if status == "None" or get_str(ticket, "status") == status:
                return _ticket_result(ticket)
        return _empty_ticket()

    def resolve_ticket(
        self, ticket_id: int, resolution: str
    ) -> TicketStatusResult:
        """Resolve a ticket with resolution details."""
        if not self.authenticated:
            return {"status": "User not authenticated"}
        for ticket in self.ticket_queue:
            if get_int(ticket, "id") == ticket_id:
                ticket["status"] = "Resolved"
                ticket["resolution"] = resolution
                return {"status": "Ticket resolved successfully"}
        return {"status": f"Ticket with ID {ticket_id} not found"}

    def ticket_login(self, username: str, password: str) -> TicketLoginResult:
        """Authenticate a user for the ticket system."""
        if not username or not password:
            return {"success": False}
        if self.username and username != self.username:
            return {"success": False}
        if self.password and password != self.password:
            return {"success": False}
        self.authenticated = True
        self.current_user = username
        return {"success": True}
