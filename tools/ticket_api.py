"""Auto-generated TicketAPI implementation."""

import json
import math
import re
import copy
import datetime
from typing import List, Dict, Any, Optional, Tuple


class TicketAPI:
    """A ticketing system that allows users to create, view, and manage support business tickets."""

    def __init__(self, initial_config: dict) -> None:
        """Initialize the TicketAPI with the given configuration, normalizing various config keys."""
        # Normalize ticket queue from various possible keys
        self.ticket_queue: List[Dict[str, Any]] = initial_config.get(
            "tickets_queue",
            initial_config.get(
                "ticket_list",
                initial_config.get("support_tickets", initial_config.get("ticket_queue", []))
            )
        )
        
        # Normalize ticket counter from various possible keys
        self.ticket_counter: int = initial_config.get(
            "ticket_count",
            initial_config.get("ticket_counter", 0)
        )
        
        # Normalize current user
        self.current_user: str = initial_config.get("current_user", "")
        
        # Normalize current ticket id
        self.current_ticket_id: Optional[int] = initial_config.get("current_ticket_id", None)
        
        # Normalize priority levels
        self.priority_levels: Dict[int, str] = initial_config.get(
            "priority_levels",
            {1: "Low", 2: "Medium", 3: "High", 4: "Urgent", 5: "Critical"}
        )

    def close_ticket(self, ticket_id: int) -> Dict[str, Any]:
        """Close a ticket.
        
        Args:
            ticket_id: ID of the ticket to be closed.
            
        Returns:
            A dict containing the status of the close operation.
        """
        for ticket in self.ticket_queue:
            if ticket.get("id") == ticket_id:
                ticket["status"] = "Closed"
                return {"status": "Ticket closed successfully"}
        return {"status": f"Ticket with ID {ticket_id} not found"}

    def create_ticket(self, title: str, description: str = '', priority: int = 1) -> Dict[str, Any]:
        """Create a ticket in the system and queue it.
        
        Args:
            title: Title of the ticket.
            description: Description of the ticket. Defaults to an empty string.
            priority: Priority of the ticket, from 1 to 5. Defaults to 1.
            
        Returns:
            A dict containing the details of the created ticket.
        """
        self.ticket_counter += 1
        new_ticket_id = self.ticket_counter
        
        # Ensure priority is within valid range
        if priority < 1:
            priority = 1
        elif priority > 5:
            priority = 5
            
        new_ticket: Dict[str, Any] = {
            "id": new_ticket_id,
            "title": title,
            "description": description,
            "status": "Open",
            "priority": priority,
            "created_by": self.current_user
        }
        
        self.ticket_queue.append(new_ticket)
        
        return {
            "id": new_ticket_id,
            "title": title,
            "description": description,
            "status": "Open",
            "priority": priority
        }

    def edit_ticket(self, ticket_id: int, updates: dict) -> Dict[str, Any]:
        """Modify the details of an existing ticket.
        
        Args:
            ticket_id: ID of the ticket to be changed.
            updates: Dictionary containing the fields to be updated.
            
        Returns:
            A dict containing the status of the update operation.
        """
        for ticket in self.ticket_queue:
            if ticket.get("id") == ticket_id:
                for key, value in updates.items():
                    if key in ticket:
                        ticket[key] = value
                return {"status": "Ticket updated successfully"}
        return {"status": f"Ticket with ID {ticket_id} not found"}

    def get_ticket(self, ticket_id: int) -> Dict[str, Any]:
        """Get a specific ticket by its ID.
        
        Args:
            ticket_id: ID of the ticket to retrieve.
            
        Returns:
            A dict containing the details of the ticket.
        """
        for ticket in self.ticket_queue:
            if ticket.get("id") == ticket_id:
                return {
                    "id": ticket.get("id", 0),
                    "title": ticket.get("title", ""),
                    "description": ticket.get("description", ""),
                    "status": ticket.get("status", ""),
                    "priority": ticket.get("priority", 0),
                    "created_by": ticket.get("created_by", "")
                }
        return {
            "id": 0,
            "title": "",
            "description": "",
            "status": "Not Found",
            "priority": 0,
            "created_by": ""
        }

    def get_user_tickets(self, status: str = 'None') -> Dict[str, Any]:
        """Get all tickets created by the current user, optionally filtered by status.
        
        Args:
            status: Status to filter tickets by. If None, return all tickets.
            
        Returns:
            A dict containing the details of the user's tickets.
        """
        user_tickets = [
            ticket for ticket in self.ticket_queue
            if ticket.get("created_by", "") == self.current_user
        ]
        
        if status != 'None':
            user_tickets = [
                ticket for ticket in user_tickets
                if ticket.get("status", "") == status
            ]
            
        if not user_tickets:
            return {
                "id": 0,
                "title": "",
                "description": "",
                "status": "",
                "priority": 0,
                "created_by": ""
            }
        
        # Return the first matching ticket as per the single-ticket response schema
        first_ticket = user_tickets[0]
        return {
            "id": first_ticket.get("id", 0),
            "title": first_ticket.get("title", ""),
            "description": first_ticket.get("description", ""),
            "status": first_ticket.get("status", ""),
            "priority": first_ticket.get("priority", 0),
            "created_by": first_ticket.get("created_by", "")
        }

    def resolve_ticket(self, ticket_id: int, resolution: str) -> Dict[str, Any]:
        """Resolve a ticket with a resolution.
        
        Args:
            ticket_id: ID of the ticket to be resolved.
            resolution: Resolution details for the ticket.
            
        Returns:
            A dict containing the status of the resolve operation.
        """
        for ticket in self.ticket_queue:
            if ticket.get("id") == ticket_id:
                ticket["status"] = "Resolved"
                ticket["resolution"] = resolution
                return {"status": "Ticket resolved successfully"}
        return {"status": f"Ticket with ID {ticket_id} not found"}

    def ticket_login(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate a user for ticket system.
        
        Args:
            username: Username of the user.
            password: Password of the user.
            
        Returns:
            A dict indicating whether the login was successful.
        """
        # Basic authentication simulation
        if username and password:
            self.current_user = username
            return {"success": True}
        return {"success": False}