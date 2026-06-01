"""Auto-generated TravelBooking implementation."""

import json
import math
import re
import copy
import datetime
from typing import List, Dict, Any, Optional, Tuple


class TravelBooking:
    """Travel booking system for managing flights, credit cards, and budgets."""

    def __init__(self, initial_config: dict) -> None:
        """Initialize the travel booking system with the given configuration."""
        self.credit_card_list: Dict[str, Dict[str, Any]] = initial_config.get("credit_card_list", {})
        self.booking_record: Dict[str, Dict[str, Any]] = initial_config.get("booking_record", {})
        self.access_token: str = initial_config.get("access_token", "")
        self.token_type: str = initial_config.get("token_type", "Bearer")
        self.token_expires_in: int = initial_config.get("token_expires_in", 0)
        self.token_scope: str = initial_config.get("token_scope", "")
        self.user_first_name: str = initial_config.get("user_first_name", "")
        self.user_last_name: str = initial_config.get("user_last_name", "")
        self.budget_limit: float = initial_config.get("budget_limit", 0.0)
        self.client_id: str = initial_config.get("client_id", "")
        self.client_secret: str = initial_config.get("client_secret", "")
        self.refresh_token: str = initial_config.get("refresh_token", "")

    def authenticate_travel(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        grant_type: str,
        user_first_name: str,
        user_last_name: str,
    ) -> Dict[str, Any]:
        """Authenticate the user with the travel API."""
        valid_grant_types = ["read_write", "read", "write"]
        if grant_type not in valid_grant_types:
            return {
                "expires_in": 0,
                "access_token": "",
                "token_type": "",
                "scope": "",
            }

        if (client_id != self.client_id or
            client_secret != self.client_secret or
            refresh_token != self.refresh_token):
            return {
                "expires_in": 0,
                "access_token": "",
                "token_type": "",
                "scope": "",
            }
        
        self.access_token = f"token_{client_id}_{refresh_token}"
        self.token_type = "Bearer"
        self.token_expires_in = 3600
        self.token_scope = grant_type
        self.user_first_name = user_first_name
        self.user_last_name = user_last_name

        return {
            "expires_in": self.token_expires_in,
            "access_token": self.access_token,
            "token_type": self.token_type,
            "scope": self.token_scope,
        }

    def book_flight(
        self,
        access_token: str,
        card_id: str,
        travel_date: str,
        travel_from: str,
        travel_to: str,
        travel_class: str,
        travel_cost: float,
    ) -> Dict[str, Any]:
        """Book a flight given the travel information."""
        if access_token != self.access_token:
            return {
                "booking_id": "",
                "transaction_id": "",
                "booking_status": False,
                "booking_history": {},
            }

        if card_id not in self.credit_card_list:
            return {
                "booking_id": "",
                "transaction_id": "",
                "booking_status": False,
                "booking_history": {},
            }

        card = self.credit_card_list[card_id]
        if card.get("balance", 0.0) < travel_cost:
            return {
                "booking_id": "",
                "transaction_id": "",
                "booking_status": False,
                "booking_history": {},
            }

        card["balance"] = card.get("balance", 0.0) - travel_cost

        booking_id = f"flight_{len(self.booking_record) + 1:03d}"
        transaction_id = f"txn_{card_id}_{booking_id}"

        booking_entry = {
            "travel_to": travel_to,
            "travel_from": travel_from,
            "insurance": "none",
            "travel_cost": travel_cost,
            "travel_date": travel_date,
            "travel_class": travel_class,
            "transaction_id": transaction_id,
            "card_id": card_id,
        }

        self.booking_record[booking_id] = booking_entry

        return {
            "booking_id": booking_id,
            "transaction_id": transaction_id,
            "booking_status": True,
            "booking_history": {
                "booking_id": booking_id,
                "transaction_id": transaction_id,
                "travel_date": travel_date,
                "travel_from": travel_from,
                "travel_to": travel_to,
                "travel_class": travel_class,
                "travel_cost": travel_cost,
            },
        }

    def cancel_booking(self, access_token: str, booking_id: str) -> Dict[str, Any]:
        """Cancel a booking."""
        if access_token != self.access_token:
            return {"cancel_status": False}

        if booking_id not in self.booking_record:
            return {"cancel_status": False}

        booking = self.booking_record[booking_id]
        card_id = booking.get("card_id", "")
        travel_cost = booking.get("travel_cost", 0.0)

        if card_id in self.credit_card_list:
            self.credit_card_list[card_id]["balance"] = (
                self.credit_card_list[card_id].get("balance", 0.0) + travel_cost
            )

        del self.booking_record[booking_id]

        return {"cancel_status": True}

    def compute_exchange_rate(
        self, base_currency: str, target_currency: str, value: float
    ) -> Dict[str, Any]:
        """Compute the exchange rate between two currencies."""
        valid_currencies = [
            "USD", "RMB", "EUR", "JPY", "GBP", "CAD", "AUD", "INR", "RUB", "BRL", "MXN"
        ]
        if base_currency not in valid_currencies or target_currency not in valid_currencies:
            return {"exchanged_value": 0.0}

        exchange_rates_to_usd = {
            "USD": 1.0,
            "RMB": 0.14,
            "EUR": 1.08,
            "JPY": 0.0067,
            "GBP": 1.27,
            "CAD": 0.74,
            "AUD": 0.65,
            "INR": 0.012,
            "RUB": 0.011,
            "BRL": 0.20,
            "MXN": 0.058,
        }

        value_in_usd = value * exchange_rates_to_usd.get(base_currency, 0.0)
        if target_currency == "USD":
            exchanged_value = value_in_usd
        else:
            target_rate = exchange_rates_to_usd.get(target_currency, 0.0)
            if target_rate == 0.0:
                return {"exchanged_value": 0.0}
            exchanged_value = value_in_usd / target_rate

        return {"exchanged_value": exchanged_value}

    def contact_customer_support(
        self, booking_id: str, message: str
    ) -> Dict[str, Any]:
        """Contact travel booking customer support."""
        if booking_id not in self.booking_record:
            return {
                "customer_support_message": f"No booking found for ID {booking_id}. Unable to process request."
            }
        return {
            "customer_support_message": f"Customer support has received your message regarding booking {booking_id}: '{message}'. A representative will reach out shortly."
        }

    def get_budget_fiscal_year(
        self, lastModifiedAfter: str = "None", includeRemoved: str = "None"
    ) -> Dict[str, Any]:
        """Get the budget fiscal year."""
        current_year = datetime.datetime.now().year
        return {"budget_fiscal_year": f"{current_year}-{current_year + 1}"}

    def get_credit_card_balance(
        self, access_token: str, card_id: str
    ) -> Dict[str, Any]:
        """Get the balance of a credit card."""
        if access_token != self.access_token:
            return {"card_balance": 0.0}

        if card_id not in self.credit_card_list:
            return {"card_balance": 0.0}

        return {"card_balance": self.credit_card_list[card_id].get("balance", 0.0)}

    def get_flight_cost(
        self,
        travel_from: str,
        travel_to: str,
        travel_date: str,
        travel_class: str,
    ) -> Dict[str, Any]:
        """Get the list of cost of a flight in USD based on location, date, and class."""
        base_costs = {
            "economy": 300.0,
            "business": 800.0,
            "first": 1500.0,
        }
        
        travel_class_lower = travel_class.lower()
        if travel_class_lower not in base_costs:
            return {"travel_cost_list": []}

        base_cost = base_costs[travel_class_lower]
        
        try:
            date_obj = datetime.datetime.strptime(travel_date, "%Y-%m-%d")
            day_of_year = date_obj.timetuple().tm_yday
            # Generate deterministic pseudo-random variation based on route and date
            variation = math.sin(day_of_year * 0.1 + len(travel_from) * 0.5 + len(travel_to) * 0.3) * 50
            cost = base_cost + variation
            return {"travel_cost_list": [round(cost, 2), round(cost * 1.1, 2), round(cost * 0.9, 2)]}
        except ValueError:
            return {"travel_cost_list": []}

    def get_nearest_airport_by_city(self, location: str) -> Dict[str, Any]:
        """Get the nearest airport to the given location."""
        city_to_airport = {
            "Rivermist": "RVM",
            "Stonebrook": "STB",
            "Maplecrest": "MPC",
            "Silverpine": "SLP",
            "Shadowridge": "SHR",
            "London": "LHR",
            "Paris": "CDG",
            "Sunset Valley": "SSV",
            "Oakendale": "OKD",
            "Willowbend": "WLB",
            "Crescent Hollow": "CSH",
            "Autumnville": "ATV",
            "Pinehaven": "PNH",
            "Greenfield": "GRF",
            "San Francisco": "SFO",
            "Los Angeles": "LAX",
            "New York": "JFK",
            "Chicago": "ORD",
            "Boston": "BOS",
            "Beijing": "PEK",
            "Hong Kong": "HKG",
            "Rome": "FCO",
            "Tokyo": "NRT",
        }
        nearest_airport = city_to_airport.get(location, "")
        return {"nearest_airport": nearest_airport}

    def purchase_insurance(
        self,
        access_token: str,
        insurance_type: str,
        insurance_cost: float,
        booking_id: str,
        card_id: str,
    ) -> Dict[str, Any]:
        """Purchase insurance."""
        if access_token != self.access_token:
            return {"insurance_id": "", "insurance_status": False}

        if booking_id not in self.booking_record:
            return {"insurance_id": "", "insurance_status": False}

        if card_id not in self.credit_card_list:
            return {"insurance_id": "", "insurance_status": False}

        card = self.credit_card_list[card_id]
        if card.get("balance", 0.0) < insurance_cost:
            return {"insurance_id": "", "insurance_status": False}

        card["balance"] = card.get("balance", 0.0) - insurance_cost
        self.booking_record[booking_id]["insurance"] = insurance_type

        insurance_id = f"ins_{booking_id}"
        return {"insurance_id": insurance_id, "insurance_status": True}

    def register_credit_card(
        self,
        access_token: str,
        card_number: str,
        expiration_date: str,
        cardholder_name: str,
        card_verification_number: int,
    ) -> Dict[str, Any]:
        """Register a credit card."""
        if access_token != self.access_token:
            return {"card_id": ""}

        card_id = str(len(self.credit_card_list) + 1)
        self.credit_card_list[card_id] = {
            "card_number": card_number,
            "expiration_date": expiration_date,
            "cardholder_name": cardholder_name,
            "card_verification_number": card_verification_number,
            "balance": 0.0,
        }

        return {"card_id": card_id}

    def retrieve_invoice(
        self, access_token: str, booking_id: str = "None", insurance_id: str = "None"
    ) -> Dict[str, Any]:
        """Retrieve the invoice for a booking."""
        if access_token != self.access_token:
            return {"invoice": {}}

        if booking_id == "None" or booking_id not in self.booking_record:
            return {"invoice": {}}

        booking = self.booking_record[booking_id]
        return {
            "invoice": {
                "booking_id": booking_id,
                "travel_date": booking.get("travel_date", ""),
                "travel_from": booking.get("travel_from", ""),
                "travel_to": booking.get("travel_to", ""),
                "travel_class": booking.get("travel_class", ""),
                "travel_cost": booking.get("travel_cost", 0.0),
                "transaction_id": booking.get("transaction_id", ""),
            }
        }

    def set_budget_limit(self, access_token: str, budget_limit: float) -> Dict[str, Any]:
        """Set the budget limit for the user."""
        if access_token != self.access_token:
            return {"budget_limit": 0.0}

        self.budget_limit = budget_limit
        return {"budget_limit": self.budget_limit}

    def verify_traveler_information(
        self,
        first_name: str,
        last_name: str,
        date_of_birth: str,
        passport_number: str,
    ) -> Dict[str, Any]:
        """Verify the traveler information."""
        if not first_name or not last_name:
            return {
                "verification_status": False,
                "verification_failure": "Name fields cannot be empty.",
            }

        try:
            datetime.datetime.strptime(date_of_birth, "%Y-%m-%d")
        except ValueError:
            return {
                "verification_status": False,
                "verification_failure": "Invalid date of birth format. Expected YYYY-MM-DD.",
            }

        if not passport_number:
            return {
                "verification_status": False,
                "verification_failure": "Passport number cannot be empty.",
            }

        return {
            "verification_status": True,
            "verification_failure": "",
        }