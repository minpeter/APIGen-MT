"""Authentication, card, budget, and traveler operations for TravelBooking."""

from datetime import date

from .travel_booking_state import TravelBookingState
from .travel_booking_types import (
    BudgetLimitResult,
    CardBalanceResult,
    CardRegistrationResult,
    TravelAuthenticationResult,
    TravelerVerificationResult,
)
from .type_utils import get_float


class TravelBookingAuthMixin(TravelBookingState):
    """Provide identity and account-related travel operations."""

    def authenticate_travel(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        grant_type: str,
        user_first_name: str,
        user_last_name: str,
    ) -> TravelAuthenticationResult:
        """Authenticate the user with the travel API."""
        valid_grant_types = ["read_write", "read", "write"]
        if grant_type not in valid_grant_types:
            return {
                "success": False,
                "error": (
                    f"Invalid grant_type '{grant_type}'. "
                    f"Must be one of {valid_grant_types}."
                ),
                "expires_in": 0,
                "access_token": "",
                "token_type": "",
                "scope": "",
            }
        has_configured_credentials = bool(
            self.client_id or self.client_secret or self.refresh_token
        )
        credentials_match = (
            client_id == self.client_id
            and client_secret == self.client_secret
            and refresh_token == self.refresh_token
        )
        if has_configured_credentials and not credentials_match:
            return {
                "success": False,
                "error": (
                    "Invalid credentials: client_id/secret/refresh_token do "
                    "not match. Use the credentials from the blueprint."
                ),
                "expires_in": 0,
                "access_token": "",
                "token_type": "",
                "scope": "",
            }

        self.access_token: str = f"token_{client_id}_{refresh_token}"
        self.token_type: str = "Bearer"
        self.token_expires_in: int = 3600
        self.token_scope: str = grant_type
        self.user_first_name: str = user_first_name
        self.user_last_name: str = user_last_name
        return {
            "expires_in": self.token_expires_in,
            "access_token": self.access_token,
            "token_type": self.token_type,
            "scope": self.token_scope,
        }

    def get_credit_card_balance(
        self, access_token: str, card_id: str
    ) -> CardBalanceResult:
        """Get the balance of a credit card."""
        if not access_token or access_token != self.access_token:
            return {"card_balance": 0.0}
        card = self.credit_card_list.get(card_id)
        return {
            "card_balance": get_float(card, "balance")
            if card is not None
            else 0.0
        }

    def register_credit_card(
        self,
        access_token: str,
        card_number: str,
        expiration_date: str,
        cardholder_name: str,
        card_verification_number: int,
    ) -> CardRegistrationResult:
        """Register a credit card."""
        if not access_token or access_token != self.access_token:
            return {"card_id": ""}
        card_id = f"card_{card_number[-4:]}"
        self.credit_card_list[card_id] = {
            "card_number": card_number,
            "expiration_date": expiration_date,
            "cardholder_name": cardholder_name,
            "card_verification_number": card_verification_number,
            "balance": 10000.0,
        }
        return {"card_id": card_id}

    def set_budget_limit(
        self, access_token: str, budget_limit: float
    ) -> BudgetLimitResult:
        """Set the budget limit for the user."""
        if not access_token or access_token != self.access_token:
            return {"budget_limit": 0.0}
        self.budget_limit: float = budget_limit
        return {"budget_limit": self.budget_limit}

    def verify_traveler_information(
        self,
        first_name: str,
        last_name: str,
        date_of_birth: str = "",
        passport_number: str = "",
    ) -> TravelerVerificationResult:
        """Verify traveler identity fields."""
        if not first_name or not last_name:
            return {
                "verification_status": False,
                "verification_failure": "Name fields cannot be empty.",
            }
        if not date_of_birth:
            return {
                "verification_status": True,
                "verification_message": (
                    "Traveler information verified (date of birth skipped)."
                ),
            }
        try:
            _ = date.fromisoformat(date_of_birth)
        except ValueError:
            return {
                "verification_status": False,
                "verification_failure": (
                    "Invalid date of birth format. Expected YYYY-MM-DD."
                ),
            }
        if not passport_number:
            return {
                "verification_status": False,
                "verification_failure": "Passport number cannot be empty.",
            }
        return {"verification_status": True, "verification_failure": ""}
