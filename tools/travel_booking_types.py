"""Typed public results returned by TravelBooking operations."""

from typing import NotRequired, TypedDict


class TravelAuthenticationResult(TypedDict):
    """Result from travel-service authentication."""

    expires_in: int
    access_token: str
    token_type: str
    scope: str
    success: NotRequired[bool]
    error: NotRequired[str]


class CardBalanceResult(TypedDict):
    """Available credit-card balance."""

    card_balance: float


class CardRegistrationResult(TypedDict):
    """Result from registering a credit card."""

    card_id: str


class BudgetLimitResult(TypedDict):
    """Result from setting a travel budget."""

    budget_limit: float


class TravelerVerificationResult(TypedDict, total=False):
    """Result from verifying traveler information."""

    verification_status: bool
    verification_failure: str
    verification_message: str


class ExchangeRateResult(TypedDict):
    """Converted currency value."""

    exchanged_value: float


class CustomerSupportResult(TypedDict):
    """Acknowledgement from travel customer support."""

    customer_support_message: str


class FiscalYearResult(TypedDict):
    """Current travel-budget fiscal year."""

    budget_fiscal_year: str


class FlightCostResult(TypedDict, total=False):
    """Available costs for a requested flight."""

    error: str
    travel_cost_list: list[float]
    travel_from: str
    travel_to: str
    travel_date: str
    travel_class: str
    currency: str


class AirportResult(TypedDict):
    """Nearest airport code for a city."""

    nearest_airport: str


class BookingHistory(TypedDict, total=False):
    """Details captured for a newly booked flight."""

    booking_id: str
    transaction_id: str
    travel_date: str
    travel_from: str
    travel_to: str
    travel_class: str
    travel_cost: float


class BookingResult(TypedDict):
    """Result from booking a flight."""

    booking_id: str
    transaction_id: str
    booking_status: bool
    booking_history: BookingHistory


class CancellationResult(TypedDict):
    """Result from cancelling a booking."""

    cancel_status: bool


class InsuranceResult(TypedDict):
    """Result from purchasing travel insurance."""

    insurance_id: str
    insurance_status: bool


class Invoice(TypedDict, total=False):
    """Invoice fields for a flight booking."""

    booking_id: str
    travel_date: str
    travel_from: str
    travel_to: str
    travel_class: str
    travel_cost: float
    transaction_id: str


class InvoiceResult(TypedDict):
    """Result from retrieving a travel invoice."""

    invoice: Invoice
