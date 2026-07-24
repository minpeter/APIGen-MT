"""TravelBooking compatibility facade."""

from .travel_booking_auth import TravelBookingAuthMixin
from .travel_booking_information import TravelBookingInformationMixin
from .travel_booking_reservations import TravelBookingReservationsMixin
from .type_utils import Config


class TravelBooking(
    TravelBookingAuthMixin,
    TravelBookingReservationsMixin,
    TravelBookingInformationMixin,
):
    """Travel booking system for managing flights, credit cards, and budgets."""

    def __init__(self, initial_config: Config) -> None:
        """Initialize the travel booking system with the given configuration."""
        super().__init__(initial_config)
