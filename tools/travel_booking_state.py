"""Shared initialized state for TravelBooking mixins."""

from .type_utils import (
    Config,
    Record,
    get_float,
    get_int,
    get_record_map,
    get_str,
)


class TravelBookingState:
    """Initialize the mutable state consumed by travel operations."""

    def __init__(self, initial_config: Config) -> None:
        self.credit_card_list: dict[str, Record] = get_record_map(
            initial_config, "credit_card_list"
        )
        self.booking_record: dict[str, Record] = get_record_map(
            initial_config, "booking_record"
        )
        self.access_token: str = get_str(initial_config, "access_token")
        self.token_type: str = get_str(
            initial_config, "token_type", "Bearer"
        )
        self.token_expires_in: int = get_int(
            initial_config, "token_expires_in"
        )
        self.token_scope: str = get_str(initial_config, "token_scope")
        self.user_first_name: str = get_str(
            initial_config, "user_first_name"
        )
        self.user_last_name: str = get_str(
            initial_config, "user_last_name"
        )
        self.budget_limit: float = get_float(initial_config, "budget_limit")
        self.client_id: str = get_str(initial_config, "client_id")
        self.client_secret: str = get_str(initial_config, "client_secret")
        self.refresh_token: str = get_str(initial_config, "refresh_token")
