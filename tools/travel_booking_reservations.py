"""Reservation and insurance operations for TravelBooking."""

from .travel_booking_state import TravelBookingState
from .travel_booking_types import (
    BookingResult,
    CancellationResult,
    InsuranceResult,
    InvoiceResult,
)
from .type_utils import Record, get_float, get_str


class TravelBookingReservationsMixin(TravelBookingState):
    """Provide stateful booking, cancellation, insurance, and invoice operations."""

    def book_flight(
        self,
        access_token: str,
        card_id: str,
        travel_date: str,
        travel_from: str,
        travel_to: str,
        travel_class: str,
        travel_cost: float,
    ) -> BookingResult:
        """Book a flight using a registered credit card."""
        empty_result: BookingResult = {
            "booking_id": "",
            "transaction_id": "",
            "booking_status": False,
            "booking_history": {},
        }
        if not access_token or access_token != self.access_token:
            return empty_result
        card = self.credit_card_list.get(card_id)
        if card is None:
            return empty_result
        balance = get_float(card, "balance")
        if balance < travel_cost:
            return empty_result
        card["balance"] = balance - travel_cost

        booking_id = f"flight_{len(self.booking_record) + 1:03d}"
        transaction_id = f"txn_{card_id}_{booking_id}"
        booking: Record = {
            "travel_to": travel_to,
            "travel_from": travel_from,
            "insurance": "none",
            "travel_cost": travel_cost,
            "travel_date": travel_date,
            "travel_class": travel_class,
            "transaction_id": transaction_id,
            "card_id": card_id,
        }
        self.booking_record[booking_id] = booking
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

    def cancel_booking(
        self, access_token: str, booking_id: str
    ) -> CancellationResult:
        """Cancel a booking and refund its flight cost."""
        if not access_token or access_token != self.access_token:
            return {"cancel_status": False}
        booking = self.booking_record.get(booking_id)
        if booking is None:
            return {"cancel_status": False}
        card = self.credit_card_list.get(get_str(booking, "card_id"))
        if card is not None:
            card["balance"] = get_float(card, "balance") + get_float(
                booking, "travel_cost"
            )
        del self.booking_record[booking_id]
        return {"cancel_status": True}

    def purchase_insurance(
        self,
        access_token: str,
        insurance_type: str,
        insurance_cost: float,
        booking_id: str,
        card_id: str,
    ) -> InsuranceResult:
        """Purchase insurance for an existing flight booking."""
        failure: InsuranceResult = {
            "insurance_id": "",
            "insurance_status": False,
        }
        if not access_token or access_token != self.access_token:
            return failure
        booking = self.booking_record.get(booking_id)
        card = self.credit_card_list.get(card_id)
        if booking is None or card is None:
            return failure
        balance = get_float(card, "balance")
        if balance < insurance_cost:
            return failure
        card["balance"] = balance - insurance_cost
        booking["insurance"] = insurance_type
        return {
            "insurance_id": f"ins_{booking_id}",
            "insurance_status": True,
        }

    def retrieve_invoice(
        self,
        access_token: str,
        booking_id: str = "None",
        insurance_id: str = "None",
    ) -> InvoiceResult:
        """Retrieve the invoice for a booking."""
        _ = insurance_id
        if not access_token or access_token != self.access_token:
            return {"invoice": {}}
        booking = self.booking_record.get(booking_id)
        if booking_id == "None" or booking is None:
            return {"invoice": {}}
        return {
            "invoice": {
                "booking_id": booking_id,
                "travel_date": get_str(booking, "travel_date"),
                "travel_from": get_str(booking, "travel_from"),
                "travel_to": get_str(booking, "travel_to"),
                "travel_class": get_str(booking, "travel_class"),
                "travel_cost": get_float(booking, "travel_cost"),
                "transaction_id": get_str(booking, "transaction_id"),
            }
        }
