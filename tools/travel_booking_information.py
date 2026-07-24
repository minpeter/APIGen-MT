"""Reference and informational operations for TravelBooking."""

import math
from datetime import date, datetime

from .travel_booking_state import TravelBookingState
from .travel_booking_types import (
    AirportResult,
    CustomerSupportResult,
    ExchangeRateResult,
    FiscalYearResult,
    FlightCostResult,
)


class TravelBookingInformationMixin(TravelBookingState):
    """Provide exchange-rate, support, flight-cost, and airport lookups."""

    def compute_exchange_rate(
        self, base_currency: str, target_currency: str, value: float
    ) -> ExchangeRateResult:
        """Compute the exchange rate between two currencies."""
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
        base_rate = exchange_rates_to_usd.get(base_currency)
        target_rate = exchange_rates_to_usd.get(target_currency)
        if base_rate is None or target_rate is None or target_rate == 0.0:
            return {"exchanged_value": 0.0}
        return {"exchanged_value": value * base_rate / target_rate}

    def contact_customer_support(
        self, booking_id: str, message: str
    ) -> CustomerSupportResult:
        """Contact travel-booking customer support."""
        if booking_id not in self.booking_record:
            return {
                "customer_support_message": (
                    f"No booking found for ID {booking_id}. "
                    "Unable to process request."
                )
            }
        return {
            "customer_support_message": (
                "Customer support has received your message regarding "
                f"booking {booking_id}: '{message}'. A representative will "
                "reach out shortly."
            )
        }

    def get_budget_fiscal_year(
        self,
        lastModifiedAfter: str = "None",
        includeRemoved: str = "None",
    ) -> FiscalYearResult:
        """Get the current budget fiscal year."""
        _ = lastModifiedAfter, includeRemoved
        current_year = datetime.now().astimezone().year
        return {"budget_fiscal_year": f"{current_year}-{current_year + 1}"}

    def get_flight_cost(
        self,
        travel_from: str,
        travel_to: str,
        travel_date: str,
        travel_class: str,
    ) -> FlightCostResult:
        """Get representative flight costs in USD."""
        if not travel_date or not travel_date.strip():
            return {
                "error": (
                    "travel_date is required and must be a non-empty date "
                    "string (YYYY-MM-DD)."
                ),
                "travel_cost_list": [],
            }
        if not travel_from or not travel_to:
            return {
                "error": "travel_from and travel_to are required.",
                "travel_cost_list": [],
            }
        base_costs = {"economy": 300.0, "business": 800.0, "first": 1500.0}
        base_cost = base_costs.get(travel_class.lower())
        if base_cost is None:
            return {"travel_cost_list": []}
        try:
            day_of_year = date.fromisoformat(travel_date).timetuple().tm_yday
        except ValueError:
            return {
                "error": (
                    f"Invalid travel_date '{travel_date}'. "
                    "Expected format: YYYY-MM-DD."
                ),
                "travel_cost_list": [],
            }
        variation = math.sin(
            day_of_year * 0.1
            + len(travel_from) * 0.5
            + len(travel_to) * 0.3
        ) * 50
        cost = base_cost + variation
        return {
            "travel_cost_list": [
                round(cost, 2),
                round(cost * 1.1, 2),
                round(cost * 0.9, 2),
            ],
            "travel_from": travel_from,
            "travel_to": travel_to,
            "travel_date": travel_date,
            "travel_class": travel_class,
            "currency": "USD",
        }

    def get_nearest_airport_by_city(self, location: str) -> AirportResult:
        """Get the nearest airport to a city."""
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
            "Seattle": "SEA",
            "Miami": "MIA",
            "Dallas": "DFW",
            "Atlanta": "ATL",
            "Denver": "DEN",
            "Phoenix": "PHX",
            "Las Vegas": "LAS",
            "Orlando": "MCO",
            "Honolulu": "HNL",
            "Washington D.C.": "DCA",
            "Dubai": "DXB",
            "Singapore": "SIN",
            "Sydney": "SYD",
            "Mumbai": "BOM",
            "Shanghai": "PVG",
            "Toronto": "YYZ",
            "Vancouver": "YVR",
            "Mexico City": "MEX",
            "Sao Paulo": "GRU",
            "Amsterdam": "AMS",
            "Madrid": "MAD",
            "Munich": "MUC",
            "Zurich": "ZRH",
            "Barcelona": "BCN",
            "Milan": "MXP",
            "Istanbul": "IST",
            "Bangkok": "BKK",
            "Jakarta": "CGK",
            "Kuala Lumpur": "KUL",
            "Manila": "MNL",
            "Seoul": "ICN",
            "Taipei": "TPE",
            "Frankfurt": "FRA",
            "Brussels": "BRU",
            "Vienna": "VIE",
            "Prague": "PRG",
            "Stockholm": "ARN",
            "Copenhagen": "CPH",
            "Oslo": "OSL",
            "Helsinki": "HEL",
            "Warsaw": "WAW",
            "Lisbon": "LIS",
            "Dublin": "DUB",
            "Athens": "ATH",
            "Moscow": "SVO",
            "San Diego": "SAN",
            "Portland": "PDX",
            "Austin": "AUS",
            "Nashville": "BNA",
            "San Jose": "SJC",
            "Tampa": "TPA",
            "Raleigh": "RDU",
            "Detroit": "DTW",
            "Charlotte": "CLT",
            "Minneapolis": "MSP",
            "Philadelphia": "PHL",
            "Baltimore": "BWI",
            "Newark": "EWR",
            "Fort Lauderdale": "FLL",
            "Pittsburgh": "PIT",
        }
        return {"nearest_airport": city_to_airport.get(location, "")}
