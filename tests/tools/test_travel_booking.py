import pytest
import json
from tools.travel_booking import TravelBooking


@pytest.fixture
def travel_booking_instance():
    initial_config = {
        "credit_card_list": {
            "12345": {
                "card_number": "123456",
                "expiration_date": "12/2028",
                "cardholder_name": "Michael Smith",
                "card_verification_number": 465,
                "balance": 50000.0
            }
        },
        "booking_record": {
            "flight_001": {
                "travel_to": "Rome",
                "travel_from": "New York",
                "insurance": "none",
                "travel_cost": 1200.5,
                "travel_date": "2024-08-08",
                "travel_class": "Business",
                "transaction_id": "12345",
                "card_id": "12345"
            }
        },
        "access_token": "abc123xyz",
        "token_type": "Bearer",
        "token_expires_in": 3600,
        "token_scope": "full_access",
        "user_first_name": "Michael",
        "user_last_name": "Smith",
        "budget_limit": 5000.0
    }
    return TravelBooking(initial_config)


class TestAuthenticateTravel:
    def test_authenticate_travel_success(self, travel_booking_instance):
        result = travel_booking_instance.authenticate_travel(
            client_id='client_520',
            client_secret='testpass',
            refresh_token='token990125',
            grant_type='read_write',
            user_first_name='Michael',
            user_last_name='Thompson'
        )
        assert result.get("access_token") != ""
        assert result.get("token_type") == "Bearer"
        assert result.get("expires_in") == 3600
        assert result.get("scope") == "read_write"

    def test_authenticate_travel_new_user(self, travel_booking_instance):
        result = travel_booking_instance.authenticate_travel(
            client_id='trav3lMaxID2023',
            client_secret='testpass',
            refresh_token='r3freshM3n0w',
            grant_type='read_write',
            user_first_name='Maxwell',
            user_last_name='Edison'
        )
        assert result.get("access_token") != ""
        assert travel_booking_instance.user_first_name == "Maxwell"
        assert travel_booking_instance.user_last_name == "Edison"

    def test_authenticate_travel_invalid_grant_type(self, travel_booking_instance):
        result = travel_booking_instance.authenticate_travel(
            client_id='client_520',
            client_secret='testpass',
            refresh_token='token990125',
            grant_type='invalid_type',
            user_first_name='Michael',
            user_last_name='Thompson'
        )
        assert result.get("access_token") == ""
        assert result.get("expires_in") == 0


class TestBookFlight:
    def test_book_flight_success(self, travel_booking_instance):
        result = travel_booking_instance.book_flight(
            access_token='abc123xyz',
            card_id='12345',
            travel_date='2024-11-10',
            travel_from='SFO',
            travel_to='LAX',
            travel_class='business',
            travel_cost=400.0
        )
        assert result.get("booking_status") is True
        assert result.get("booking_id") != ""
        assert result.get("transaction_id") != ""

    def test_book_flight_insufficient_balance(self, travel_booking_instance):
        result = travel_booking_instance.book_flight(
            access_token='abc123xyz',
            card_id='12345',
            travel_date='2024-11-15',
            travel_from='SFO',
            travel_to='ORD',
            travel_cost=60000.0,
            travel_class='first'
        )
        assert result.get("booking_status") is False
        assert result.get("booking_id") == ""

    def test_book_flight_invalid_token(self, travel_booking_instance):
        result = travel_booking_instance.book_flight(
            access_token='invalid_token',
            card_id='12345',
            travel_date='2024-11-10',
            travel_from='SFO',
            travel_to='LAX',
            travel_class='economy',
            travel_cost=300.0
        )
        assert result.get("booking_status") is False


class TestCancelBooking:
    def test_cancel_booking_success(self, travel_booking_instance):
        initial_balance = travel_booking_instance.credit_card_list["12345"]["balance"]
        result = travel_booking_instance.cancel_booking(
            access_token='abc123xyz',
            booking_id='flight_001'
        )
        assert result.get("cancel_status") is True
        assert travel_booking_instance.credit_card_list["12345"]["balance"] == initial_balance + 1200.5

    def test_cancel_booking_invalid_token(self, travel_booking_instance):
        result = travel_booking_instance.cancel_booking(
            access_token='wrong_token',
            booking_id='flight_001'
        )
        assert result.get("cancel_status") is False

    def test_cancel_booking_not_found(self, travel_booking_instance):
        result = travel_booking_instance.cancel_booking(
            access_token='abc123xyz',
            booking_id='nonexistent_booking'
        )
        assert result.get("cancel_status") is False


class TestComputeExchangeRate:
    def test_compute_exchange_rate_gbp_to_usd(self, travel_booking_instance):
        result = travel_booking_instance.compute_exchange_rate(
            base_currency='GBP',
            target_currency='USD',
            value=15400.0
        )
        assert result.get("exchanged_value") > 0

    def test_compute_exchange_rate_usd_to_eur(self, travel_booking_instance):
        result = travel_booking_instance.compute_exchange_rate(
            base_currency='USD',
            target_currency='EUR',
            value=400.0
        )
        assert result.get("exchanged_value") > 0

    def test_compute_exchange_rate_same_currency(self, travel_booking_instance):
        result = travel_booking_instance.compute_exchange_rate(
            base_currency='USD',
            target_currency='USD',
            value=100.0
        )
        assert result.get("exchanged_value") == 100.0

    def test_compute_exchange_rate_invalid_currency(self, travel_booking_instance):
        result = travel_booking_instance.compute_exchange_rate(
            base_currency='INVALID',
            target_currency='USD',
            value=100.0
        )
        assert result.get("exchanged_value") == 0.0


class TestContactCustomerSupport:
    def test_contact_customer_support_success(self, travel_booking_instance):
        result = travel_booking_instance.contact_customer_support(
            booking_id='flight_001',
            message='Urgent: last-minute complication with my reservation.'
        )
        assert "flight_001" in result.get("customer_support_message", "")

    def test_contact_customer_support_not_found(self, travel_booking_instance):
        result = travel_booking_instance.contact_customer_support(
            booking_id='nonexistent_booking',
            message='Help!'
        )
        assert "No booking found" in result.get("customer_support_message", "")

    def test_contact_customer_support_empty_message(self, travel_booking_instance):
        result = travel_booking_instance.contact_customer_support(
            booking_id='flight_001',
            message=''
        )
        assert "customer_support_message" in result


class TestGetBudgetFiscalYear:
    def test_get_budget_fiscal_year_default(self, travel_booking_instance):
        result = travel_booking_instance.get_budget_fiscal_year()
        assert "budget_fiscal_year" in result

    def test_get_budget_fiscal_year_with_modified_after(self, travel_booking_instance):
        result = travel_booking_instance.get_budget_fiscal_year(
            lastModifiedAfter='2023-01-01T00:00:00'
        )
        assert "budget_fiscal_year" in result

    def test_get_budget_fiscal_year_include_removed(self, travel_booking_instance):
        result = travel_booking_instance.get_budget_fiscal_year(
            includeRemoved=True
        )
        assert "budget_fiscal_year" in result


class TestGetCreditCardBalance:
    def test_get_credit_card_balance_success(self, travel_booking_instance):
        result = travel_booking_instance.get_credit_card_balance(
            access_token='abc123xyz',
            card_id='12345'
        )
        assert result.get("card_balance") == 50000.0

    def test_get_credit_card_balance_invalid_token(self, travel_booking_instance):
        result = travel_booking_instance.get_credit_card_balance(
            access_token='wrong_token',
            card_id='12345'
        )
        assert result.get("card_balance") == 0.0

    def test_get_credit_card_balance_not_found(self, travel_booking_instance):
        result = travel_booking_instance.get_credit_card_balance(
            access_token='abc123xyz',
            card_id='nonexistent_card'
        )
        assert result.get("card_balance") == 0.0


class TestGetFlightCost:
    def test_get_flight_cost_business(self, travel_booking_instance):
        result = travel_booking_instance.get_flight_cost(
            travel_from='SFO',
            travel_to='LAX',
            travel_date='2024-11-10',
            travel_class='business'
        )
        costs = result.get("travel_cost_list", [])
        assert isinstance(costs, list) and len(costs) > 0

    def test_get_flight_cost_economy(self, travel_booking_instance):
        result = travel_booking_instance.get_flight_cost(
            travel_from='RMS',
            travel_to='SBK',
            travel_date='2024-10-06',
            travel_class='economy'
        )
        costs = result.get("travel_cost_list", [])
        assert isinstance(costs, list) and len(costs) > 0

    def test_get_flight_cost_invalid_class(self, travel_booking_instance):
        result = travel_booking_instance.get_flight_cost(
            travel_from='SFO',
            travel_to='LAX',
            travel_date='2024-11-10',
            travel_class='luxury'
        )
        assert result.get("travel_cost_list") == []


class TestGetNearestAirportByCity:
    def test_get_nearest_airport_crescent_hollow(self, travel_booking_instance):
        result = travel_booking_instance.get_nearest_airport_by_city(
            location='Crescent Hollow'
        )
        assert result.get("nearest_airport") == "CSH"

    def test_get_nearest_airport_rivermist(self, travel_booking_instance):
        result = travel_booking_instance.get_nearest_airport_by_city(
            location='Rivermist'
        )
        assert result.get("nearest_airport") == "RVM"

    def test_get_nearest_airport_unknown_city(self, travel_booking_instance):
        result = travel_booking_instance.get_nearest_airport_by_city(
            location='Unknown City XYZ'
        )
        assert result.get("nearest_airport") == ""


class TestPurchaseInsurance:
    def test_purchase_insurance_comprehensive(self, travel_booking_instance):
        result = travel_booking_instance.purchase_insurance(
            access_token='abc123xyz',
            insurance_type='comprehensive',
            insurance_cost=120.0,
            booking_id='flight_001',
            card_id='12345'
        )
        assert result.get("insurance_status") is True
        assert result.get("insurance_id") != ""

    def test_purchase_insurance_insufficient_balance(self, travel_booking_instance):
        result = travel_booking_instance.purchase_insurance(
            access_token='abc123xyz',
            insurance_type='comprehensive',
            booking_id='flight_001',
            insurance_cost=99999.0,
            card_id='12345'
        )
        assert result.get("insurance_status") is False

    def test_purchase_insurance_invalid_token(self, travel_booking_instance):
        result = travel_booking_instance.purchase_insurance(
            access_token='invalid_token',
            insurance_type='basic',
            insurance_cost=50.0,
            booking_id='flight_001',
            card_id='12345'
        )
        assert result.get("insurance_status") is False


class TestRegisterCreditCard:
    def test_register_credit_card_success(self, travel_booking_instance):
        result = travel_booking_instance.register_credit_card(
            access_token='abc123xyz',
            card_number='4012888888881881',
            expiration_date='12/2028',
            card_verification_number=465,
            cardholder_name='Michael Smith'
        )
        assert result.get("card_id") != ""

    def test_register_credit_card_invalid_token(self, travel_booking_instance):
        result = travel_booking_instance.register_credit_card(
            access_token='invalid_token',
            card_number='4012888888881881',
            expiration_date='12/2028',
            card_verification_number=465,
            cardholder_name='Michael Smith'
        )
        assert result.get("card_id") == ""


class TestRetrieveInvoice:
    def test_retrieve_invoice_success(self, travel_booking_instance):
        result = travel_booking_instance.retrieve_invoice(
            access_token='abc123xyz',
            booking_id='flight_001'
        )
        assert "invoice" in result
        assert result["invoice"].get("booking_id") == "flight_001"

    def test_retrieve_invoice_invalid_token(self, travel_booking_instance):
        result = travel_booking_instance.retrieve_invoice(
            access_token='wrong_token',
            booking_id='flight_001'
        )
        assert result.get("invoice") == {}

    def test_retrieve_invoice_not_found(self, travel_booking_instance):
        result = travel_booking_instance.retrieve_invoice(
            access_token='abc123xyz',
            booking_id='nonexistent_booking'
        )
        assert result.get("invoice") == {}


class TestSetBudgetLimit:
    def test_set_budget_limit_success(self, travel_booking_instance):
        result = travel_booking_instance.set_budget_limit(
            access_token='abc123xyz',
            budget_limit=5000
        )
        assert result.get("budget_limit") == 5000

    def test_set_budget_limit_invalid_token(self, travel_booking_instance):
        result = travel_booking_instance.set_budget_limit(
            access_token='invalid_token',
            budget_limit=1000.0
        )
        assert result.get("budget_limit") == 0.0


class TestVerifyTravelerInformation:
    def test_verify_traveler_information_valid(self, travel_booking_instance):
        result = travel_booking_instance.verify_traveler_information(
            first_name='Michael',
            last_name='Smith',
            date_of_birth='1962-02-14',
            passport_number='P87654321'
        )
        assert result.get("verification_status") is True
        assert result.get("verification_failure") == ""

    def test_verify_traveler_information_invalid_dob(self, travel_booking_instance):
        result = travel_booking_instance.verify_traveler_information(
            first_name='Michael',
            last_name='Thompson',
            date_of_birth='invalid-date',
            passport_number='P12345678'
        )
        assert result.get("verification_status") is False
        assert "date of birth" in result.get("verification_failure", "").lower()

    def test_verify_traveler_information_empty_passport(self, travel_booking_instance):
        result = travel_booking_instance.verify_traveler_information(
            first_name='John',
            last_name='Doe',
            date_of_birth='1990-05-15',
            passport_number=''
        )
        assert result.get("verification_status") is False

    def test_verify_traveler_information_empty_name(self, travel_booking_instance):
        result = travel_booking_instance.verify_traveler_information(
            first_name='',
            last_name='Doe',
            date_of_birth='1990-05-15',
            passport_number='P123'
        )
        assert result.get("verification_status") is False
