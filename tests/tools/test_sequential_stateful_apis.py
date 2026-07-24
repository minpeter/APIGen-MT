"""Sequential function-calling tests for stateful APIs.

Tests correct and problematic sequential scenarios where the output of one
call is required as input to the next call.  These validate that:

1. Proper login/auth → subsequent operation sequences succeed
2. Missing or wrong auth → subsequent operations fail predictably
3. Chained operations where step N depends on step N-1 output work correctly
4. State is properly isolated between independent sequences
"""

import json
import pytest

from tools.message_api import MessageAPI
from tools.posting_api import PostingAPI
from tools.trading_bot import TradingBot
from tools.travel_booking import TravelBooking
from tools.ticket_api import TicketAPI


# ─── Shared configs (match FULL_INITIAL_CONFIGS in tool_manager.py) ──────────

MESSAGE_API_CONFIG = {
    "workspace_id": "WS123456",
    "user_count": 4,
    "user_map": {
        "Michael": "USR005",
        "Sarah": "USR006",
        "David": "USR007",
        "Emma": "USR008",
    },
    "messages_sent_map": {
        "USR005": {"USR006": ["Please review the attached document."]},
        "USR006": {"USR005": ["Got it, thanks!"]},
    },
    "messages_inbox_map": {
        "USR005": {"USR006": ["Got it, thanks!"]},
        "USR006": {"USR005": ["Please review the attached document."]},
    },
    "message_count": 8,
    "current_user": "USR005",
}

POSTING_API_CONFIG = {
    "authenticated": False,
    "tweet_counter": 10,
    "tweets": {
        "0": {
            "id": 0,
            "username": "genealogy_enthusiast",
            "content": "Excited to start my genealogy journey!",
            "tags": ["#genealogy"],
            "mentions": [],
        }
    },
    "comments": {},
    "retweets": [],
    "following_list": ["history_buff"],
    "users": {
        "history_buff": {"tweet_count": 25, "following_count": 50, "retweet_count": 10},
    },
    "username": "genealogy_enthusiast",
    "password": "testpass",
}

TRADING_BOT_CONFIG = {
    "account_info": {
        "account_id": 12345,
        "balance": 10000.0,
        "binding_card": 1974202140965533,
    },
    "authenticated": False,
    "market_status": "Closed",
    "order_counter": 12446,
    "stocks": {
        "AAPL": {"price": 227.16, "percent_change": 0.17, "volume": 2.552, "MA(5)": 227.11, "MA(20)": 227.09},
        "TSLA": {"price": 667.92, "percent_change": -0.12, "volume": 1.654, "MA(5)": 671.15, "MA(20)": 668.2},
    },
    "watch_list": ["NVDA"],
    "transaction_history": [],
}

TRAVEL_CONFIG = {
    "credit_card_list": {
        "12345": {
            "card_number": "123456",
            "expiration_date": "12/2028",
            "cardholder_name": "Michael Smith",
            "card_verification_number": 465,
            "balance": 50000.0,
        }
    },
    "booking_record": {},
    "access_token": "",
    "token_type": "Bearer",
    "token_expires_in": 0,
    "token_scope": "",
    "user_first_name": "",
    "user_last_name": "",
    "budget_limit": 5000.0,
}

TICKET_API_CONFIG = {
    "tickets_queue": [
        {"id": 123456, "title": "System Error", "description": "Critical error.", "status": "Open", "priority": 3, "created_by": "agent_a"},
    ],
    "ticket_count": 123456,
    "current_user": "",
}


# ═══════════════════════════════════════════════════════════════════════════
# MessageAPI sequential tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMessageAPISequentialCorrect:
    """Correct sequences: login → lookup → send → search → delete."""

    def test_login_then_send_message(self):
        api = MessageAPI(initial_config=json.loads(json.dumps(MESSAGE_API_CONFIG)))
        login_result = api.message_login(user_id="USR006")
        assert login_result["login_status"] is True
        send_result = api.send_message(receiver_id="USR005", message="Hello from Sarah!")
        assert send_result["sent_status"] is True
        assert send_result["message_id"] is not None

    def test_login_then_lookup_then_send(self):
        api = MessageAPI(initial_config=json.loads(json.dumps(MESSAGE_API_CONFIG)))
        api.message_login(user_id="USR005")
        lookup = api.get_user_id(user="Sarah")
        assert lookup["user_id"] == "USR006"
        send = api.send_message(receiver_id=lookup["user_id"], message="Hi Sarah!")
        assert send["sent_status"] is True

    def test_login_send_then_search(self):
        api = MessageAPI(initial_config=json.loads(json.dumps(MESSAGE_API_CONFIG)))
        api.message_login(user_id="USR005")
        api.send_message(receiver_id="USR006", message="Quarterly report is ready")
        search = api.search_messages(keyword="Quarterly")
        assert len(search["results"]) > 0

    def test_login_send_then_delete(self):
        api = MessageAPI(initial_config=json.loads(json.dumps(MESSAGE_API_CONFIG)))
        api.message_login(user_id="USR005")
        api.send_message(receiver_id="USR006", message="Temporary message")
        delete = api.delete_message(receiver_id="USR006")
        assert delete["deleted_status"] is True

    def test_add_contact_then_lookup_and_login(self):
        api = MessageAPI(initial_config=json.loads(json.dumps(MESSAGE_API_CONFIG)))
        add = api.add_contact(user_name="Frank")
        assert add["added_status"] is True
        new_id = add["user_id"]
        lookup = api.get_user_id(user="Frank")
        assert lookup["user_id"] == new_id
        login = api.message_login(user_id=new_id)
        assert login["login_status"] is True

    def test_login_as_different_user_then_send(self):
        api = MessageAPI(initial_config=json.loads(json.dumps(MESSAGE_API_CONFIG)))
        api.message_login(user_id="USR006")
        send = api.send_message(receiver_id="USR007", message="From Sarah to David")
        assert send["sent_status"] is True


class TestMessageAPISequentialProblematic:
    """Problematic sequences: operations without login or with bad state."""

    def test_send_without_login(self):
        config = json.loads(json.dumps(MESSAGE_API_CONFIG))
        config["current_user"] = ""
        api = MessageAPI(initial_config=config)
        send = api.send_message(receiver_id="USR006", message="Hello?")
        assert send["sent_status"] is False

    def test_delete_without_login(self):
        config = json.loads(json.dumps(MESSAGE_API_CONFIG))
        config["current_user"] = ""
        api = MessageAPI(initial_config=config)
        delete = api.delete_message(receiver_id="USR006")
        assert delete["deleted_status"] is False

    def test_search_without_login(self):
        config = json.loads(json.dumps(MESSAGE_API_CONFIG))
        config["current_user"] = ""
        api = MessageAPI(initial_config=config)
        search = api.search_messages(keyword="Meeting")
        assert search["results"] == []

    def test_login_with_nonexistent_user_still_has_old_session(self):
        config = json.loads(json.dumps(MESSAGE_API_CONFIG))
        config["current_user"] = ""
        api = MessageAPI(initial_config=config)
        login = api.message_login(user_id="USR999")
        assert login["login_status"] is False
        send = api.send_message(receiver_id="USR006", message="Should fail")
        assert send["sent_status"] is False

    def test_login_empty_then_send_with_no_prior_session(self):
        config = json.loads(json.dumps(MESSAGE_API_CONFIG))
        config["current_user"] = ""
        api = MessageAPI(initial_config=config)
        login = api.message_login(user_id="")
        assert login["login_status"] is False
        send = api.send_message(receiver_id="USR006", message="Should fail")
        assert send["sent_status"] is False

    def test_login_then_send_to_nonexistent_receiver(self):
        api = MessageAPI(initial_config=json.loads(json.dumps(MESSAGE_API_CONFIG)))
        api.message_login(user_id="USR005")
        send = api.send_message(receiver_id="USR999", message="Hello?")
        assert send["sent_status"] is True

    def test_send_empty_message_after_login(self):
        api = MessageAPI(initial_config=json.loads(json.dumps(MESSAGE_API_CONFIG)))
        api.message_login(user_id="USR005")
        send = api.send_message(receiver_id="USR006", message="")
        assert send["sent_status"] is False


# ═══════════════════════════════════════════════════════════════════════════
# PostingAPI sequential tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPostingAPISequentialCorrect:
    """Correct sequences: authenticate → post → comment → retweet → follow."""

    def test_authenticate_then_post_tweet(self):
        api = PostingAPI(initial_config=json.loads(json.dumps(POSTING_API_CONFIG)))
        assert api.authenticated is False
        auth = api.authenticate_twitter(username="genealogy_enthusiast", password="testpass")
        assert auth["authentication_status"] is True
        tweet = api.post_tweet(content="My first post!", tags=["#hello"])
        assert tweet["id"] != 0
        assert tweet["content"] == "My first post!"

    def test_authenticate_then_post_then_comment(self):
        api = PostingAPI(initial_config=json.loads(json.dumps(POSTING_API_CONFIG)))
        api.authenticate_twitter(username="genealogy_enthusiast", password="testpass")
        tweet = api.post_tweet(content="Needs feedback", tags=["#review"])
        comment = api.comment(tweet_id=tweet["id"], comment_content="Looks great!")
        assert "successfully" in comment["comment_status"].lower()

    def test_authenticate_then_retweet(self):
        api = PostingAPI(initial_config=json.loads(json.dumps(POSTING_API_CONFIG)))
        api.authenticate_twitter(username="genealogy_enthusiast", password="testpass")
        retweet = api.retweet(tweet_id=0)
        assert "successfully" in retweet["retweet_status"].lower()

    def test_authenticate_then_follow_then_unfollow(self):
        api = PostingAPI(initial_config=json.loads(json.dumps(POSTING_API_CONFIG)))
        api.authenticate_twitter(username="genealogy_enthusiast", password="testpass")
        follow = api.follow_user(username_to_follow="puzzle_solver")
        assert follow["follow_status"] is True
        unfollow = api.unfollow_user(username_to_unfollow="puzzle_solver")
        assert unfollow["unfollow_status"] is True

    def test_authenticate_then_mention(self):
        api = PostingAPI(initial_config=json.loads(json.dumps(POSTING_API_CONFIG)))
        api.authenticate_twitter(username="genealogy_enthusiast", password="testpass")
        mention = api.mention(tweet_id=0, mentioned_usernames=["@history_buff"])
        assert "successfully" in mention["mention_status"].lower()

    def test_post_then_get_tweet(self):
        api = PostingAPI(initial_config=json.loads(json.dumps(POSTING_API_CONFIG)))
        api.authenticate_twitter(username="genealogy_enthusiast", password="testpass")
        posted = api.post_tweet(content="Read my new blog post", tags=["#blog"])
        fetched = api.get_tweet(tweet_id=posted["id"])
        assert fetched["id"] == posted["id"]
        assert fetched["content"] == "Read my new blog post"

    def test_post_then_search(self):
        api = PostingAPI(initial_config=json.loads(json.dumps(POSTING_API_CONFIG)))
        api.authenticate_twitter(username="genealogy_enthusiast", password="testpass")
        api.post_tweet(content="Quantum computing breakthrough!", tags=["#quantum"])
        search = api.search_tweets(keyword="Quantum")
        assert len(search["matching_tweets"]) > 0


class TestPostingAPISequentialProblematic:
    """Problematic sequences: operations without authentication."""

    def test_post_without_auth(self):
        api = PostingAPI(initial_config=json.loads(json.dumps(POSTING_API_CONFIG)))
        assert api.authenticated is False
        tweet = api.post_tweet(content="Should not work")
        assert tweet["id"] == 0
        assert tweet["content"] == ""

    def test_comment_without_auth(self):
        api = PostingAPI(initial_config=json.loads(json.dumps(POSTING_API_CONFIG)))
        comment = api.comment(tweet_id=0, comment_content="Nice!")
        assert "not authenticated" in comment["comment_status"].lower()

    def test_retweet_without_auth(self):
        api = PostingAPI(initial_config=json.loads(json.dumps(POSTING_API_CONFIG)))
        retweet = api.retweet(tweet_id=0)
        assert "not authenticated" in retweet["retweet_status"].lower()

    def test_follow_without_auth(self):
        api = PostingAPI(initial_config=json.loads(json.dumps(POSTING_API_CONFIG)))
        follow = api.follow_user(username_to_follow="history_buff")
        assert follow["follow_status"] is False

    def test_unfollow_without_auth(self):
        api = PostingAPI(initial_config=json.loads(json.dumps(POSTING_API_CONFIG)))
        unfollow = api.unfollow_user(username_to_unfollow="history_buff")
        assert unfollow["unfollow_status"] is False

    def test_mention_without_auth(self):
        api = PostingAPI(initial_config=json.loads(json.dumps(POSTING_API_CONFIG)))
        mention = api.mention(tweet_id=0, mentioned_usernames=["@history_buff"])
        assert "not authenticated" in mention["mention_status"].lower()

    def test_authenticate_wrong_password_then_post(self):
        api = PostingAPI(initial_config=json.loads(json.dumps(POSTING_API_CONFIG)))
        auth = api.authenticate_twitter(username="genealogy_enthusiast", password="wrong")
        assert auth["authentication_status"] is False
        tweet = api.post_tweet(content="Should fail")
        assert tweet["id"] == 0

    def test_authenticate_wrong_username_then_post(self):
        api = PostingAPI(initial_config=json.loads(json.dumps(POSTING_API_CONFIG)))
        auth = api.authenticate_twitter(username="wrong_user", password="testpass")
        assert auth["authentication_status"] is False
        tweet = api.post_tweet(content="Should fail")
        assert tweet["id"] == 0

    def test_comment_on_nonexistent_tweet_after_auth(self):
        api = PostingAPI(initial_config=json.loads(json.dumps(POSTING_API_CONFIG)))
        api.authenticate_twitter(username="genealogy_enthusiast", password="testpass")
        comment = api.comment(tweet_id=99999, comment_content="Ghost comment")
        assert "not found" in comment["comment_status"].lower()


# ═══════════════════════════════════════════════════════════════════════════
# TradingBot sequential tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTradingBotSequentialCorrect:
    """Correct sequences: login → place order → get details → cancel."""

    def test_login_then_place_order(self):
        bot = TradingBot(initial_config=json.loads(json.dumps(TRADING_BOT_CONFIG)))
        assert bot.authenticated is False
        login = bot.trading_login(username="trader", password="testpass")
        assert login["status"] == "Login successful"
        assert bot.authenticated is True
        order = bot.place_order(order_type="Buy", symbol="AAPL", price=227.16, amount=10)
        assert order["order_id"] is not None

    def test_login_place_order_then_get_details(self):
        bot = TradingBot(initial_config=json.loads(json.dumps(TRADING_BOT_CONFIG)))
        bot.trading_login(username="trader", password="testpass")
        order = bot.place_order(order_type="Buy", symbol="TSLA", price=667.92, amount=5)
        details = bot.get_order_details(order_id=order["order_id"])
        assert details["id"] == order["order_id"]
        assert details["status"] == "Open"

    def test_login_place_order_then_cancel(self):
        bot = TradingBot(initial_config=json.loads(json.dumps(TRADING_BOT_CONFIG)))
        bot.trading_login(username="trader", password="testpass")
        order = bot.place_order(order_type="Buy", symbol="AAPL", price=227.16, amount=10)
        cancel = bot.cancel_order(order_id=order["order_id"])
        assert cancel["status"] == "Cancelled"
        details = bot.get_order_details(order_id=order["order_id"])
        assert details["status"] == "Cancelled"

    def test_login_then_fund_then_check_balance(self):
        bot = TradingBot(initial_config=json.loads(json.dumps(TRADING_BOT_CONFIG)))
        bot.trading_login(username="trader", password="testpass")
        initial_balance = bot.account_info["balance"]
        fund = bot.fund_account(amount=5000.0)
        assert fund["status"] == "Account funded successfully"
        assert bot.account_info["balance"] == initial_balance + 5000.0

    def test_login_then_deposit_then_withdraw(self):
        bot = TradingBot(initial_config=json.loads(json.dumps(TRADING_BOT_CONFIG)))
        bot.trading_login(username="trader", password="testpass")
        deposit = bot.make_transaction(account_id=12345, xact_type="deposit", amount=2000.0)
        assert "successful" in deposit["status"].lower()
        withdraw = bot.make_transaction(account_id=12345, xact_type="withdrawal", amount=500.0)
        assert "successful" in withdraw["status"].lower()
        assert bot.account_info["balance"] == 10000.0 + 2000.0 - 500.0

    def test_login_then_add_and_remove_from_watchlist(self):
        bot = TradingBot(initial_config=json.loads(json.dumps(TRADING_BOT_CONFIG)))
        bot.trading_login(username="trader", password="testpass")
        add = bot.add_to_watchlist(stock="AAPL")
        assert "AAPL" in bot.watch_list
        remove = bot.remove_stock_from_watchlist(symbol="AAPL")
        assert "AAPL" not in bot.watch_list

    def test_get_stock_info_then_filter_by_price(self):
        bot = TradingBot(initial_config=json.loads(json.dumps(TRADING_BOT_CONFIG)))
        info = bot.get_stock_info(symbol="AAPL")
        assert info["price"] == 227.16
        filtered = bot.filter_stocks_by_price(
            stocks=["AAPL", "TSLA"], min_price=200.0, max_price=500.0
        )
        assert "AAPL" in filtered["filtered_stocks"]
        assert "TSLA" not in filtered["filtered_stocks"]


class TestTradingBotSequentialProblematic:
    """Problematic sequences for TradingBot (fewer auth gates, but bad state)."""

    def test_cancel_nonexistent_order(self):
        bot = TradingBot(initial_config=json.loads(json.dumps(TRADING_BOT_CONFIG)))
        cancel = bot.cancel_order(order_id=99999)
        assert cancel["status"] == "Order not found"

    def test_withdraw_more_than_balance(self):
        bot = TradingBot(initial_config=json.loads(json.dumps(TRADING_BOT_CONFIG)))
        withdraw = bot.make_transaction(account_id=12345, xact_type="withdrawal", amount=99999.0)
        assert "failed" in withdraw["status"].lower() or "Failed" in withdraw["status"]

    def test_transaction_wrong_account(self):
        bot = TradingBot(initial_config=json.loads(json.dumps(TRADING_BOT_CONFIG)))
        result = bot.make_transaction(account_id=99999, xact_type="deposit", amount=500.0)
        assert "failed" in result["status"].lower() or "Failed" in result["status"]

    def test_fund_account_negative(self):
        bot = TradingBot(initial_config=json.loads(json.dumps(TRADING_BOT_CONFIG)))
        result = bot.fund_account(amount=-100.0)
        assert "failed" in result["status"].lower() or "Failed" in result["status"]

    def test_double_cancel_same_order(self):
        bot = TradingBot(initial_config=json.loads(json.dumps(TRADING_BOT_CONFIG)))
        order = bot.place_order(order_type="Buy", symbol="AAPL", price=227.16, amount=10)
        bot.cancel_order(order_id=order["order_id"])
        second_cancel = bot.cancel_order(order_id=order["order_id"])
        assert second_cancel["status"] == "Cancelled"

    def test_get_stock_info_nonexistent(self):
        bot = TradingBot(initial_config=json.loads(json.dumps(TRADING_BOT_CONFIG)))
        info = bot.get_stock_info(symbol="FAKE")
        assert info["price"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TravelBooking sequential tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTravelBookingSequentialCorrect:
    """Correct sequences: authenticate → book → purchase insurance → cancel."""

    def test_authenticate_then_book_flight(self):
        api = TravelBooking(initial_config=json.loads(json.dumps(TRAVEL_CONFIG)))
        assert api.access_token == ""
        auth = api.authenticate_travel(
            client_id="client1", client_secret="secret1",
            refresh_token="refresh1", grant_type="read_write",
            user_first_name="Michael", user_last_name="Smith"
        )
        assert auth["access_token"] != ""
        book = api.book_flight(
            access_token=api.access_token, card_id="12345",
            travel_date="2025-06-01", travel_from="SFO",
            travel_to="LAX", travel_class="economy", travel_cost=300.0
        )
        assert book["booking_status"] is True
        assert book["booking_id"] != ""

    def test_authenticate_book_then_get_balance(self):
        api = TravelBooking(initial_config=json.loads(json.dumps(TRAVEL_CONFIG)))
        api.authenticate_travel(
            client_id="c1", client_secret="s1",
            refresh_token="r1", grant_type="read_write",
            user_first_name="M", user_last_name="S"
        )
        initial_balance = api.get_credit_card_balance(access_token=api.access_token, card_id="12345")
        assert initial_balance["card_balance"] == 50000.0
        api.book_flight(
            access_token=api.access_token, card_id="12345",
            travel_date="2025-07-01", travel_from="JFK",
            travel_to="LHR", travel_class="business", travel_cost=800.0
        )
        after_balance = api.get_credit_card_balance(access_token=api.access_token, card_id="12345")
        assert after_balance["card_balance"] == 50000.0 - 800.0

    def test_authenticate_book_then_cancel(self):
        api = TravelBooking(initial_config=json.loads(json.dumps(TRAVEL_CONFIG)))
        api.authenticate_travel(
            client_id="c1", client_secret="s1",
            refresh_token="r1", grant_type="read_write",
            user_first_name="M", user_last_name="S"
        )
        book = api.book_flight(
            access_token=api.access_token, card_id="12345",
            travel_date="2025-08-01", travel_from="SFO",
            travel_to="ORD", travel_class="economy", travel_cost=400.0
        )
        cancel = api.cancel_booking(access_token=api.access_token, booking_id=book["booking_id"])
        assert cancel["cancel_status"] is True
        balance = api.get_credit_card_balance(access_token=api.access_token, card_id="12345")
        assert balance["card_balance"] == 50000.0

    def test_authenticate_book_then_purchase_insurance(self):
        api = TravelBooking(initial_config=json.loads(json.dumps(TRAVEL_CONFIG)))
        api.authenticate_travel(
            client_id="c1", client_secret="s1",
            refresh_token="r1", grant_type="read_write",
            user_first_name="M", user_last_name="S"
        )
        book = api.book_flight(
            access_token=api.access_token, card_id="12345",
            travel_date="2025-09-01", travel_from="SFO",
            travel_to="LAX", travel_class="first", travel_cost=1500.0
        )
        insurance = api.purchase_insurance(
            access_token=api.access_token, insurance_type="comprehensive",
            insurance_cost=200.0, booking_id=book["booking_id"], card_id="12345"
        )
        assert insurance["insurance_status"] is True

    def test_authenticate_then_register_card_then_book(self):
        api = TravelBooking(initial_config=json.loads(json.dumps(TRAVEL_CONFIG)))
        api.authenticate_travel(
            client_id="c1", client_secret="s1",
            refresh_token="r1", grant_type="read_write",
            user_first_name="M", user_last_name="S"
        )
        reg = api.register_credit_card(
            access_token=api.access_token, card_number="5555666677778888",
            expiration_date="12/2030", cardholder_name="Test User",
            card_verification_number=123
        )
        assert reg["card_id"] != ""
        fund_card_id = reg["card_id"]
        api.credit_card_list[fund_card_id]["balance"] = 10000.0
        book = api.book_flight(
            access_token=api.access_token, card_id=fund_card_id,
            travel_date="2025-10-01", travel_from="BOS",
            travel_to="CDG", travel_class="economy", travel_cost=500.0
        )
        assert book["booking_status"] is True

    def test_authenticate_then_set_budget(self):
        api = TravelBooking(initial_config=json.loads(json.dumps(TRAVEL_CONFIG)))
        api.authenticate_travel(
            client_id="c1", client_secret="s1",
            refresh_token="r1", grant_type="read_write",
            user_first_name="M", user_last_name="S"
        )
        budget = api.set_budget_limit(access_token=api.access_token, budget_limit=3000.0)
        assert budget["budget_limit"] == 3000.0

    def test_authenticate_book_then_retrieve_invoice(self):
        api = TravelBooking(initial_config=json.loads(json.dumps(TRAVEL_CONFIG)))
        api.authenticate_travel(
            client_id="c1", client_secret="s1",
            refresh_token="r1", grant_type="read_write",
            user_first_name="M", user_last_name="S"
        )
        book = api.book_flight(
            access_token=api.access_token, card_id="12345",
            travel_date="2025-11-01", travel_from="NYC",
            travel_to="ROM", travel_class="business", travel_cost=900.0
        )
        invoice = api.retrieve_invoice(access_token=api.access_token, booking_id=book["booking_id"])
        assert invoice["invoice"]["booking_id"] == book["booking_id"]


class TestTravelBookingSequentialProblematic:
    """Problematic sequences: operations with wrong/missing access token."""

    def test_book_without_auth(self):
        api = TravelBooking(initial_config=json.loads(json.dumps(TRAVEL_CONFIG)))
        assert api.access_token == ""
        book = api.book_flight(
            access_token="wrong_token", card_id="12345",
            travel_date="2025-06-01", travel_from="SFO",
            travel_to="LAX", travel_class="economy", travel_cost=300.0
        )
        assert book["booking_status"] is False

    def test_cancel_with_wrong_token(self):
        api = TravelBooking(initial_config=json.loads(json.dumps(TRAVEL_CONFIG)))
        api.authenticate_travel(
            client_id="c1", client_secret="s1",
            refresh_token="r1", grant_type="read_write",
            user_first_name="M", user_last_name="S"
        )
        book = api.book_flight(
            access_token=api.access_token, card_id="12345",
            travel_date="2025-07-01", travel_from="SFO",
            travel_to="LAX", travel_class="economy", travel_cost=300.0
        )
        cancel = api.cancel_booking(access_token="wrong_token", booking_id=book["booking_id"])
        assert cancel["cancel_status"] is False

    def test_get_balance_wrong_token(self):
        api = TravelBooking(initial_config=json.loads(json.dumps(TRAVEL_CONFIG)))
        api.authenticate_travel(
            client_id="c1", client_secret="s1",
            refresh_token="r1", grant_type="read_write",
            user_first_name="M", user_last_name="S"
        )
        balance = api.get_credit_card_balance(access_token="bad_token", card_id="12345")
        assert balance["card_balance"] == 0.0

    def test_purchase_insurance_wrong_token(self):
        api = TravelBooking(initial_config=json.loads(json.dumps(TRAVEL_CONFIG)))
        api.authenticate_travel(
            client_id="c1", client_secret="s1",
            refresh_token="r1", grant_type="read_write",
            user_first_name="M", user_last_name="S"
        )
        book = api.book_flight(
            access_token=api.access_token, card_id="12345",
            travel_date="2025-08-01", travel_from="SFO",
            travel_to="LAX", travel_class="economy", travel_cost=300.0
        )
        insurance = api.purchase_insurance(
            access_token="bad_token", insurance_type="basic",
            insurance_cost=50.0, booking_id=book["booking_id"], card_id="12345"
        )
        assert insurance["insurance_status"] is False

    def test_register_card_wrong_token(self):
        api = TravelBooking(initial_config=json.loads(json.dumps(TRAVEL_CONFIG)))
        reg = api.register_credit_card(
            access_token="bad_token", card_number="1234567890123456",
            expiration_date="12/2030", cardholder_name="Test",
            card_verification_number=123
        )
        assert reg["card_id"] == ""

    def test_set_budget_wrong_token(self):
        api = TravelBooking(initial_config=json.loads(json.dumps(TRAVEL_CONFIG)))
        budget = api.set_budget_limit(access_token="bad_token", budget_limit=5000.0)
        assert budget["budget_limit"] == 0.0

    def test_authenticate_invalid_grant_type(self):
        api = TravelBooking(initial_config=json.loads(json.dumps(TRAVEL_CONFIG)))
        auth = api.authenticate_travel(
            client_id="c1", client_secret="s1",
            refresh_token="r1", grant_type="invalid",
            user_first_name="M", user_last_name="S"
        )
        assert auth["access_token"] == ""
        book = api.book_flight(
            access_token="any_token", card_id="12345",
            travel_date="2025-01-01", travel_from="SFO",
            travel_to="LAX", travel_class="economy", travel_cost=300.0
        )
        assert book["booking_status"] is False

    def test_book_insufficient_balance(self):
        api = TravelBooking(initial_config=json.loads(json.dumps(TRAVEL_CONFIG)))
        api.authenticate_travel(
            client_id="c1", client_secret="s1",
            refresh_token="r1", grant_type="read_write",
            user_first_name="M", user_last_name="S"
        )
        book = api.book_flight(
            access_token=api.access_token, card_id="12345",
            travel_date="2025-01-01", travel_from="SFO",
            travel_to="LAX", travel_class="first", travel_cost=99999.0
        )
        assert book["booking_status"] is False

    def test_cancel_nonexistent_booking(self):
        api = TravelBooking(initial_config=json.loads(json.dumps(TRAVEL_CONFIG)))
        api.authenticate_travel(
            client_id="c1", client_secret="s1",
            refresh_token="r1", grant_type="read_write",
            user_first_name="M", user_last_name="S"
        )
        cancel = api.cancel_booking(access_token=api.access_token, booking_id="fake_001")
        assert cancel["cancel_status"] is False


# ═══════════════════════════════════════════════════════════════════════════
# TicketAPI sequential tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTicketAPISequentialCorrect:
    """Correct sequences: login → create → edit → resolve → get."""

    def test_login_then_create_ticket(self):
        config = json.loads(json.dumps(TICKET_API_CONFIG))
        config["current_user"] = ""
        api = TicketAPI(initial_config=config)
        login = api.ticket_login(username="agent_a", password="testpass")
        assert login["success"] is True
        assert api.current_user == "agent_a"
        ticket = api.create_ticket(title="New bug", description="App crashes on startup", priority=3)
        assert ticket["title"] == "New bug"
        assert ticket["status"] == "Open"

    def test_login_create_then_edit(self):
        api = TicketAPI(initial_config=json.loads(json.dumps(TICKET_API_CONFIG)))
        api.ticket_login(username="agent_a", password="testpass")
        ticket = api.create_ticket(title="Login issue", description="Users cannot log in", priority=2)
        edit = api.edit_ticket(ticket_id=ticket["id"], updates={"priority": 4})
        assert edit["status"] == "Ticket updated successfully"
        fetched = api.get_ticket(ticket_id=ticket["id"])
        assert fetched["priority"] == 4

    def test_login_create_then_resolve(self):
        api = TicketAPI(initial_config=json.loads(json.dumps(TICKET_API_CONFIG)))
        api.ticket_login(username="agent_a", password="testpass")
        ticket = api.create_ticket(title="Server down", description="Primary server unresponsive", priority=5)
        resolve = api.resolve_ticket(ticket_id=ticket["id"], resolution="Restarted server")
        assert resolve["status"] == "Ticket resolved successfully"
        fetched = api.get_ticket(ticket_id=ticket["id"])
        assert fetched["status"] == "Resolved"

    def test_login_create_then_close(self):
        api = TicketAPI(initial_config=json.loads(json.dumps(TICKET_API_CONFIG)))
        api.ticket_login(username="agent_a", password="testpass")
        ticket = api.create_ticket(title="Minor glitch", description="UI flicker", priority=1)
        close = api.close_ticket(ticket_id=ticket["id"])
        assert close["status"] == "Ticket closed successfully"

    def test_login_create_then_get_user_tickets(self):
        api = TicketAPI(initial_config=json.loads(json.dumps(TICKET_API_CONFIG)))
        api.ticket_login(username="agent_a", password="testpass")
        api.create_ticket(title="Bug 1", description="First bug", priority=2)
        user_tickets = api.get_user_tickets()
        assert user_tickets["created_by"] == "agent_a"

    def test_get_ticket_then_resolve_existing(self):
        api = TicketAPI(initial_config=json.loads(json.dumps(TICKET_API_CONFIG)))
        api.ticket_login(username="agent_a", password="testpass")
        existing = api.get_ticket(ticket_id=123456)
        assert existing["id"] == 123456
        resolve = api.resolve_ticket(ticket_id=123456, resolution="Fixed the system error")
        assert resolve["status"] == "Ticket resolved successfully"

    def test_login_create_edit_resolve_full_lifecycle(self):
        api = TicketAPI(initial_config=json.loads(json.dumps(TICKET_API_CONFIG)))
        api.ticket_login(username="agent_a", password="testpass")
        ticket = api.create_ticket(title="Full lifecycle", description="Test full flow", priority=2)
        api.edit_ticket(ticket_id=ticket["id"], updates={"priority": 4, "description": "Updated desc"})
        resolve = api.resolve_ticket(ticket_id=ticket["id"], resolution="All good now")
        assert resolve["status"] == "Ticket resolved successfully"
        fetched = api.get_ticket(ticket_id=ticket["id"])
        assert fetched["status"] == "Resolved"
        assert fetched["priority"] == 4


class TestTicketAPISequentialProblematic:
    """Problematic sequences for TicketAPI."""

    def test_create_ticket_without_login(self):
        config = json.loads(json.dumps(TICKET_API_CONFIG))
        config["current_user"] = ""
        api = TicketAPI(initial_config=config)
        ticket = api.create_ticket(title="No login ticket", description="Created without login")
        assert ticket["title"] == "No login ticket"
        assert ticket.get("created_by", "") == ""

    def test_login_empty_then_create(self):
        config = json.loads(json.dumps(TICKET_API_CONFIG))
        config["current_user"] = ""
        api = TicketAPI(initial_config=config)
        login = api.ticket_login(username="", password="")
        assert login["success"] is False
        ticket = api.create_ticket(title="After failed login", description="Should still work but with empty user")
        assert ticket["title"] == "After failed login"

    def test_edit_nonexistent_ticket(self):
        api = TicketAPI(initial_config=json.loads(json.dumps(TICKET_API_CONFIG)))
        edit = api.edit_ticket(ticket_id=99999, updates={"priority": 5})
        assert "not found" in edit["status"].lower()

    def test_resolve_nonexistent_ticket(self):
        api = TicketAPI(initial_config=json.loads(json.dumps(TICKET_API_CONFIG)))
        resolve = api.resolve_ticket(ticket_id=99999, resolution="N/A")
        assert "not found" in resolve["status"].lower()

    def test_close_nonexistent_ticket(self):
        api = TicketAPI(initial_config=json.loads(json.dumps(TICKET_API_CONFIG)))
        close = api.close_ticket(ticket_id=99999)
        assert "not found" in close["status"].lower()

    def test_get_user_tickets_no_matching_status(self):
        api = TicketAPI(initial_config=json.loads(json.dumps(TICKET_API_CONFIG)))
        api.ticket_login(username="agent_a", password="testpass")
        result = api.get_user_tickets(status="Closed")
        assert result["id"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Cross-API sequential tests (using ToolManager-style instantiation)
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossAPISequentialCorrect:
    """Sequences that span multiple APIs (realistic multi-tool workflows)."""

    def test_message_login_then_ticket_create(self):
        """User logs into messaging, then creates a support ticket."""
        msg_api = MessageAPI(initial_config=json.loads(json.dumps(MESSAGE_API_CONFIG)))
        tkt_api = TicketAPI(initial_config=json.loads(json.dumps(TICKET_API_CONFIG)))

        login = msg_api.message_login(user_id="USR005")
        assert login["login_status"] is True

        tkt_api.ticket_login(username="agent_a", password="pass")
        ticket = tkt_api.create_ticket(title="Message system issue", description="Cannot send messages", priority=3)
        assert ticket["status"] == "Open"

    def test_authenticate_travel_then_message_about_booking(self):
        """User books a flight then sends a message about it."""
        travel = TravelBooking(initial_config=json.loads(json.dumps(TRAVEL_CONFIG)))
        msg = MessageAPI(initial_config=json.loads(json.dumps(MESSAGE_API_CONFIG)))

        travel.authenticate_travel(
            client_id="c1", client_secret="s1",
            refresh_token="r1", grant_type="read_write",
            user_first_name="Michael", user_last_name="Smith"
        )
        book = travel.book_flight(
            access_token=travel.access_token, card_id="12345",
            travel_date="2025-06-01", travel_from="SFO",
            travel_to="LAX", travel_class="economy", travel_cost=300.0
        )
        assert book["booking_status"] is True

        msg.message_login(user_id="USR005")
        sent = msg.send_message(
            receiver_id="USR006",
            message=f"Booked flight {book['booking_id']} to LAX!"
        )
        assert sent["sent_status"] is True

    def test_trading_buy_then_post_about_it(self):
        """User buys stock then tweets about it."""
        bot = TradingBot(initial_config=json.loads(json.dumps(TRADING_BOT_CONFIG)))
        posting = PostingAPI(initial_config=json.loads(json.dumps(POSTING_API_CONFIG)))

        bot.trading_login(username="trader", password="pass")
        order = bot.place_order(order_type="Buy", symbol="AAPL", price=227.16, amount=10)
        assert order["order_id"] is not None

        posting.authenticate_twitter(username="genealogy_enthusiast", password="testpass")
        tweet = posting.post_tweet(
            content=f"Just bought 10 shares of AAPL!",
            tags=["#trading", "#AAPL"]
        )
        assert tweet["id"] != 0
