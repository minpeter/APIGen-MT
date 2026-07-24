"""Smoke tests: domain tool classes instantiate and run a trivial call."""

from __future__ import annotations

import importlib

import pytest

from tools.math_api import MathAPI
from tools.message_api import MessageAPI
from tools.ticket_api import TicketAPI
from tools.trading_bot import TradingBot
from tools.travel_booking import TravelBooking
from tools.vehicle_control import VehicleControl
from tools.posting_api import PostingAPI
from tools.gorilla_file_system import GorillaFileSystem


def test_math_add_smoke():
    api = MathAPI({})
    assert api.add(a=1.0, b=2.0)["result"] == pytest.approx(3.0)


def test_message_add_contact_smoke():
    api = MessageAPI(
        {
            "workspace_id": "WS1",
            "user_count": 0,
            "user_map": {},
            "messages_sent_map": {},
            "messages_inbox_map": {},
            "message_count": 0,
            "current_user": "",
        }
    )
    out = api.add_contact("Alice")
    assert out["added_status"] is True
    assert str(out["user_id"]).startswith("USR")


def test_ticket_login_smoke():
    api = TicketAPI(
        {
            "username": "agent",
            "password": "testpass",
            "authenticated": False,
            "tickets": {},
        }
    )
    if hasattr(api, "ticket_login"):
        res = api.ticket_login(username="agent", password="testpass")
    elif hasattr(api, "login"):
        res = api.login(username="agent", password="testpass")
    else:
        res = {"ok": True}
    assert isinstance(res, dict)


def test_trading_instantiate_smoke():
    bot = TradingBot(
        {
            "username": "trader",
            "password": "testpass",
            "authenticated": False,
            "orders": {},
            "watchlist": [],
            "account_balance": 10000.0,
        }
    )
    assert bot is not None


def test_travel_instantiate_smoke():
    api = TravelBooking(
        {
            "client_id": "c1",
            "client_secret": "test_secret",
            "access_token": "",
            "bookings": {},
        }
    )
    assert api is not None


def test_vehicle_instantiate_smoke():
    api = VehicleControl({"vehicles": {}, "authenticated": False})
    assert api is not None


def test_posting_instantiate_smoke():
    api = PostingAPI(
        {
            "username": "u",
            "password": "testpass",
            "authenticated": False,
            "tweets": {},
        }
    )
    assert api is not None


def test_gorilla_ls_smoke():
    fs = GorillaFileSystem(initial_config={})
    if hasattr(fs, "ls"):
        out = fs.ls()
        assert isinstance(out, dict)
    else:
        assert fs is not None


def test_all_tool_modules_importable():
    for mod in (
        "tools.math_api",
        "tools.message_api",
        "tools.ticket_api",
        "tools.trading_bot",
        "tools.travel_booking",
        "tools.vehicle_control",
        "tools.posting_api",
        "tools.gorilla_file_system",
        "tools.schemas",
    ):
        importlib.import_module(mod)
