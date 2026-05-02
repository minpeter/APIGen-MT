import pytest
import json
from tools.trading_bot import TradingBot

INITIAL_CONFIG = {
    "account_info": {
        "account_id": 12345,
        "balance": 10000.0,
        "binding_card": 1974202140965533
    },
    "authenticated": True,
    "market_status": "Open",
    "order_counter": 12446,
    "stocks": {
        "AAPL": {
            "price": 227.16,
            "percent_change": 0.17,
            "volume": 2.552,
            "MA(5)": 227.11,
            "MA(20)": 227.09
        },
        "GOOG": {
            "price": 2840.34,
            "percent_change": 0.24,
            "volume": 1.123,
            "MA(5)": 2835.67,
            "MA(20)": 2842.15
        },
        "TSLA": {
            "price": 667.92,
            "percent_change": -0.12,
            "volume": 1.654,
            "MA(5)": 671.15,
            "MA(20)": 668.2
        },
        "MSFT": {
            "price": 310.23,
            "percent_change": 0.09,
            "volume": 3.234,
            "MA(5)": 309.88,
            "MA(20)": 310.11
        },
        "NVDA": {
            "price": 220.34,
            "percent_change": 0.34,
            "volume": 1.234,
            "MA(5)": 220.45,
            "MA(20)": 220.67
        },
        "ALPH": {
            "price": 1320.45,
            "percent_change": -0.08,
            "volume": 1.567,
            "MA(5)": 1321.12,
            "MA(20)": 1325.78
        },
        "OMEG": {
            "price": 457.23,
            "percent_change": 0.12,
            "volume": 2.345,
            "MA(5)": 456.78,
            "MA(20)": 458.12
        },
        "QUAS": {
            "price": 725.89,
            "percent_change": -0.03,
            "volume": 1.789,
            "MA(5)": 726.45,
            "MA(20)": 728.0
        },
        "NEPT": {
            "price": 88.34,
            "percent_change": 0.19,
            "volume": 0.654,
            "MA(5)": 88.21,
            "MA(20)": 88.67
        },
        "SYNX": {
            "price": 345.67,
            "percent_change": 0.11,
            "volume": 2.112,
            "MA(5)": 345.34,
            "MA(20)": 346.12
        },
        "ZETA": {
            "price": 150.45,
            "percent_change": 0.05,
            "volume": 1.789,
            "MA(5)": 150.0,
            "MA(20)": 149.5
        }
    },
    "watch_list": [
        "NVDA",
        "ZETA"
    ],
    "transaction_history": [
        {
            "order_id": 12346,
            "symbol": "GOOG",
            "price": 2840.34,
            "num_shares": 5,
            "status": "Pending",
            "timestamp": "2024-10-27 14:10:53"
        }
    ]
}


@pytest.fixture
def trading_bot():
    return TradingBot(initial_config=INITIAL_CONFIG)


def test_add_to_watchlist_normal(trading_bot):
    result = trading_bot.add_to_watchlist(stock='OMEG')
    assert "OMEG" in trading_bot.watch_list


def test_add_to_watchlist_already_exists(trading_bot):
    result = trading_bot.add_to_watchlist(stock='NVDA')
    assert "NVDA" in trading_bot.watch_list


def test_add_to_watchlist_invalid_stock(trading_bot):
    result = trading_bot.add_to_watchlist(stock='INVALID')
    assert "INVALID" in result.get("symbol", "")


def test_cancel_order_normal(trading_bot):
    result = trading_bot.cancel_order(order_id=12346)
    assert result.get("order_id") == 12346


def test_cancel_order_not_found(trading_bot):
    result = trading_bot.cancel_order(order_id=99999)
    assert result.get("status") == "Order not found"


def test_cancel_order_already_cancelled(trading_bot):
    trading_bot.cancel_order(order_id=12346)
    result = trading_bot.cancel_order(order_id=12346)
    assert result.get("status") == "Cancelled"


def test_filter_stocks_by_price_normal(trading_bot):
    result = trading_bot.filter_stocks_by_price(stocks=["AAPL", "TSLA", "MSFT"], min_price=200.0, max_price=500.0)
    filtered = result.get("filtered_stocks", [])
    assert "AAPL" in filtered
    assert "TSLA" not in filtered


def test_filter_stocks_by_price_no_match(trading_bot):
    result = trading_bot.filter_stocks_by_price(stocks=["AAPL", "MSFT"], min_price=1000.0, max_price=2000.0)
    filtered = result.get("filtered_stocks", [])
    assert len(filtered) == 0


def test_filter_stocks_by_price_invalid_range(trading_bot):
    result = trading_bot.filter_stocks_by_price(stocks=["AAPL"], min_price=500.0, max_price=100.0)
    filtered = result.get("filtered_stocks", [])
    assert len(filtered) == 0


def test_fund_account_normal(trading_bot):
    initial_balance = trading_bot.account_info["balance"]
    result = trading_bot.fund_account(amount=2203.4)
    assert trading_bot.account_info["balance"] == initial_balance + 2203.4


def test_fund_account_zero(trading_bot):
    result = trading_bot.fund_account(amount=0)
    assert "Failed" in result.get("status", "") or result.get("new_balance") == trading_bot.account_info["balance"]


def test_fund_account_negative(trading_bot):
    result = trading_bot.fund_account(amount=-500.0)
    assert "Failed" in result.get("status", "")


def test_get_available_stocks_normal(trading_bot):
    result = trading_bot.get_available_stocks(sector='Technology')
    assert isinstance(result.get("stock_list", []), list)


def test_get_available_stocks_invalid_sector(trading_bot):
    result = trading_bot.get_available_stocks(sector='NonExistentSector')
    assert len(result.get("stock_list", [])) == 0


def test_get_available_stocks_case_insensitive(trading_bot):
    result = trading_bot.get_available_stocks(sector='technology')
    assert isinstance(result.get("stock_list", []), list) and len(result.get("stock_list", [])) == 0


def test_get_order_details_normal(trading_bot):
    result = trading_bot.get_order_details(order_id=12346)
    assert result.get("id") == 12346
    assert result.get("status") == "Pending"


def test_get_order_details_not_found(trading_bot):
    result = trading_bot.get_order_details(order_id=99999)
    assert result.get("status") == "Order not found"


def test_get_order_details_invalid_id(trading_bot):
    result = trading_bot.get_order_details(order_id=-1)
    assert result.get("status") == "Order not found"


def test_get_stock_info_normal(trading_bot):
    result = trading_bot.get_stock_info(symbol='NVDA')
    assert result.get("price") == 220.34


def test_get_stock_info_not_found(trading_bot):
    result = trading_bot.get_stock_info(symbol='XTC')
    assert result.get("price") == 0.0


def test_get_stock_info_empty_symbol(trading_bot):
    result = trading_bot.get_stock_info(symbol='')
    assert result.get("price") == 0.0


def test_get_symbol_by_name_normal(trading_bot):
    result = trading_bot.get_symbol_by_name(name='Nvidia')
    assert result.get("symbol") == "NVDA"


def test_get_symbol_by_name_not_found(trading_bot):
    result = trading_bot.get_symbol_by_name(name='NonExistentCorp')
    assert result.get("symbol") == "Stock not found"


def test_get_symbol_by_name_case_insensitive(trading_bot):
    result = trading_bot.get_symbol_by_name(name='nvidia')
    assert result.get("symbol") == "NVDA"


def test_get_transaction_history_normal(trading_bot):
    result = trading_bot.get_transaction_history()
    assert isinstance(result.get("transaction_history", []), list)


def test_get_transaction_history_with_dates(trading_bot):
    result = trading_bot.get_transaction_history(start_date='2024-10-01', end_date='2024-10-31')
    assert isinstance(result.get("transaction_history", []), list)


def test_get_transaction_history_invalid_dates(trading_bot):
    result = trading_bot.get_transaction_history(start_date='invalid-date', end_date='2024-10-31')
    assert isinstance(result.get("transaction_history", []), list)


def test_make_transaction_withdrawal_normal(trading_bot):
    initial_balance = trading_bot.account_info["balance"]
    result = trading_bot.make_transaction(account_id=12345, xact_type='withdrawal', amount=500)
    assert trading_bot.account_info["balance"] == initial_balance - 500


def test_make_transaction_deposit_normal(trading_bot):
    initial_balance = trading_bot.account_info["balance"]
    result = trading_bot.make_transaction(account_id=12345, xact_type='deposit', amount=500)
    assert trading_bot.account_info["balance"] == initial_balance + 500


def test_make_transaction_invalid_account(trading_bot):
    result = trading_bot.make_transaction(account_id=99999, xact_type='withdrawal', amount=500)
    assert "Failed" in result.get("status", "")


def test_notify_price_change_triggered(trading_bot):
    result = trading_bot.notify_price_change(stocks=["NVDA"], threshold=0.1)
    assert "NVDA" in result.get("notification", "")


def test_notify_price_change_not_triggered(trading_bot):
    result = trading_bot.notify_price_change(stocks=["AAPL"], threshold=0.5)
    assert "No significant" in result.get("notification", "")


def test_notify_price_change_invalid_stock(trading_bot):
    result = trading_bot.notify_price_change(stocks=["INVALID"], threshold=0.1)
    assert "No significant" in result.get("notification", "")


def test_place_order_buy_normal(trading_bot):
    result = trading_bot.place_order(order_type='Buy', symbol='TSLA', price=700, amount=100)
    assert result.get("order_id") is not None


def test_place_order_sell_normal(trading_bot):
    result = trading_bot.place_order(order_type='Sell', symbol='AAPL', price=227.16, amount=10)
    assert result.get("order_id") is not None


def test_place_order_invalid_symbol(trading_bot):
    result = trading_bot.place_order(order_type='Buy', symbol='INVALID', price=100, amount=10)
    assert result.get("order_id") is not None


def test_remove_stock_from_watchlist_normal(trading_bot):
    result = trading_bot.remove_stock_from_watchlist(symbol='ZETA')
    assert "ZETA" not in trading_bot.watch_list


def test_remove_stock_from_watchlist_not_exists(trading_bot):
    result = trading_bot.remove_stock_from_watchlist(symbol='AAPL')
    assert "not in watchlist" in result.get("status", "")


def test_remove_stock_from_watchlist_invalid_symbol(trading_bot):
    result = trading_bot.remove_stock_from_watchlist(symbol='INVALID')
    assert "removed from watchlist" in result.get("status", "")


def test_trading_login_normal(trading_bot):
    result = trading_bot.trading_login(username='user', password='pass')
    assert result.get("status") == "Login successful"
    assert trading_bot.authenticated is True


def test_trading_login_invalid(trading_bot):
    result = trading_bot.trading_login(username='invalid', password='invalid')
    assert result.get("status") == "Login successful"


def test_trading_login_empty_credentials(trading_bot):
    result = trading_bot.trading_login(username='', password='')
    assert result.get("status") == "Login successful"


def test_update_market_status_open(trading_bot):
    result = trading_bot.update_market_status(current_time_str='10:30 AM')
    assert result.get("status") == "Open"


def test_update_market_status_closed(trading_bot):
    result = trading_bot.update_market_status(current_time_str='5:00 PM')
    assert result.get("status") == "Closed"


def test_update_market_status_invalid_time(trading_bot):
    result = trading_bot.update_market_status(current_time_str='invalid_time')
    assert result.get("status") == "Closed"


def test_update_stock_price_normal(trading_bot):
    result = trading_bot.update_stock_price(symbol='AAPL', new_price=230.00)
    assert trading_bot.stocks["AAPL"]["price"] == 230.00


def test_update_stock_price_invalid_symbol(trading_bot):
    result = trading_bot.update_stock_price(symbol='INVALID', new_price=100.00)
    assert result.get("old_price") == 0.0


def test_update_stock_price_negative_price(trading_bot):
    result = trading_bot.update_stock_price(symbol='AAPL', new_price=-50.00)
    assert result.get("new_price") == -50.00
