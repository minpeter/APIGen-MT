"""Typed public results returned by TradingBot operations."""

from typing import NotRequired, TypedDict


class TradingStatusResult(TypedDict):
    """Result carrying an operation status."""

    status: str


class WatchlistAddResult(TypedDict):
    """Result from adding a stock to a watchlist."""

    symbol: str
    watchlist_status: NotRequired[str]


class CancelOrderResult(TypedDict):
    """Result from cancelling an order."""

    order_id: int
    status: str


class BalanceResult(TypedDict):
    """Result from changing an account balance."""

    status: str
    new_balance: float


class OrderDetailsResult(TypedDict):
    """Details of a placed order."""

    id: int
    order_type: str
    symbol: str
    price: float
    amount: int
    status: str


class TransactionHistoryItem(TypedDict):
    """One completed trading transaction."""

    type: str
    symbol: str
    total_cost: float
    timestamp: str


class TransactionHistoryResult(TypedDict):
    """Transactions in a requested date range."""

    transaction_history: list[TransactionHistoryItem]
    status: NotRequired[str]


class PlaceOrderResult(TypedDict):
    """Result from placing an order."""

    order_id: int
    order_type: str
    status: str
    price: float
    amount: int


class TradingLoginResult(TypedDict):
    """Result from authenticating with the trading API."""

    status: str


class FilteredStocksResult(TypedDict):
    """Stock symbols that fall within a price range."""

    filtered_stocks: list[str]


class StockListResult(TypedDict):
    """Stock symbols available in a sector."""

    stock_list: list[str]


StockInfoResult = TypedDict(
    "StockInfoResult",
    {
        "price": float,
        "percent_change": float,
        "volume": float,
        "MA(5)": float,
        "MA(20)": float,
    },
)


class SymbolResult(TypedDict):
    """Ticker symbol corresponding to a company name."""

    symbol: str


class NotificationResult(TypedDict):
    """Price-change notification text."""

    notification: str


class UpdatePriceResult(TypedDict):
    """Result from changing a stock price."""

    symbol: str
    old_price: float
    new_price: float
    status: NotRequired[str]
