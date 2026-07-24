"""Stock, market, and watchlist operations for TradingBot."""

from .trading_bot_state import TradingBotState
from .trading_bot_types import (
    FilteredStocksResult,
    NotificationResult,
    StockInfoResult,
    StockListResult,
    SymbolResult,
    TradingStatusResult,
    UpdatePriceResult,
    WatchlistAddResult,
)


class TradingBotMarketMixin(TradingBotState):
    """Provide market data, pricing, and watchlist operations."""

    def add_to_watchlist(self, stock: str) -> WatchlistAddResult:
        """Add a stock to the watchlist."""
        if not self.authenticated:
            return {
                "symbol": "",
                "watchlist_status": "User not authenticated",
            }
        if stock and stock not in self.watch_list:
            self.watch_list.append(stock)
        return {"symbol": stock}

    def filter_stocks_by_price(
        self, stocks: list[str], min_price: float, max_price: float
    ) -> FilteredStocksResult:
        """Filter stocks based on a price range."""
        filtered = [
            symbol
            for symbol in stocks
            if symbol in self.stocks
            and min_price
            <= self._number(self.stocks[symbol], "price")
            <= max_price
        ]
        return {"filtered_stocks": filtered}

    def get_available_stocks(self, sector: str) -> StockListResult:
        """Get stock symbols in the given sector."""
        return {"stock_list": self.sector_mapping.get(sector, [])}

    def get_stock_info(self, symbol: str) -> StockInfoResult:
        """Get market details for a stock."""
        stock = self.stocks.get(symbol, {})
        return {
            "price": self._number(stock, "price"),
            "percent_change": self._number(stock, "percent_change"),
            "volume": self._number(stock, "volume"),
            "MA(5)": self._number(stock, "MA(5)"),
            "MA(20)": self._number(stock, "MA(20)"),
        }

    def get_symbol_by_name(self, name: str) -> SymbolResult:
        """Get a stock symbol by company name."""
        return {
            "symbol": self.name_to_symbol.get(name.lower(), "Stock not found")
        }

    def notify_price_change(
        self, stocks: list[str], threshold: float
    ) -> NotificationResult:
        """Notify when requested stocks exceed a change threshold."""
        notifications: list[str] = []
        for symbol in stocks:
            stock = self.stocks.get(symbol)
            if stock is None:
                continue
            percent_change = self._number(stock, "percent_change")
            if abs(percent_change) >= threshold:
                notifications.append(f"{symbol} changed by {percent_change}%")
        if notifications:
            return {
                "notification": "Significant changes: "
                + "; ".join(notifications)
            }
        return {"notification": "No significant price changes."}

    def remove_stock_from_watchlist(self, symbol: str) -> TradingStatusResult:
        """Remove a stock from the watchlist."""
        if not self.authenticated:
            return {"status": "User not authenticated"}
        if symbol in self.watch_list:
            self.watch_list.remove(symbol)
            return {"status": f"{symbol} removed from watchlist"}
        return {"status": f"{symbol} not in watchlist"}

    def update_market_status(self, current_time_str: str) -> TradingStatusResult:
        """Update market status from a 12-hour clock value."""
        market_status = "Closed"
        try:
            clock, period = current_time_str.strip().upper().split()
            hour_text, minute_text = clock.split(":")
            hour = int(hour_text)
            minute = int(minute_text)
            if period not in {"AM", "PM"} or not 1 <= hour <= 12:
                raise ValueError
            if not 0 <= minute <= 59:
                raise ValueError
            hour = hour % 12 + (12 if period == "PM" else 0)
            total_minutes = hour * 60 + minute
            market_status = (
                "Open"
                if 9 * 60 + 30 <= total_minutes <= 16 * 60
                else "Closed"
            )
        except ValueError:
            self.market_status = "Closed"
            return {"status": self.market_status}
        self.market_status: str = market_status
        return {"status": self.market_status}

    def update_stock_price(
        self, symbol: str, new_price: float
    ) -> UpdatePriceResult:
        """Update the price of a stock."""
        if not self.authenticated:
            return {
                "symbol": symbol,
                "old_price": 0.0,
                "new_price": 0.0,
                "status": "User not authenticated",
            }
        stock = self.stocks.get(symbol)
        if stock is None:
            return {"symbol": symbol, "old_price": 0.0, "new_price": 0.0}
        old_price = self._number(stock, "price")
        stock["price"] = new_price
        return {
            "symbol": symbol,
            "old_price": old_price,
            "new_price": new_price,
        }
