"""Auto-generated TradingBot implementation."""

import json
import math
import re
import copy
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


class TradingBot:
    """A trading bot that allows users to trade stocks, manage their account, and view stock information."""

    def __init__(self, initial_config: dict) -> None:
        """Initialize the trading bot with the provided configuration."""
        self.account_info: dict = initial_config.get("account_info", {
            "account_id": 0,
            "balance": 0.0,
            "binding_card": 0
        })
        self.authenticated: bool = initial_config.get("authenticated", False)
        self.current_user: str = initial_config.get("current_user", "")
        self.market_status: str = initial_config.get("market_status", "Closed")
        self.order_counter: int = initial_config.get("order_counter", 0)
        self.stocks: Dict[str, Dict[str, Any]] = initial_config.get("stocks", {})
        self.watch_list: List[str] = initial_config.get("watch_list", [])
        
        raw_history = initial_config.get("transaction_history", [])
        self.transaction_history: List[Dict[str, Any]] = []
        for item in raw_history:
            clean_item = {k: v for k, v in item.items() if k != "order_id"}
            self.transaction_history.append(clean_item)
            
        self.orders: Dict[int, Dict[str, Any]] = {}
        for item in raw_history:
            if "order_id" in item:
                order_id = item["order_id"]
                self.orders[order_id] = {
                    "id": order_id,
                    "order_type": item.get("order_type", "Buy"),
                    "symbol": item.get("symbol", ""),
                    "price": item.get("price", 0.0),
                    "amount": item.get("num_shares", item.get("amount", 0)),
                    "status": item.get("status", "Pending")
                }
                
        self.sector_mapping: Dict[str, List[str]] = {
            "Technology": ["AAPL", "GOOG", "MSFT", "NVDA"],
            "Automotive": ["TSLA"],
            "Finance": ["ALPH", "OMEG"],
            "Energy": ["QUAS", "NEPT"],
            "Healthcare": ["SYNX", "ZETA"]
        }
        
        self.name_to_symbol: Dict[str, str] = {
            "apple": "AAPL",
            "apple inc": "AAPL",
            "apple inc.": "AAPL",
            "google": "GOOG",
            "google llc": "GOOG",
            "tesla": "TSLA",
            "tesla inc": "TSLA",
            "tesla inc.": "TSLA",
            "microsoft": "MSFT",
            "microsoft corporation": "MSFT",
            "nvidia": "NVDA",
            "nvidia corporation": "NVDA",
            "alpha": "ALPH",
            "omega": "OMEG",
            "quasar": "QUAS",
            "neptune": "NEPT",
            "synx": "SYNX",
            "zeta": "ZETA"
        }

    def add_to_watchlist(self, stock: str) -> Dict[str, str]:
        """Add a stock to the watchlist."""
        if stock and stock not in self.watch_list:
            self.watch_list.append(stock)
        return {"symbol": stock}

    def cancel_order(self, order_id: int) -> Dict[str, Any]:
        """Cancel an order."""
        if order_id in self.orders:
            self.orders[order_id]["status"] = "Cancelled"
            return {"order_id": order_id, "status": "Cancelled"}
        return {"order_id": order_id, "status": "Order not found"}

    def filter_stocks_by_price(self, stocks: List[str], min_price: float, max_price: float) -> Dict[str, List[str]]:
        """Filter stocks based on a price range."""
        filtered = []
        for symbol in stocks:
            if symbol in self.stocks:
                price = self.stocks[symbol].get("price", 0.0)
                if min_price <= price <= max_price:
                    filtered.append(symbol)
        return {"filtered_stocks": filtered}

    def fund_account(self, amount: float) -> Dict[str, Any]:
        """Fund the account with the specified amount."""
        if amount <= 0:
            return {"status": "Failed: invalid amount", "new_balance": self.account_info.get("balance", 0.0)}
        self.account_info["balance"] = self.account_info.get("balance", 0.0) + amount
        return {"status": "Account funded successfully", "new_balance": self.account_info["balance"]}

    def get_available_stocks(self, sector: str) -> Dict[str, List[str]]:
        """Get a list of stock symbols in the given sector."""
        stock_list = self.sector_mapping.get(sector, [])
        return {"stock_list": stock_list}

    def get_order_details(self, order_id: int) -> Dict[str, Any]:
        """Get the details of an order."""
        if order_id in self.orders:
            order = self.orders[order_id]
            return {
                "id": order.get("id", order_id),
                "order_type": order.get("order_type", ""),
                "symbol": order.get("symbol", ""),
                "price": order.get("price", 0.0),
                "amount": order.get("amount", 0),
                "status": order.get("status", "")
            }
        return {
            "id": order_id,
            "order_type": "",
            "symbol": "",
            "price": 0.0,
            "amount": 0,
            "status": "Order not found"
        }

    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get the details of a stock."""
        if symbol in self.stocks:
            stock = self.stocks[symbol]
            return {
                "price": stock.get("price", 0.0),
                "percent_change": stock.get("percent_change", 0.0),
                "volume": stock.get("volume", 0.0),
                "MA(5)": stock.get("MA(5)", 0.0),
                "MA(20)": stock.get("MA(20)", 0.0)
            }
        return {
            "price": 0.0,
            "percent_change": 0.0,
            "volume": 0.0,
            "MA(5)": 0.0,
            "MA(20)": 0.0
        }

    def get_symbol_by_name(self, name: str) -> Dict[str, str]:
        """Get the symbol of a stock by company name."""
        symbol = self.name_to_symbol.get(name.lower(), "Stock not found")
        return {"symbol": symbol}

    def get_transaction_history(self, start_date: str = 'None', end_date: str = 'None') -> Dict[str, Any]:
        """Get the transaction history within a specified date range."""
        result = []
        for t in self.transaction_history:
            ts_str = t.get("timestamp", "")
            try:
                ts_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                continue
            
            include = True
            if start_date and start_date != 'None':
                try:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    if ts_dt < start_dt:
                        include = False
                except ValueError:
                    pass
            if end_date and end_date != 'None':
                try:
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    if ts_dt > end_dt:
                        include = False
                except ValueError:
                    pass
            
            if include:
                result.append({
                    "type": t.get("type", ""),
                    "amount": t.get("amount", 0.0),
                    "timestamp": ts_str
                })
        return {"transaction_history": result}

    def make_transaction(self, account_id: int, xact_type: str, amount: float) -> Dict[str, Any]:
        """Make a deposit or withdrawal based on specified amount."""
        if account_id != self.account_info.get("account_id"):
            return {"status": "Failed: account not found", "new_balance": self.account_info.get("balance", 0.0)}
        
        if xact_type == "deposit":
            if amount <= 0:
                return {"status": "Failed: invalid amount", "new_balance": self.account_info.get("balance", 0.0)}
            self.account_info["balance"] = self.account_info.get("balance", 0.0) + amount
            status_msg = "Deposit successful"
        elif xact_type == "withdrawal":
            if amount <= 0 or amount > self.account_info.get("balance", 0.0):
                return {"status": "Failed: invalid amount or insufficient funds", "new_balance": self.account_info.get("balance", 0.0)}
            self.account_info["balance"] = self.account_info.get("balance", 0.0) - amount
            status_msg = "Withdrawal successful"
        else:
            return {"status": "Failed: invalid transaction type", "new_balance": self.account_info.get("balance", 0.0)}
            
        return {"status": status_msg, "new_balance": self.account_info["balance"]}

    def notify_price_change(self, stocks: List[str], threshold: float) -> Dict[str, str]:
        """Notify if there is a significant price change in the stocks."""
        notifications = []
        for symbol in stocks:
            if symbol in self.stocks:
                pct_change = abs(self.stocks[symbol].get("percent_change", 0.0))
                if pct_change >= threshold:
                    notifications.append(f"{symbol} changed by {self.stocks[symbol].get('percent_change', 0.0)}%")
        
        if notifications:
            return {"notification": "Significant changes: " + "; ".join(notifications)}
        return {"notification": "No significant price changes."}

    def place_order(self, order_type: str, symbol: str, price: float, amount: int) -> Dict[str, Any]:
        """Place an order."""
        self.order_counter += 1
        order_id = self.order_counter
        self.orders[order_id] = {
            "id": order_id,
            "order_type": order_type,
            "symbol": symbol,
            "price": price,
            "amount": amount,
            "status": "Open"
        }
        return {
            "order_id": order_id,
            "order_type": order_type,
            "status": "Open",
            "price": price,
            "amount": amount
        }

    def remove_stock_from_watchlist(self, symbol: str) -> Dict[str, str]:
        """Remove a stock from the watchlist."""
        if symbol in self.watch_list:
            self.watch_list.remove(symbol)
            return {"status": f"{symbol} removed from watchlist"}
        return {"status": f"{symbol} not in watchlist"}

    def trading_login(self, username: str, password: str) -> Dict[str, str]:
        """Handle user login."""
        self.authenticated = True
        self.current_user = username
        return {"status": "Login successful"}

    def update_market_status(self, current_time_str: str) -> Dict[str, str]:
        """Update the market status based on the current time."""
        try:
            time_part, ampm = current_time_str.strip().rsplit(' ', 1)
            time_str_24h = datetime.strptime(f"{time_part} {ampm}", "%I:%M %p").strftime("%H:%")
            hour = int(time_str_24h.split(":")[0])
            minute = int(datetime.strptime(f"{time_part} {ampm}", "%I:%M %p").strftime("%M"))
            
            total_minutes = hour * 60 + minute
            if (9 * 60 + 30) <= total_minutes <= (16 * 60):
                self.market_status = "Open"
            else:
                self.market_status = "Closed"
        except (ValueError, IndexError):
            self.market_status = "Closed"
            
        return {"status": self.market_status}

    def update_stock_price(self, symbol: str, new_price: float) -> Dict[str, Any]:
        """Update the price of a stock."""
        if symbol in self.stocks:
            old_price = self.stocks[symbol].get("price", 0.0)
            self.stocks[symbol]["price"] = new_price
            return {"symbol": symbol, "old_price": old_price, "new_price": new_price}
        return {"symbol": symbol, "old_price": 0.0, "new_price": new_price}