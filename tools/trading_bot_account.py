"""Authentication, account, and order operations for TradingBot."""

from datetime import date, datetime

from .trading_bot_state import TradingBotState
from .trading_bot_types import (
    BalanceResult,
    CancelOrderResult,
    OrderDetailsResult,
    PlaceOrderResult,
    TradingLoginResult,
    TransactionHistoryItem,
    TransactionHistoryResult,
)
from .type_utils import Record, get_int, get_str


class TradingBotAccountMixin(TradingBotState):
    """Provide account funding, authentication, and order operations."""

    def cancel_order(self, order_id: int) -> CancelOrderResult:
        """Cancel an order."""
        if not self.authenticated:
            return {"order_id": order_id, "status": "User not authenticated"}
        order = self.orders.get(order_id)
        if order is None:
            return {"order_id": order_id, "status": "Order not found"}
        order["status"] = "Cancelled"
        return {"order_id": order_id, "status": "Cancelled"}

    def fund_account(self, amount: float) -> BalanceResult:
        """Fund the account with the specified amount."""
        balance = self._number(self.account_info, "balance")
        if not self.authenticated:
            return {"status": "User not authenticated", "new_balance": balance}
        if amount <= 0:
            return {"status": "Failed: invalid amount", "new_balance": balance}
        balance += amount
        self.account_info["balance"] = balance
        return {"status": "Account funded successfully", "new_balance": balance}

    def get_order_details(self, order_id: int) -> OrderDetailsResult:
        """Get the details of an order."""
        order = self.orders.get(order_id)
        if order is None:
            return {
                "id": order_id,
                "order_type": "",
                "symbol": "",
                "price": 0.0,
                "amount": 0,
                "status": "Order not found",
            }
        return {
            "id": get_int(order, "id", order_id),
            "order_type": get_str(order, "order_type"),
            "symbol": get_str(order, "symbol"),
            "price": self._number(order, "price"),
            "amount": get_int(order, "amount"),
            "status": get_str(order, "status"),
        }

    def get_transaction_history(
        self, start_date: str = "None", end_date: str = "None"
    ) -> TransactionHistoryResult:
        """Get the transaction history within a specified date range."""
        if not self.authenticated:
            return {
                "transaction_history": [],
                "status": "User not authenticated",
            }
        parsed_start = self._optional_date(start_date)
        parsed_end = self._optional_date(end_date)
        result: list[TransactionHistoryItem] = []
        for transaction in self.transaction_history:
            timestamp_string = get_str(transaction, "timestamp")
            try:
                transaction_date = datetime.fromisoformat(
                    timestamp_string
                ).date()
            except ValueError:
                continue
            if parsed_start and transaction_date < parsed_start:
                continue
            if parsed_end and transaction_date > parsed_end:
                continue
            total_cost = self._number(
                transaction,
                "total_cost",
                self._number(transaction, "price")
                * get_int(transaction, "num_shares"),
            )
            result.append(
                {
                    "type": get_str(transaction, "type"),
                    "symbol": get_str(transaction, "symbol"),
                    "total_cost": total_cost,
                    "timestamp": timestamp_string,
                }
            )
        return {"transaction_history": result}

    def make_transaction(
        self, account_id: int, xact_type: str, amount: float
    ) -> BalanceResult:
        """Make a deposit or withdrawal against the trading account."""
        balance = self._number(self.account_info, "balance")
        if not self.authenticated:
            return {"status": "User not authenticated", "new_balance": balance}
        if account_id != get_int(self.account_info, "account_id"):
            return {"status": "Failed: account not found", "new_balance": balance}
        if xact_type == "deposit":
            if amount <= 0:
                return {"status": "Failed: invalid amount", "new_balance": balance}
            balance += amount
            status = "Deposit successful"
        elif xact_type == "withdrawal":
            if amount <= 0 or amount > balance:
                return {
                    "status": "Failed: invalid amount or insufficient funds",
                    "new_balance": balance,
                }
            balance -= amount
            status = "Withdrawal successful"
        else:
            return {
                "status": "Failed: invalid transaction type",
                "new_balance": balance,
            }
        self.account_info["balance"] = balance
        return {"status": status, "new_balance": balance}

    def place_order(
        self, order_type: str, symbol: str, price: float, amount: int
    ) -> PlaceOrderResult:
        """Place an order."""
        if not self.authenticated:
            return {
                "order_id": 0,
                "order_type": order_type,
                "status": "User not authenticated",
                "price": price,
                "amount": amount,
            }
        self.order_counter: int = self.order_counter + 1
        order_id = self.order_counter
        self.orders[order_id] = {
            "id": order_id,
            "order_type": order_type,
            "symbol": symbol,
            "price": price,
            "amount": amount,
            "status": "Open",
        }
        transaction: Record = {
            "order_id": order_id,
            "type": order_type,
            "symbol": symbol,
            "price": price,
            "num_shares": amount,
            "total_cost": price * amount,
            "status": "Filled",
            "timestamp": datetime.now().astimezone().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
        self.transaction_history.append(transaction)
        return {
            "order_id": order_id,
            "order_type": order_type,
            "status": "Open",
            "price": price,
            "amount": amount,
        }

    def trading_login(self, username: str, password: str) -> TradingLoginResult:
        """Authenticate a trading user."""
        if self.username or self.password:
            if not username or not password:
                return {
                    "status": "Login failed: username and password required"
                }
            if username != self.username or password != self.password:
                return {"status": "Login failed: invalid credentials"}
        self.authenticated: bool = True
        self.current_user: str = username
        return {"status": "Login successful"}

    @staticmethod
    def _optional_date(value: str) -> date | None:
        if not value or value == "None":
            return None
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
