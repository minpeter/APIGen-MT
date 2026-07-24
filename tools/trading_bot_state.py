"""Initialized state shared by TradingBot operation mixins."""

from .type_utils import (
    Config,
    Record,
    get_bool,
    get_int,
    get_record,
    get_record_list,
    get_record_map,
    get_str,
    get_string_list,
)


class TradingBotState:
    """Initialize the mutable state consumed by trading operations."""

    def __init__(self, initial_config: Config) -> None:
        self.account_info: Record = get_record(
            initial_config, "account_info"
        )
        if not self.account_info:
            self.account_info = {
                "account_id": 0,
                "balance": 0.0,
                "binding_card": 0,
            }
        self.authenticated: bool = get_bool(
            initial_config, "authenticated"
        )
        self.current_user: str = get_str(initial_config, "current_user")
        self.market_status: str = get_str(
            initial_config, "market_status", "Closed"
        )
        self.order_counter: int = get_int(initial_config, "order_counter")
        self.stocks: dict[str, Record] = get_record_map(
            initial_config, "stocks"
        )
        self.watch_list: list[str] = get_string_list(
            initial_config, "watch_list"
        )
        self.username: str = get_str(initial_config, "username")
        self.password: str = get_str(initial_config, "password")

        raw_history = get_record_list(initial_config, "transaction_history")
        self.transaction_history: list[Record] = [
            {key: value for key, value in item.items() if key != "order_id"}
            for item in raw_history
        ]
        self.orders: dict[int, Record] = {}
        for item in raw_history:
            order_id = item.get("order_id")
            if not isinstance(order_id, int):
                continue
            self.orders[order_id] = {
                "id": order_id,
                "order_type": get_str(item, "order_type", "Buy"),
                "symbol": get_str(item, "symbol"),
                "price": self._number(item, "price"),
                "amount": get_int(
                    item,
                    "num_shares",
                    get_int(item, "amount"),
                ),
                "status": get_str(item, "status", "Pending"),
            }

        self.sector_mapping: dict[str, list[str]] = {
            "Technology": ["AAPL", "GOOG", "MSFT", "NVDA"],
            "Automotive": ["TSLA"],
            "Finance": ["ALPH", "OMEG"],
            "Energy": ["QUAS", "NEPT"],
            "Healthcare": ["SYNX", "ZETA"],
        }
        self.name_to_symbol: dict[str, str] = {
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
            "zeta": "ZETA",
        }

    @staticmethod
    def _number(record: Config, key: str, default: float = 0.0) -> float:
        value = record.get(key, default)
        return float(value) if isinstance(value, int | float) else default
