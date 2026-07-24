"""TradingBot compatibility facade."""

from .trading_bot_account import TradingBotAccountMixin
from .trading_bot_market import TradingBotMarketMixin
from .type_utils import Config


class TradingBot(TradingBotAccountMixin, TradingBotMarketMixin):
    """Trade stocks, manage an account, and inspect market information."""

    def __init__(self, initial_config: Config) -> None:
        """Initialize the trading bot with the provided configuration."""
        super().__init__(initial_config)
