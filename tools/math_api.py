"""MathAPI compatibility facade."""

from .math_api_conversions import MathAPIConversionsMixin
from .math_api_operations import MathAPIOperationsMixin
from .type_utils import Config


class MathAPI(MathAPIOperationsMixin, MathAPIConversionsMixin):
    """A class providing various mathematical operations."""

    def __init__(self, initial_config: Config) -> None:
        """
        Initialize the MathAPI with an initial configuration.
        Note: All methods are pure - they take parameters directly.
        The initial_config is kept for API compatibility but no fields are used.
        """
        _ = initial_config
        super().__init__()
