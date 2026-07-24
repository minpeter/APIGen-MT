"""Virtual in-memory filesystem for tool simulation."""

from .gorilla_file_system_read import GorillaFileSystemReadOperations
from .gorilla_file_system_write import GorillaFileSystemWriteOperations
from .type_utils import Config


class GorillaFileSystem(
    GorillaFileSystemReadOperations,
    GorillaFileSystemWriteOperations,
):
    """Maintain an isolated virtual filesystem across tool calls."""

    def __init__(self, initial_config: Config | None = None) -> None:
        """Initialize with optional serialized filesystem state."""
        super().__init__(initial_config)
