"""Dual-import compatibility for the public tool-manager facade."""

from __future__ import annotations

if __package__:
    from . import tool_manager_bootstrap as bootstrap
    from . import tool_manager_catalog as catalog
    from . import tool_manager_cli as cli
    from . import tool_manager_core_lifecycle as core_lifecycle
    from . import tool_manager_core_lookup as core_lookup
    from . import tool_manager_loading as loading
    from . import tool_manager_output_validation as output_validation
    from . import tool_manager_python as python_tools
    from . import tool_manager_simulation as simulation
    from . import tool_manager_state as state
    from . import tool_manager_types as types
    from .llm_remote_client import LLMClient
else:
    import tool_manager_bootstrap as bootstrap
    import tool_manager_catalog as catalog
    import tool_manager_cli as cli
    import tool_manager_core_lifecycle as core_lifecycle
    import tool_manager_core_lookup as core_lookup
    import tool_manager_loading as loading
    import tool_manager_output_validation as output_validation
    import tool_manager_python as python_tools
    import tool_manager_simulation as simulation
    import tool_manager_state as state
    import tool_manager_types as types
    from llm_remote_client import LLMClient

__all__ = [
    "LLMClient",
    "bootstrap",
    "catalog",
    "cli",
    "core_lifecycle",
    "core_lookup",
    "loading",
    "output_validation",
    "python_tools",
    "simulation",
    "state",
    "types",
]
