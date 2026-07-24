"""Typed startup data and dynamic third-party boundaries for ToolManager."""

from __future__ import annotations

import importlib
import random

if __package__:
    from . import tool_manager_catalog as catalog
    from .tool_manager_types import (
        SchemaBuilder,
        get_attribute,
        is_config_factory,
        is_schema_builder,
        require_config_list,
    )
else:
    import tool_manager_catalog as catalog
    from tool_manager_types import (
        SchemaBuilder,
        get_attribute,
        is_config_factory,
        is_schema_builder,
        require_config_list,
    )

_SIBLING_PREFIX = f"{__package__}." if __package__ else ""
_config_pool = importlib.import_module(f"{_SIBLING_PREFIX}config_pool")
_schema_module = importlib.import_module(f"{_SIBLING_PREFIX}function_schema")

MESSAGE_CONFIGS = require_config_list(
    get_attribute(_config_pool, "MESSAGE_CONFIGS"), source="MESSAGE_CONFIGS"
)
_random_config_candidate = get_attribute(_config_pool, "generate_random_config")
if not is_config_factory(_random_config_candidate):
    raise TypeError("config_pool.generate_random_config must be callable")
generate_random_config = _random_config_candidate

_schema_builder_candidate = get_attribute(_schema_module, "get_function_schema")
if not is_schema_builder(_schema_builder_candidate):
    raise TypeError("function_schema.get_function_schema must be callable")
get_function_schema: SchemaBuilder = _schema_builder_candidate

CLASS_KEY_TO_CLASS_NAME = catalog.CLASS_KEY_TO_CLASS_NAME
CLASS_KEY_TO_INITIAL_CONFIG_KEY = catalog.CLASS_KEY_TO_INITIAL_CONFIG_KEY
TOOL_CLASS_KEYS = catalog.TOOL_CLASS_KEYS
TOOL_NAME_TO_CLASS_KEY = catalog.TOOL_NAME_TO_CLASS_KEY
FULL_INITIAL_CONFIGS = catalog.build_full_initial_configs(
    random.choice(MESSAGE_CONFIGS)
)
