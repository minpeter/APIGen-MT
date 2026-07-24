"""Shared runtime narrowing helpers for tool configuration data."""

from collections.abc import Mapping
from typing import TypeIs

type Config = Mapping[str, object]
type Record = dict[str, object]


def is_object_dict(value: object) -> TypeIs[dict[object, object]]:
    """Narrow a runtime value to a dictionary of objects."""
    return isinstance(value, dict)


def is_object_list(value: object) -> TypeIs[list[object]]:
    """Narrow a runtime value to a list of objects."""
    return isinstance(value, list)


def is_record(value: object) -> TypeIs[Record]:
    """Return whether a value is a string-keyed mutable record."""
    return is_object_dict(value) and all(isinstance(key, str) for key in value)


def is_record_list(value: object) -> TypeIs[list[Record]]:
    """Return whether a value is a list of mutable records."""
    return is_object_list(value) and all(is_record(item) for item in value)


def is_record_map(value: object) -> TypeIs[dict[str, Record]]:
    """Return whether a value maps strings to mutable records."""
    return is_record(value) and all(is_record(item) for item in value.values())


def is_string_list(value: object) -> TypeIs[list[str]]:
    """Return whether a value is a list of strings."""
    return is_object_list(value) and all(isinstance(item, str) for item in value)


def is_string_map(value: object) -> TypeIs[dict[str, str]]:
    """Return whether a value maps strings to strings."""
    return is_record(value) and all(
        isinstance(item, str) for item in value.values()
    )


def get_bool(config: Config, key: str, default: bool = False) -> bool:
    """Read a boolean configuration value."""
    value = config.get(key, default)
    return value if isinstance(value, bool) else default


def get_float(config: Config, key: str, default: float = 0.0) -> float:
    """Read a numeric configuration value as a float."""
    value = config.get(key, default)
    return float(value) if isinstance(value, int | float) else default


def get_int(config: Config, key: str, default: int = 0) -> int:
    """Read an integer configuration value."""
    value = config.get(key, default)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def get_record(config: Config, key: str) -> Record:
    """Read a mutable record configuration value."""
    value = config.get(key)
    return value if is_record(value) else {}


def get_record_list(config: Config, key: str) -> list[Record]:
    """Read a list of mutable records from configuration."""
    value = config.get(key)
    return value if is_record_list(value) else []


def get_record_map(config: Config, key: str) -> dict[str, Record]:
    """Read a string-to-record mapping from configuration."""
    value = config.get(key)
    return value if is_record_map(value) else {}


def get_str(config: Config, key: str, default: str = "") -> str:
    """Read a string configuration value."""
    value = config.get(key, default)
    return value if isinstance(value, str) else default


def get_string_list(config: Config, key: str) -> list[str]:
    """Read a list of strings from configuration."""
    value = config.get(key)
    return value if is_string_list(value) else []


def get_string_map(config: Config, key: str) -> dict[str, str]:
    """Read a string-to-string mapping from configuration."""
    value = config.get(key)
    return value if is_string_map(value) else {}
