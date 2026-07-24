"""
Parser for BFCL-v3 (Berkeley Function Calling Leaderboard v3) multi-turn
function documentation.

Dataset: gorilla-llm/Berkeley-Function-Calling-Leaderboard on HuggingFace
GitHub:  https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard

Relevant files for function definitions used by Magnet (Section 3.4):
    berkeley-function-call-leaderboard/
        data/
            multi_turn_func_doc/
                {class_name}.json     ← OpenAI-style function schemas per class
            BFCL_v3_multi_turn_base.json
            BFCL_v3_multi_turn_miss_func.json
            BFCL_v3_multi_turn_miss_param.json
            BFCL_v3_multi_turn_long_context.json

Each ``multi_turn_func_doc/{class_name}.json`` contains a list of function
objects following the OpenAI function-calling schema:
[
    {
        "name": "function_name",
        "description": "...",
        "parameters": {
            "type": "object",         ← BFCL uses "object"; Magnet normalises to "dict"
            "properties": {
                "param_name": {
                    "type": "string",
                    "description": "..."
                }
            },
            "required": ["param_name"]
        }
    },
    ...
]

The multi-turn test entries look like:
{
    "id": "...",
    "involved_classes": ["gorilla_file_system", "trading_bot"],
    "turns": [ ... ],
    ...
}

Magnet uses these functions as a separate function pool whose names are
rewritten by an LLM to avoid benchmark contamination (Section 3.4).

This parser:
    1. Reads all func-doc JSON files from the ``multi_turn_func_doc/`` directory.
    2. Optionally reads the test set JSON lines to discover which classes are
       actually referenced (for filtering).
    3. Returns ToolDefinition objects in the Magnet canonical format.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypeIs

from magnet_tool_extraction.tool_definition import ToolDefinition, ToolParameters

logger = logging.getLogger(__name__)

_JSON_LOADS: Callable[[str], object] = json.loads


def _is_object_dict(value: object) -> TypeIs[dict[object, object]]:
    """Narrow an unknown mapping without introducing an untyped dictionary."""
    return isinstance(value, dict)


def _is_json_object(value: object) -> TypeIs[dict[str, object]]:
    """Return whether a decoded JSON value is an object with string keys."""
    return _is_object_dict(value) and all(isinstance(key, str) for key in value)


def _is_object_list(value: object) -> TypeIs[list[object]]:
    """Narrow an unknown JSON array without checker casts."""
    return isinstance(value, list)


def _load_json_records(content: str, source: Path) -> list[object]:
    """Decode a JSON array/object or a JSON-lines document."""
    try:
        decoded = _JSON_LOADS(content)
    except json.JSONDecodeError:
        records: list[object] = []
        for line_number, raw_line in enumerate(content.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(_JSON_LOADS(line))
            except json.JSONDecodeError as exc:
                logger.debug(
                    "Skipping invalid JSON line %d in %s: %s",
                    line_number,
                    source,
                    exc,
                )
        return records

    return decoded if _is_object_list(decoded) else [decoded]


def _get_json_object(record: dict[str, object], key: str) -> dict[str, object]:
    """Read an optional nested JSON object, rejecting malformed schemas."""
    value = record.get(key, {})
    if not _is_json_object(value):
        raise TypeError(f"{key!r} must be a JSON object")
    return value


def _get_string(record: dict[str, object], key: str) -> str:
    """Read an optional string field, rejecting malformed schemas."""
    value = record.get(key, "")
    if not isinstance(value, str):
        raise TypeError(f"{key!r} must be a string")
    return value


def _get_string_list(record: dict[str, object], key: str) -> list[str]:
    """Read an optional array whose entries must all be strings."""
    value = record.get(key, [])
    if not _is_object_list(value) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{key!r} must be an array of strings")
    return [item for item in value if isinstance(item, str)]


# BFCL multi-turn test file names (relative to the ``data/`` directory)
BFCL_MULTI_TURN_FILES = [
    "BFCL_v3_multi_turn_base.json",
    "BFCL_v3_multi_turn_miss_func.json",
    "BFCL_v3_multi_turn_miss_param.json",
    "BFCL_v3_multi_turn_long_context.json",
]

# Known BFCL domain ↔ category mapping (best-effort; falls back to class name)
_BFCL_CATEGORY_MAP: dict[str, str] = {
    "gorilla_file_system": "Storage",
    "trading_bot": "Finance",
    "ticket_api": "Events",
    "weather_api": "Weather",
    "math_api": "Science",
    "message_api": "Communication",
    "calendar_api": "Business_Software",
}


def _resolve_category(class_name: str) -> str:
    """Map a BFCL class name to a Magnet category string."""
    return _BFCL_CATEGORY_MAP.get(class_name, class_name.replace("_", " ").title())


def _parse_openai_parameters(
    params_schema: dict[str, object],
) -> ToolParameters:
    """
    Convert an OpenAI-style parameters schema to a :class:`ToolParameters`.

    BFCL uses ``"type": "object"``; we normalise it to ``"dict"`` to match
    the Magnet template.

    Args:
        params_schema: The ``parameters`` sub-dict from a function doc entry.

    Returns:
        A :class:`ToolParameters` with ``properties``, ``required``, and
        ``optional`` filled in.
    """
    properties = _get_json_object(params_schema, "properties")
    required_names = _get_string_list(params_schema, "required")
    optional_names = [name for name in properties if name not in required_names]

    return ToolParameters(
        type="dict",
        properties=properties,
        required=required_names,
        optional=optional_names,
    )


def parse_bfcl_func_doc(
    func_doc_dir: str | Path,
    *,
    class_names: list[str] | None = None,
    require_parameters: bool = True,
) -> list[ToolDefinition]:
    """
    Parse function documentation from the BFCL ``multi_turn_func_doc/``
    directory.

    Args:
        func_doc_dir: Path to ``multi_turn_func_doc/``.
        class_names: Optional allow-list of class names (file stems) to parse.
                     If None, all JSON files in the directory are parsed.
        require_parameters: If True (matching Magnet's filter), skip functions
                            with no parameters at all.

    Returns:
        A flat list of :class:`ToolDefinition` objects (one per function).
    """
    func_doc_dir = Path(func_doc_dir)
    if not func_doc_dir.is_dir():
        raise FileNotFoundError(f"BFCL func-doc directory not found: {func_doc_dir}")

    json_files = sorted(func_doc_dir.glob("*.json"))
    if class_names is not None:
        allowed = set(class_names)
        json_files = [f for f in json_files if f.stem in allowed]

    definitions: list[ToolDefinition] = []

    for json_path in json_files:
        class_name = json_path.stem
        category = _resolve_category(class_name)
        # Use the class name as a stand-in for tool_name (no explicit tool
        # description metadata exists in BFCL func-doc files).
        tool_name = class_name
        tool_description = (
            f"Functions provided by the {class_name.replace('_', ' ')} toolkit."
        )

        try:
            content = json_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Skipping %s — could not read: %s", json_path, exc)
            continue

        for raw_func in _load_json_records(content, json_path):
            if not _is_json_object(raw_func):
                logger.debug("Skipping non-object function in %s", json_path)
                continue

            try:
                api_name = _get_string(raw_func, "name").strip()
                api_description = _get_string(raw_func, "description")
                params_schema = _get_json_object(raw_func, "parameters")
                parameters = _parse_openai_parameters(params_schema)
            except TypeError as exc:
                logger.warning("Skipping malformed function in %s: %s", json_path, exc)
                continue

            if not api_name:
                continue
            if require_parameters and not parameters.properties:
                logger.debug("Skipping %s::%s — no parameters", class_name, api_name)
                continue

            definitions.append(
                ToolDefinition(
                    category=category,
                    tool_name=tool_name,
                    tool_description=tool_description,
                    api_name=api_name,
                    api_description=api_description,
                    parameters=parameters,
                )
            )

    logger.info(
        "Parsed %d function definitions from BFCL func-doc (%s)",
        len(definitions),
        func_doc_dir,
    )
    return definitions


def discover_bfcl_classes(data_dir: str | Path) -> set[str]:
    """
    Scan the BFCL multi-turn test JSON files and collect all class names
    referenced in ``involved_classes`` fields.

    Args:
        data_dir: Path to the BFCL ``data/`` directory.

    Returns:
        Set of unique class name strings present in the test files.
    """
    data_dir = Path(data_dir)
    class_names: set[str] = set()

    for fname in BFCL_MULTI_TURN_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            logger.debug("BFCL test file not found (skipping): %s", fpath)
            continue

        try:
            with fpath.open(encoding="utf-8") as fh:
                # The BFCL test set is a list of JSON objects (one per line
                # or a single JSON array — handle both).
                content = fh.read().strip()
        except OSError as exc:
            logger.warning("Could not read %s: %s", fpath, exc)
            continue

        for raw_entry in _load_json_records(content, fpath):
            if not _is_json_object(raw_entry):
                logger.debug("Skipping non-object test entry in %s", fpath)
                continue
            try:
                involved_classes = _get_string_list(raw_entry, "involved_classes")
            except TypeError as exc:
                logger.debug("Skipping malformed entry in %s: %s", fpath, exc)
                continue
            class_names.update(involved_classes)

    logger.info(
        "Discovered %d BFCL class names from test files in %s",
        len(class_names),
        data_dir,
    )
    return class_names


def parse_bfcl_jsonl(
    data_dir: str | Path,
    *,
    require_parameters: bool = True,
) -> list[ToolDefinition]:
    """
    Parse function definitions from all BFCL_v3 JSONL files in the data directory.

    Args:
        data_dir: Path to the BFCL ``data/`` directory containing BFCL_v3_*.json.
        require_parameters: If True, skip functions with no parameters.

    Returns:
        A list of unique ToolDefinition objects.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"BFCL data directory not found: {data_dir}")

    # Find all BFCL_v3 files (except the multi_turn ones which are handled
    # separately by parse_bfcl_func_doc if needed, but here we cast a wide net).
    json_files = sorted(data_dir.glob("BFCL_v3_*.json"))

    # unique_functions maps function_name -> ToolDefinition
    unique_functions: dict[str, ToolDefinition] = {}

    for json_path in json_files:
        logger.info("Parsing BFCL JSONL file: %s", json_path)
        try:
            with json_path.open(encoding="utf-8") as fh:
                content = fh.read().strip()
        except OSError as exc:
            logger.warning("Could not read %s: %s", json_path, exc)
            continue

        # Each entry has a "function" list.
        for raw_entry in _load_json_records(content, json_path):
            if not _is_json_object(raw_entry):
                logger.debug("Skipping non-object entry in %s", json_path)
                continue
            raw_functions = raw_entry.get("function", [])
            if not _is_object_list(raw_functions):
                logger.debug("Skipping malformed function list in %s", json_path)
                continue

            for raw_func in raw_functions:
                if not _is_json_object(raw_func):
                    logger.debug("Skipping non-object function in %s", json_path)
                    continue
                try:
                    api_name = _get_string(raw_func, "name").strip()
                    api_description = _get_string(raw_func, "description")
                    params_schema = _get_json_object(raw_func, "parameters")
                    parameters = _parse_openai_parameters(params_schema)
                except TypeError as exc:
                    logger.warning(
                        "Skipping malformed function in %s: %s",
                        json_path,
                        exc,
                    )
                    continue

                if not api_name or api_name in unique_functions:
                    continue
                if require_parameters and not parameters.properties:
                    continue

                # For these generic tools, we don't have a class_name.
                # Use a generic category based on the file name or just "General".
                file_stem = json_path.stem
                if "java" in file_stem.lower():
                    category = "Java"
                elif "javascript" in file_stem.lower():
                    category = "JavaScript"
                elif "rest" in file_stem.lower():
                    category = "REST"
                elif "sql" in file_stem.lower():
                    category = "SQL"
                else:
                    category = "General"

                unique_functions[api_name] = ToolDefinition(
                    category=category,
                    tool_name="BFCL Generic",
                    tool_description=(
                        "Functions from the Berkeley Function Calling Leaderboard."
                    ),
                    api_name=api_name,
                    api_description=api_description,
                    parameters=parameters,
                )

    definitions = list(unique_functions.values())
    logger.info(
        "Extracted %d unique function definitions from BFCL JSONL files in %s",
        len(definitions),
        data_dir,
    )
    return definitions
