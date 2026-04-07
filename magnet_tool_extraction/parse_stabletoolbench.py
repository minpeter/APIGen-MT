"""
Parser for StableToolBench / ToolBench tool environment.

Dataset: stabletoolbench/ToolEnv2404 on HuggingFace
GitHub:  https://github.com/THUNLP-MT/StableToolBench

Folder layout (after downloading and extracting):
    tools/
        {Category}/
            {tool_name}.json          ← one JSON per tool
            {tool_name}/              ← optional: executable code
                api.py

Each JSON file has the shape:
{
    "tool_name": "...",
    "tool_description": "...",
    "title": "...",
    "standardized_name": "...",
    "api_list": [
        {
            "name": "api_function_name",
            "description": "...",
            "method": "GET",
            "required_parameters": [
                {"name": "...", "type": "string", "description": "...", "default": ""}
            ],
            "optional_parameters": [
                {"name": "...", "type": "string", "description": "...", "default": ""}
            ]
        },
        ...
    ]
}

Magnet (Section 3.4) filters these by:
  - must have at least one parameter (implemented via require_parameters)
  - must be executable (verified by simulated calls; we check avgSuccessRate > 0)
"""

import re

import json
import logging
from pathlib import Path

from tool_definition import ToolDefinition, ToolParameters

logger = logging.getLogger(__name__)


def _to_pascal_case(s: str) -> str:
    """Convert a string to PascalCase (e.g. 'diet api' -> 'DietApi')."""
    words = re.sub(r'[^a-zA-Z0-9]', ' ', s).split()
    return "".join(w.capitalize() for w in words)


def _to_snake_case(s: str) -> str:
    """Convert a string to snake_case (e.g. 'Get Diet Food' -> 'get_diet_food')."""
    words = re.sub(r'[^a-zA-Z0-9]', ' ', s).split()
    return "_".join(w.lower() for w in words)


def _parse_param_list(
    param_list: list[dict],
) -> tuple[dict[str, dict], list[str]]:
    """
    Convert a ToolBench required/optional parameter list into a
    JSON-schema-like ``properties`` dict and a list of param names.

    Args:
        param_list: List of parameter dicts with ``name``, ``type``,
                    ``description``, and optionally ``default``.

    Returns:
        (properties, names): ``properties`` maps name → schema object;
        ``names`` is the ordered list of parameter names.
    """
    properties: dict[str, dict] = {}
    names: list[str] = []
    for p in param_list:
        pname = p.get("name", "").strip()
        if not pname:
            continue
        prop: dict = {
            "type": p.get("type", "string"),
            "description": p.get("description", ""),
        }
        if "default" in p and p["default"] not in ("", None):
            prop["default"] = p["default"]
        properties[pname] = prop
        names.append(pname)
    return properties, names


def parse_stable_toolbench(
    tools_root: str | Path,
    *,
    require_parameters: bool = True,
) -> list[ToolDefinition]:
    """
    Parse all tools from a StableToolBench / ToolEnv2404 directory tree.

    Args:
        tools_root: Path to the root ``tools/`` directory that contains
                    one sub-folder per category.
        require_parameters: If True (default, matching Magnet's filter),
                            skip APIs with no required **and** no optional
                            parameters.

    Returns:
        A flat list of :class:`ToolDefinition` objects, one per API endpoint.
    """
    tools_root = Path(tools_root)
    if not tools_root.is_dir():
        raise FileNotFoundError(f"Tools directory not found: {tools_root}")

    definitions: list[ToolDefinition] = []
    json_files = list(tools_root.rglob("*.json"))
    logger.info("Found %d JSON files under %s", len(json_files), tools_root)

    for json_path in json_files:
        # Infer the category from the immediate parent directory name
        # (tools/{Category}/{tool_name}.json)
        category = json_path.parent.name

        try:
            with json_path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s — could not parse: %s", json_path, exc)
            continue

        # Magnet (Section 3.4) filter: verified by simulated calls.
        # In ToolEnv2404, we use 'avgSuccessRate > 0' as a proxy for this.
        score = data.get("score")
        success_rate = (score or {}).get("avgSuccessRate", 0)
        if success_rate == 0:
            logger.debug("Skipping %s — avgSuccessRate is 0", json_path)
            continue

        orig_tool_name: str = data.get("tool_name") or data.get("title", json_path.stem)
        tool_name = _to_pascal_case(orig_tool_name)
        tool_description: str = data.get("tool_description", "")
        api_list: list[dict] = data.get("api_list", [])

        if not api_list:
            logger.debug("Skipping %s — no api_list", json_path)
            continue

        for api in api_list:
            orig_api_name: str = api.get("name", "").strip()
            if not orig_api_name:
                continue
            api_name = _to_snake_case(orig_api_name)

            api_description: str = api.get("description", "")

            req_params_raw: list[dict] = api.get("required_parameters", [])
            opt_params_raw: list[dict] = api.get("optional_parameters", [])

            if require_parameters and not req_params_raw and not opt_params_raw:
                logger.debug(
                    "Skipping %s::%s — no parameters", tool_name, api_name
                )
                continue

            req_props, req_names = _parse_param_list(req_params_raw)
            opt_props, opt_names = _parse_param_list(opt_params_raw)

            parameters = ToolParameters(
                properties={**req_props, **opt_props},
                required=req_names,
                optional=opt_names,
            )

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
        "Parsed %d API definitions from StableToolBench (%s)",
        len(definitions),
        tools_root,
    )
    return definitions
