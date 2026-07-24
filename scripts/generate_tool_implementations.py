#!/usr/bin/env python3
"""LLM-driven generator for BFCL tool implementations.

Reads tool definitions from bfcl_v3_tools_with_outputs.jsonl, response schemas
from BFCL func_doc files, and invocation examples from
bfcl_v3_invocation_examples.jsonl, then prompts an LLM to generate:

1. Stateful class modules (tools/{class}.py) with methods per tool
2. Pydantic input schemas (tools/schemas.py)
3. Unit tests (tests/tools/test_{class}.py)

Usage:
    python scripts/generate_tool_implementations.py
    python scripts/generate_tool_implementations.py --classes math_api,gorilla_file_system
    python scripts/generate_tool_implementations.py --skip-existing --verbose
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import NotRequired, TypedDict, TypeIs

import requests
from dotenv import load_dotenv

_ = load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from runtime_config import DEFAULT_API_BASE, DEFAULT_MODEL

type JsonObject = dict[str, object]

_JSON_LOADS: Callable[[str], object] = json.loads


class ChatOptions(TypedDict):
    """Options accepted by the lightweight chat client."""

    temperature: float
    max_tokens: int
    timeout: NotRequired[float]


class AuthInfo(TypedDict):
    """Authentication metadata used while generating sequential tests."""

    login_method: str
    login_params: JsonObject
    auth_state_field: str
    auth_failure_value: object
    gated_methods: list[str]
    non_gated_methods: list[str]
    token_field: NotRequired[str]


class CliArgs(argparse.Namespace):
    """Typed command-line arguments for the generator."""

    classes: str | None = None
    output_dir: str = ""
    test_dir: str = ""
    model: str = DEFAULT_MODEL
    api_base: str | None = None
    api_key: str | None = None
    skip_existing: bool = False
    verbose: bool = False
    max_retries: int = 0
    skip_tests: bool = False
    only_tests: bool = False
    only_schemas: bool = False
    sequential_tests: bool = False


def _is_object_dict(value: object) -> TypeIs[dict[object, object]]:
    """Narrow an unknown mapping without introducing an untyped dictionary."""
    return isinstance(value, dict)


def _is_json_object(value: object) -> TypeIs[JsonObject]:
    """Return whether a decoded JSON value is an object with string keys."""
    return _is_object_dict(value) and all(isinstance(key, str) for key in value)


def _is_object_list(value: object) -> TypeIs[list[object]]:
    """Narrow an unknown JSON array without checker casts."""
    return isinstance(value, list)


def _load_json_object(text: str, source: str) -> JsonObject:
    """Decode one JSON object and report malformed input at its boundary."""
    decoded = _JSON_LOADS(text)
    if not _is_json_object(decoded):
        raise TypeError(f"Expected a JSON object in {source}")
    return decoded


def _require_json_object(value: object, field: str) -> JsonObject:
    """Validate a nested JSON object."""
    if not _is_json_object(value):
        raise TypeError(f"{field!r} must be a JSON object")
    return value


def _get_json_object(record: JsonObject, key: str) -> JsonObject:
    """Read an optional nested JSON object."""
    return _require_json_object(record.get(key, {}), key)


def _get_object_list(record: JsonObject, key: str) -> list[object]:
    """Read an optional JSON array."""
    value = record.get(key, [])
    if not _is_object_list(value):
        raise TypeError(f"{key!r} must be a JSON array")
    return value


def _get_string(record: JsonObject, key: str, default: str = "") -> str:
    """Read an optional string field."""
    value = record.get(key, default)
    if not isinstance(value, str):
        raise TypeError(f"{key!r} must be a string")
    return value


def _get_string_list(record: JsonObject, key: str) -> list[str]:
    """Read an optional array containing only strings."""
    values = _get_object_list(record, key)
    if not all(isinstance(value, str) for value in values):
        raise TypeError(f"{key!r} must be an array of strings")
    return [value for value in values if isinstance(value, str)]


def _tool_sort_key(tool: JsonObject) -> str:
    """Return the stable API-name sort key used in generated prompts."""
    return _get_string(tool, "api_name")


TOOLS_EXTRACTION_DIR = PROJECT_ROOT / "magnet_tool_extraction"
BFCL_FUNC_DOC_DIR = (
    PROJECT_ROOT.parent / "magnet_mt" / "data" / "BFCL_v3" / "multi_turn_func_doc"
)

CLASS_KEY_TO_FUNC_DOC_FILE = {
    "gorilla_file_system": "gorilla_file_system.json",
    "math_api": "math_api.json",
    "message_api": "message_api.json",
    "posting_api": "posting_api.json",
    "ticket_api": "ticket_api.json",
    "trading_bot": "trading_bot.json",
    "travel_booking": "travel_booking.json",
    "vehicle_control": "vehicle_control.json",
}

CLASS_KEY_TO_CLASS_NAME = {
    "gorilla_file_system": "GorillaFileSystem",
    "math_api": "MathAPI",
    "message_api": "MessageAPI",
    "posting_api": "PostingAPI",
    "ticket_api": "TicketAPI",
    "trading_bot": "TradingBot",
    "travel_booking": "TravelBooking",
    "vehicle_control": "VehicleControl",
}

CLASS_KEY_TO_INITIAL_CONFIG_KEY = {
    "gorilla_file_system": "GorillaFileSystem",
    "math_api": "MathAPI",
    "message_api": "MessageAPI",
    "posting_api": "TwitterAPI",
    "ticket_api": "TicketAPI",
    "trading_bot": "TradingBot",
    "travel_booking": "TravelAPI",
    "vehicle_control": "VehicleControlAPI",
}


# ─── Data Loading ───────────────────────────────────────────────────────────


def load_tool_definitions() -> list[JsonObject]:
    """Load all 105 tool definitions from the extracted JSONL file."""
    path = TOOLS_EXTRACTION_DIR / "bfcl_v3_tools_with_outputs.jsonl"
    tools: list[JsonObject] = []
    with path.open() as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if line:
                tools.append(_load_json_object(line, f"{path}:{line_number}"))
    return tools


def load_func_doc_schemas() -> dict[str, JsonObject]:
    """Load func-doc response schemas, keyed by function name."""
    schemas: dict[str, JsonObject] = {}
    for filename in CLASS_KEY_TO_FUNC_DOC_FILE.values():
        filepath = BFCL_FUNC_DOC_DIR / filename
        if not filepath.exists():
            print(f"  WARNING: func_doc file not found: {filepath}")
            continue
        with filepath.open() as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                entry = _load_json_object(line, f"{filepath}:{line_number}")
                schemas[_get_string(entry, "name")] = entry
    return schemas


def load_invocation_examples() -> list[JsonObject]:
    """Load all invocation examples from the extracted JSONL file."""
    path = TOOLS_EXTRACTION_DIR / "bfcl_v3_invocation_examples.jsonl"
    examples: list[JsonObject] = []
    with path.open() as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if line:
                examples.append(_load_json_object(line, f"{path}:{line_number}"))
    return examples


def group_tools_by_class(
    tools: list[JsonObject],
) -> dict[str, list[JsonObject]]:
    """Group tool definitions by their tool_name field."""
    groups: defaultdict[str, list[JsonObject]] = defaultdict(list)
    for tool in tools:
        groups[_get_string(tool, "tool_name")].append(tool)
    return dict(groups)


def group_examples_by_function(
    examples: list[JsonObject],
) -> dict[str, list[JsonObject]]:
    """Group invocation examples by function_name."""
    groups: defaultdict[str, list[JsonObject]] = defaultdict(list)
    for example in examples:
        function_name = _get_string(example, "function_name")
        if function_name:
            groups[function_name].append(example)
    return dict(groups)


def get_canonical_initial_configs(
    examples: list[JsonObject],
) -> dict[str, JsonObject]:
    """Extract the most representative (largest) initial_config per class."""
    configs_by_class: defaultdict[str, list[JsonObject]] = defaultdict(list)
    for example in examples:
        initial_config = _get_json_object(example, "initial_config")
        for class_name, class_config in initial_config.items():
            if _is_json_object(class_config):
                configs_by_class[class_name].append(class_config)

    canonical: dict[str, JsonObject] = {}
    for class_name, configs in configs_by_class.items():
        canonical[class_name] = max(configs, key=lambda config: len(json.dumps(config)))
    return canonical


def select_diverse_examples(
    examples_for_fn: list[JsonObject], max_examples: int = 5
) -> list[JsonObject]:
    """Select up to max_examples diverse invocation examples.

    Prioritizes examples from different test cases with different argument patterns.
    """
    if len(examples_for_fn) <= max_examples:
        return examples_for_fn

    seen_test_cases: set[str] = set()
    selected: list[JsonObject] = []
    remaining: list[JsonObject] = []

    for example in examples_for_fn:
        test_case_id = _get_string(example, "test_case_id")
        if test_case_id not in seen_test_cases and len(selected) < max_examples:
            selected.append(example)
            seen_test_cases.add(test_case_id)
        else:
            remaining.append(example)

    while len(selected) < max_examples and remaining:
        selected.append(remaining.pop(0))

    return selected[:max_examples]


# ─── LLM Client Setup ───────────────────────────────────────────────────────


class SimpleLLMClient:
    """Lightweight OpenAI-compatible LLM client (no transformers dependency)."""

    def __init__(
        self,
        url: str = DEFAULT_API_BASE,
        api_key: str = "",
        api_model: str = DEFAULT_MODEL,
        debug_mode: bool = False,
    ) -> None:
        self.url: str = url
        self.api_model: str = api_model
        self.debug_mode: bool = debug_mode
        self.headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.total_calls: int = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        kwargs: ChatOptions,
        max_retries: int = 10,
        base_delay: float = 2.0,
    ) -> tuple[str, str]:
        request_timeout = kwargs.pop("timeout", 600)
        payload: dict[str, object] = {
            "model": self.api_model,
            "messages": messages,
            **kwargs,
        }
        rate_limit_retries: int = 0
        attempt: int = 0

        while attempt < max_retries:
            try:
                resp = requests.request(
                    "POST",
                    url=f"{self.url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=request_timeout,
                )
                if resp.status_code >= 500:
                    if attempt < max_retries - 1:
                        delay = base_delay * pow(2.0, attempt)
                        print(
                            f"  [LLM] Server error {resp.status_code} (attempt {attempt + 1}/{max_retries}), retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        attempt += 1
                        continue
                    raise RuntimeError(
                        f"API server error {resp.status_code}: {resp.text[:300]}"
                    )

                try:
                    data = _load_json_object(resp.text, "API response")
                except (json.JSONDecodeError, TypeError) as exc:
                    if attempt < max_retries - 1:
                        delay = base_delay * pow(2.0, attempt)
                        print(
                            "  [LLM] JSON decode error "
                            + f"(attempt {attempt + 1}/{max_retries}), "
                            + f"retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        attempt += 1
                        continue
                    raise RuntimeError(
                        f"API returned non-JSON response: {resp.text[:300]}"
                    ) from exc

                if "choices" not in data:
                    if resp.status_code == 429:
                        rate_limit_retries += 1
                        delay = min(
                            base_delay * pow(2.0, min(rate_limit_retries, 10)),
                            300,
                        )
                        print(
                            f"  [LLM] Rate limited (429), retrying in {delay}s... (retry #{rate_limit_retries})"
                        )
                        time.sleep(delay)
                        continue
                    raise RuntimeError(
                        f"Unexpected API response (status={resp.status_code}): {json.dumps(data)[:300]}"
                    )

                choices = _get_object_list(data, "choices")
                choice = _require_json_object(choices[0], "choices[0]")
                message = _get_json_object(choice, "message")
                raw_content = message.get("content")
                if raw_content is None:
                    content = ""
                elif isinstance(raw_content, str):
                    content = raw_content
                else:
                    raise TypeError("API response content must be a string")

                # Strip thinking/reasoning tags (various model formats)
                reasoning = ""
                clean_content = content
                for ts, te in (
                    ("<think>", "</think>"),
                    ("<thinking>", "</thinking>"),
                ):
                    pat = re.escape(ts) + r"(.*?)" + re.escape(te)
                    m = re.search(pat, clean_content, re.DOTALL)
                    if m:
                        reasoning = m.group(1).strip()
                        clean_content = re.sub(
                            pat, "", clean_content, flags=re.DOTALL
                        ).strip()
                        break

                self.total_calls += 1
                return clean_content, reasoning

            except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout):
                delay = base_delay * pow(2.0, attempt)
                print(
                    f"  [LLM] Timeout (attempt {attempt + 1}/{max_retries}), retrying in {delay}s..."
                )
                time.sleep(delay)
                attempt += 1
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
            ) as e:
                delay = base_delay * pow(2.0, attempt)
                print(
                    f"  [LLM] Connection error (attempt {attempt + 1}/{max_retries}): {e}, retrying in {delay}s..."
                )
                time.sleep(delay)
                attempt += 1

        raise RuntimeError(f"LLM call failed after {max_retries} attempts")


def create_llm_client(
    model: str = DEFAULT_MODEL,
    api_base: str | None = None,
    api_key: str | None = None,
    verbose: bool = False,
) -> SimpleLLMClient:
    """Create a lightweight LLM client (no transformers dependency)."""

    url = api_base or os.getenv("OPENAI_API_BASE", DEFAULT_API_BASE)
    key = api_key or os.getenv("OPENAI_API_KEY", "")

    client = SimpleLLMClient(
        url=url,
        api_key=key,
        api_model=model,
        debug_mode=verbose,
    )
    return client


def call_llm(
    client: SimpleLLMClient,
    system_prompt: str,
    user_prompt: str,
    verbose: bool = False,
    max_tokens: int = 16384,
) -> str:
    """Call the LLM and return the response text."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs: ChatOptions = {"temperature": 0.3, "max_tokens": max_tokens}
    print(
        f"  [LLM] Calling {client.api_model} (prompt={len(user_prompt)} chars, max_tokens={max_tokens})...",
        flush=True,
    )
    t0 = time.time()
    response, reasoning = client.chat(messages=messages, kwargs=kwargs)
    elapsed = time.time() - t0
    print(
        f"  [LLM] Response received in {elapsed:.1f}s ({len(response)} chars)",
        flush=True,
    )
    if verbose and reasoning:
        print(f"  [REASONING] {reasoning[:200]}...")
    return response


def extract_code_block(text: str) -> str:
    """Extract Python code from markdown code blocks in LLM response."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match is not None else text.strip()


# ─── Prompt Builders ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Python code generator producing production-quality, working Python code.

Rules:
- Return ONLY valid Python code in a single markdown code block (```python ... ```)
- Use type hints on all function signatures
- Methods return dicts matching the specified response schema EXACTLY
- Stateful methods must mutate self state appropriately
- Handle edge cases: missing args, invalid values, not-found scenarios
- Never raise exceptions from methods - return error info in the response dict
- Follow the exact parameter names from the tool definitions (keep camelCase as-is)
- Import only: json, math, re, copy, datetime, typing (List, Dict, Any, Optional, Tuple)
- Do NOT include any explanatory text outside the code block
- Each method must have a docstring
- The class __init__ must accept initial_config: dict and set up internal state
- For classes with multiple initial_config variants, normalize to a canonical form in __init__"""


def build_class_prompt(
    class_key: str,
    tools: list[JsonObject],
    func_doc_schemas: dict[str, JsonObject],
    canonical_config: JsonObject,
    examples_by_fn: dict[str, list[JsonObject]],
    config_key: str,
) -> str:
    """Build the LLM prompt for generating a complete class with all methods."""
    class_name = CLASS_KEY_TO_CLASS_NAME[class_key]
    parts: list[str] = []
    parts.append(f"## Task: Generate the {class_name} class")
    parts.append(f"File: tools/{class_key}.py")
    parts.append("")

    # Class state
    parts.append("### Class State (initial_config)")
    if canonical_config:
        config_json = json.dumps(canonical_config, indent=2)
        if len(config_json) > 2000:
            config_json = config_json[:2000] + "\n... (truncated)"
        parts.append("```json")
        parts.append(config_json)
        parts.append("```")
    else:
        parts.append("(This class is stateless - no initial_config)")
    parts.append("")

    # Tools to implement
    parts.append(f"### Tools to implement ({len(tools)} methods)")
    parts.append("")

    api_name = ""
    for tool in sorted(tools, key=_tool_sort_key):
        api_name = _get_string(tool, "api_name")
        parts.append(f"#### Method: `{api_name}`")

        # Tool description
        description = _get_string(tool, "api_description")
        if description:
            parts.append(f"Description: {description}")

        # Parameters
        parameters = _get_json_object(tool, "parameters")
        if parameters:
            parts.append("Parameters:")
            properties = _get_json_object(parameters, "properties")
            required = _get_string_list(parameters, "required")
            for parameter_name, raw_info in properties.items():
                parameter_info = _require_json_object(
                    raw_info, f"parameters.properties.{parameter_name}"
                )
                parameter_type = _get_string(parameter_info, "type", "any")
                parameter_description = _get_string(parameter_info, "description")
                parameter_default = parameter_info.get("default")
                is_req = parameter_name in required
                req_str = "required" if is_req else "optional"
                default_str = (
                    f", default={parameter_default!r}"
                    if parameter_default is not None or not is_req
                    else ""
                )
                if not is_req and parameter_default is None:
                    default_str = ", default=None"
                parts.append(
                    f"  - {parameter_name}: {parameter_type} "
                    + f"({req_str}{default_str}) - {parameter_description}"
                )
        else:
            parts.append("Parameters: none")

        # Response schema from func_doc
        if api_name in func_doc_schemas:
            response_schema = func_doc_schemas[api_name].get("response", {})
            parts.append("Response schema (return dict must match this exactly):")
            parts.append("```json")
            parts.append(json.dumps(response_schema, indent=2))
            parts.append("```")
        else:
            output_type = _get_string(tool, "output_type", "dict")
            output_description = _get_string(tool, "output_description")
            parts.append(f"Return type: {output_type}")
            if output_description:
                parts.append(f"Return description: {output_description}")

    # Invocation examples
    if api_name in examples_by_fn:
        examples = select_diverse_examples(examples_by_fn[api_name], max_examples=3)
        parts.append(f"Invocation examples ({len(examples)} shown):")
        for index, example in enumerate(examples):
            parts.append(f"  Example {index + 1}:")
            parts.append(f"    call_string: {_get_string(example, 'call_string')}")
            parts.append("    arguments: " + json.dumps(example.get("arguments", {})))
            initial_config = _get_json_object(example, "initial_config")
            example_config = initial_config.get(config_key, {})
            if example_config:
                config_summary = json.dumps(example_config, indent=2)[:200]
                parts.append(f"    state at call time: {config_summary}")
            user_message = _get_string(example, "user_message")
            parts.append(f"    user intent: {user_message[:100]}")
        parts.append("")

    # Special instructions per class
    parts.append("### Special instructions:")
    if class_key == "gorilla_file_system":
        parts.append(
            "- The class must track a 'current_dir' (list of path components, e.g. ['workspace', 'document'])"
        )
        parts.append(
            "- root is a nested dict: {name: {type: 'directory'|'file', contents: {...}, content: '...'}}"
        )
        parts.append(
            "- cd() changes current_dir, ls() lists current_dir contents, mkdir/touch create entries"
        )
        parts.append(
            "- echo() writes content to a file (if file_name given) or returns terminal_output"
        )
        parts.append("- cat() reads file content, mv/rm/cp/rmdir operate on the tree")
        parts.append(
            "- grep/find/diff/sort/tail/wc/du operate on file contents or directory listings"
        )
    elif class_key == "math_api":
        parts.append("- This class is stateless - all methods are pure computation")
        parts.append(
            "- initial_config may contain 'numbers' list, 'precision', 'base', 'value', 'complex_value'"
        )
        parts.append(
            "- Methods should accept these as direct parameters, not from state"
        )
    elif class_key == "ticket_api":
        parts.append("- initial_config has multiple variants. Normalize in __init__:")
        parts.append(
            "  - ticket_list or tickets_queue or support_tickets -> self.ticket_queue (list)"
        )
        parts.append("  - ticket_count or ticket_counter -> self.ticket_counter (int)")
        parts.append("  - current_user -> self.current_user (str)")
        parts.append("  - current_ticket_id -> self.current_ticket_id")
        parts.append("  - priority_levels -> self.priority_levels")
    elif class_key == "posting_api":
        parts.append("- The initial_config key in BFCL data is 'TwitterAPI'")
        parts.append(
            "- Track: tweets (dict by id), comments (dict by tweet_id), retweets, following_list"
        )
        parts.append("- authenticate_twitter checks username/password against state")
    elif class_key == "vehicle_control":
        parts.append(
            "- State is flat key-value pairs (fuelLevel, engineState, doorStatus dict, etc.)"
        )
        parts.append(
            "- displayCarStatus returns different subsets of state based on 'option' param"
        )
        parts.append(
            "- Method names are camelCase (startEngine, lockDoors, etc.) - keep as-is"
        )
    elif class_key == "trading_bot":
        parts.append(
            "- Track: orders (dict by order_id), stocks (dict by symbol), account_info, watch_list"
        )
        parts.append("- place_order adds to orders, fund_account updates balance, etc.")
        parts.append("- market_status can be 'Open' or 'Closed'")
    elif class_key == "travel_booking":
        parts.append(
            "- Track: credit_card_list, booking_record, access_token, budget_limit"
        )
        parts.append(
            "- book_flight adds to booking_record and updates credit card balance"
        )
        parts.append("- authenticate_travel verifies access_token")
    elif class_key == "message_api":
        parts.append(
            "- Track: user_map, messages_sent_map, messages_inbox_map, current_user"
        )
        parts.append("- send_message updates both sent and inbox maps")

    parts.append("")
    parts.append("### Output format:")
    parts.append(
        "Return the complete Python class in a single ```python ... ``` code block."
    )
    parts.append(
        "The class must include __init__(self, initial_config: dict) and all listed methods."
    )
    parts.append(
        "Each method signature must use the exact parameter names from the tool definitions."
    )

    return "\n".join(parts)


def build_schemas_prompt(
    class_key: str,
    tools: list[JsonObject],
) -> str:
    """Build the LLM prompt for generating Pydantic input schemas."""
    class_name = CLASS_KEY_TO_CLASS_NAME[class_key]
    parts: list[str] = []
    parts.append(f"## Task: Generate Pydantic input schemas for {class_name} tools")
    parts.append("")
    parts.append(
        "Generate one Pydantic BaseModel per tool method, named as {MethodName}Input."
    )
    parts.append(
        "For example, if the method is 'get_stock_info', the schema class is 'GetStockInfoInput'."
    )
    parts.append("For camelCase methods like 'startEngine', use 'StartEngineInput'.")
    parts.append("")
    parts.append("Rules:")
    parts.append(
        "- Use exact parameter names from the tool definitions (keep camelCase as-is)"
    )
    parts.append(
        "- Required params have no default; optional params use their declared default or None"
    )
    parts.append(
        "- Use proper Python types: str, int, float, bool, list, dict, Optional[], List[]"
    )
    parts.append("- For enum parameters, use Literal[] from typing")
    parts.append("- Each model must have a docstring")
    parts.append(
        "- Do NOT import anything beyond: from pydantic import BaseModel, Field; from typing import Optional, List, Dict, Any, Literal, Union"
    )
    parts.append("")
    parts.append(f"### Tools ({len(tools)} schemas needed):")
    parts.append("")

    for tool in sorted(tools, key=_tool_sort_key):
        api_name = _get_string(tool, "api_name")
        parameters = _get_json_object(tool, "parameters")
        properties = _get_json_object(parameters, "properties")
        required = _get_string_list(parameters, "required")
        parts.append(f"#### {api_name}")
        parts.append(f"Required: {required}")
        for parameter_name, raw_info in properties.items():
            parameter_info = _require_json_object(
                raw_info, f"parameters.properties.{parameter_name}"
            )
            parameter_type = _get_string(parameter_info, "type", "any")
            parameter_description = _get_string(parameter_info, "description")
            parameter_default = parameter_info.get("default")
            parts.append(
                f"  {parameter_name}: type={parameter_type}, "
                + f"required={parameter_name in required}, "
                + f"default={parameter_default!r}, "
                + f"desc={parameter_description}"
            )
        parts.append("")

    parts.append("Return all schema classes in a single ```python ... ``` code block.")
    parts.append("Format: class {MethodName}Input(BaseModel): ...")

    return "\n".join(parts)


# ─── Auth-gate metadata for sequential test generation ──────────────────────

STATEFUL_AUTH_INFO: dict[str, AuthInfo | None] = {
    "message_api": {
        "login_method": "message_login",
        "login_params": {"user_id": "USR005"},
        "auth_state_field": "current_user",
        "auth_failure_value": "",
        "gated_methods": ["send_message", "delete_message", "search_messages"],
        "non_gated_methods": ["get_user_id", "add_contact", "message_login"],
    },
    "posting_api": {
        "login_method": "authenticate_twitter",
        "login_params": {"username": "genealogy_enthusiast", "password": "testpass"},
        "auth_state_field": "authenticated",
        "auth_failure_value": False,
        "gated_methods": [
            "post_tweet",
            "comment",
            "retweet",
            "follow_user",
            "unfollow_user",
            "mention",
        ],
        "non_gated_methods": [
            "get_tweet",
            "get_tweet_comments",
            "get_user_stats",
            "get_user_tweets",
            "search_tweets",
            "authenticate_twitter",
        ],
    },
    "trading_bot": {
        "login_method": "trading_login",
        "login_params": {"username": "trader", "password": "testpass"},
        "auth_state_field": "authenticated",
        "auth_failure_value": False,
        "gated_methods": [],
        "non_gated_methods": [
            "trading_login",
            "place_order",
            "cancel_order",
            "get_order_details",
            "fund_account",
            "make_transaction",
            "add_to_watchlist",
            "remove_stock_from_watchlist",
            "get_stock_info",
            "get_symbol_by_name",
            "get_available_stocks",
            "filter_stocks_by_price",
            "notify_price_change",
            "update_market_status",
            "update_stock_price",
            "get_transaction_history",
        ],
    },
    "travel_booking": {
        "login_method": "authenticate_travel",
        "login_params": {
            "client_id": "c1",
            "client_secret": "test_secret",
            "refresh_token": "r1",
            "grant_type": "read_write",
            "user_first_name": "M",
            "user_last_name": "S",
        },
        "auth_state_field": "access_token",
        "auth_failure_value": "",
        "token_field": "access_token",
        "gated_methods": [
            "book_flight",
            "cancel_booking",
            "get_credit_card_balance",
            "purchase_insurance",
            "register_credit_card",
            "retrieve_invoice",
            "set_budget_limit",
        ],
        "non_gated_methods": [
            "authenticate_travel",
            "get_flight_cost",
            "get_nearest_airport_by_city",
            "compute_exchange_rate",
            "contact_customer_support",
            "get_budget_fiscal_year",
            "verify_traveler_information",
        ],
    },
    "ticket_api": {
        "login_method": "ticket_login",
        "login_params": {"username": "agent_a", "password": "testpass"},
        "auth_state_field": "current_user",
        "auth_failure_value": "",
        "gated_methods": [],
        "non_gated_methods": [
            "ticket_login",
            "create_ticket",
            "get_ticket",
            "edit_ticket",
            "close_ticket",
            "resolve_ticket",
            "get_user_tickets",
        ],
    },
    "gorilla_file_system": None,
    "math_api": None,
    "vehicle_control": None,
}


def build_sequential_tests_prompt(
    class_key: str,
    class_code: str,
    canonical_config: JsonObject,
    tools: list[JsonObject],
    config_key: str,
) -> str:
    """Build the LLM prompt for generating sequential stateful API tests.

    These tests validate that login/auth → operation sequences work correctly,
    and that operations without prior auth fail predictably.
    """
    _ = config_key
    class_name = CLASS_KEY_TO_CLASS_NAME[class_key]
    auth_info = STATEFUL_AUTH_INFO.get(class_key)

    parts: list[str] = []
    parts.append(f"## Task: Generate sequential stateful API tests for {class_name}")
    parts.append("")
    parts.append(f"File: tests/tools/test_sequential_{class_key}.py")
    parts.append("")
    parts.append("### Overview:")
    parts.append("Generate tests for sequential function-calling scenarios where")
    parts.append("the output of one call is required as input to the next, or where")
    parts.append("authentication state determines whether subsequent calls succeed.")
    parts.append("")

    if auth_info is None:
        parts.append("### NOTE: This class has NO authentication/login gate.")
        parts.append(
            "Generate tests for chained operations where step N depends on step N-1 output."
        )
        parts.append(
            "For example: create resource → get resource → update resource → delete resource."
        )
    else:
        parts.append("### Authentication info:")
        parts.append(f"- Login method: `{auth_info['login_method']}`")
        parts.append(f"- Login params: {json.dumps(auth_info['login_params'])}")
        parts.append(f"- Auth state field: `self.{auth_info['auth_state_field']}`")
        parts.append(f"- Auth failure value: `{auth_info['auth_failure_value']}`")
        if auth_info.get("gated_methods"):
            parts.append(
                f"- Gated methods (require auth): {auth_info['gated_methods']}"
            )
        if auth_info.get("non_gated_methods"):
            parts.append(
                f"- Non-gated methods (work without auth): {auth_info['non_gated_methods']}"
            )
        token_field = auth_info.get("token_field")
        if token_field:
            parts.append(
                f"- Token field: `self.{token_field}` "
                + "(passed as param to gated methods)"
            )
        parts.append("")

    parts.append("### Test classes to generate:")
    if auth_info is not None:
        parts.append(
            "1. `Test{class_name}SequentialCorrect` - Correct login → operation sequences"
        )
        parts.append("   - login then call each gated method")
        parts.append(
            "   - login → operation A → operation B where B depends on A's output"
        )
        parts.append("   - login → operation → verify state change")
        parts.append("")
        parts.append(
            "2. `Test{class_name}SequentialProblematic` - Problematic sequences"
        )
        parts.append("   - call gated method WITHOUT login → should fail")
        parts.append(
            "   - login with wrong credentials → then call gated method → should fail"
        )
        parts.append("   - login → call with invalid params → verify error handling")
        parts.append(
            "   - call gated method with wrong/expired token (for token-based auth)"
        )
    else:
        parts.append(
            "1. `Test{class_name}SequentialCorrect` - Correct chained operation sequences"
        )
        parts.append("   - create → get → verify created data")
        parts.append("   - create → update → get → verify update")
        parts.append("   - multi-step dependent operations")
        parts.append("")
        parts.append(
            "2. `Test{class_name}SequentialProblematic` - Problematic sequences"
        )
        parts.append("   - get nonexistent resource → verify empty/error response")
        parts.append("   - update nonexistent resource → verify error")
        parts.append("   - invalid param chains")

    parts.append("")
    parts.append("### Requirements:")
    parts.append("- Use pytest fixtures (NOT unittest.TestCase)")
    parts.append("- Each test must call 2+ methods in sequence (that's the point!)")
    parts.append(
        "- Start with unauthenticated state in fixtures (set auth field to failure value)"
    )
    parts.append("- Use json.loads(json.dumps(config)) for deep copy in fixtures")
    parts.append("- Test that correct sequences SUCCEED and problematic sequences FAIL")
    parts.append("- Import: import pytest, import json")
    parts.append(f"- Import: from tools.{class_key} import {class_name}")
    parts.append("- Do NOT import or use any LLM client or tool_manager")
    parts.append("- Each test should be self-contained with its own API instance")
    parts.append("")

    parts.append(
        "### initial_config (with auth fields set to failure values for fixtures):"
    )
    unauth_config = copy.deepcopy(canonical_config) if canonical_config else {}
    if auth_info and unauth_config:
        field = auth_info["auth_state_field"]
        if field in unauth_config:
            unauth_config[field] = auth_info["auth_failure_value"]
    parts.append("```json")
    parts.append(json.dumps(unauth_config, indent=2)[:3000])
    parts.append("```")
    parts.append("")

    parts.append("### Class code (for reference):")
    parts.append("```python")
    if len(class_code) > 4000:
        lines = class_code.split("\n")
        kept: list[str] = []
        for line in lines:
            if (
                (
                    line.strip().startswith("def ")
                    or line.strip().startswith("class ")
                    or line.strip().startswith("@")
                    or not line.strip()
                    or line.strip().startswith('"""')
                    or line.strip().startswith("'''")
                )
                or len(kept) > 0
                and not line.startswith(" ")
            ):
                kept.append(line)
        truncated = "\n".join(kept)
        if len(truncated) > 4000:
            truncated = truncated[:4000] + "\n # ... (truncated)"
        parts.append(truncated)
    else:
        parts.append(class_code)
    parts.append("```")
    parts.append("")

    parts.append("### Tools available on this class:")
    for tool in sorted(tools, key=_tool_sort_key):
        api_name = _get_string(tool, "api_name")
        parameters = _get_json_object(tool, "parameters")
        properties = _get_json_object(parameters, "properties")
        parameter_names = list(properties)
        parts.append(f"- `{api_name}({', '.join(parameter_names)})`")
    parts.append("")

    parts.append(
        "Return the complete test file in a single ```python ... ``` code block."
    )
    parts.append(
        "Generate 3-5 correct sequence tests and 3-5 problematic sequence tests."
    )

    return "\n".join(parts)


def build_tests_prompt(
    class_key: str,
    class_code: str,
    schemas_code: str,
    canonical_config: JsonObject,
    tools: list[JsonObject],
    examples_by_fn: dict[str, list[JsonObject]],
    config_key: str,
) -> str:
    """Build the LLM prompt for generating unit tests."""
    _ = config_key
    class_name = CLASS_KEY_TO_CLASS_NAME[class_key]
    parts: list[str] = []
    parts.append(f"## Task: Generate pytest unit tests for {class_name}")
    parts.append("")
    parts.append(f"File: tests/tools/test_{class_key}.py")
    parts.append("")
    parts.append("### Requirements:")
    parts.append(f"- Generate 2-3 tests per method ({len(tools)} methods)")
    parts.append("- Use pytest fixtures for class instance setup")
    parts.append("- Test normal operation, edge cases, and error handling")
    parts.append(f"- Import the class from tools.{class_key}")
    parts.append("- Import schemas from tools.schemas")
    parts.append("- Use the initial_config below for fixture setup")
    parts.append("")
    parts.append("### Test structure:")
    parts.append("```python")
    parts.append("import pytest")
    parts.append("import json")
    parts.append(f"from tools.{class_key} import {class_name}")
    parts.append("```")
    parts.append("")
    parts.append("### initial_config for fixtures:")
    if canonical_config:
        parts.append("```json")
        parts.append(json.dumps(canonical_config, indent=2))
        parts.append("```")
    else:
        parts.append("{}  # stateless class")
    parts.append("")

    # Include representative invocation examples for test inspiration
    parts.append("### Example invocations (use as test case inspiration):")
    for tool in sorted(tools, key=_tool_sort_key):
        api_name = _get_string(tool, "api_name")
        if api_name in examples_by_fn:
            examples = select_diverse_examples(examples_by_fn[api_name], max_examples=2)
            for example in examples:
                call_string = _get_string(example, "call_string", api_name)
                user_message = _get_string(example, "user_message")
                parts.append(f"  {call_string}")
                parts.append(f"    # user intent: {user_message[:100]}")
    parts.append("")

    # Include the class code for reference
    parts.append("### Generated class code (for reference):")
    parts.append("```python")
    if len(class_code) > 4000:
        lines = class_code.split("\n")
        kept: list[str] = []
        for line in lines:
            if (
                line.strip().startswith("def ")
                or line.strip().startswith("class ")
                or line.strip().startswith("@")
                or not line.strip()
                or line.strip().startswith('"""')
                or line.strip().startswith("'''")
                or len(kept) > 0
                and not line.startswith(" ")
            ):
                kept.append(line)
        truncated = "\n".join(kept)
        if len(truncated) > 4000:
            truncated = truncated[:4000] + "\n    # ... (truncated)"
        parts.append(truncated)
    else:
        parts.append(class_code)
    parts.append("```")
    parts.append("")

    # Include the schema code for reference
    if schemas_code:
        parts.append("### Generated schema code (for reference):")
        parts.append("```python")
        parts.append(schemas_code[:3000])
        parts.append("```")
        parts.append("")
    else:
        parts.append(
            "### Available Pydantic schema classes " + "(import from tools.schemas):"
        )
        schemas_path = PROJECT_ROOT / "tools" / "schemas.py"
        try:
            schemas_source = schemas_path.read_text()
        except OSError as exc:
            parts.append(
                "# Schema discovery unavailable: " + f"{type(exc).__name__}: {exc}"
            )
        else:
            available = [
                match.group(1)
                for match in re.finditer(
                    r"^class\s+([A-Za-z_]\w*)\s*\(",
                    schemas_source,
                    re.MULTILINE,
                )
            ]
            for tool in sorted(tools, key=_tool_sort_key):
                api_name = _get_string(tool, "api_name")
                expected_name = (
                    "".join(word.capitalize() for word in api_name.split("_")) + "Input"
                )
                matching = [
                    name for name in available if name.lower() == expected_name.lower()
                ]
                if matching:
                    parts.append(f"# {api_name} -> {matching[0]}")
        parts.append("")

    parts.append(
        "Return the complete test file in a single ```python ... ``` code block."
    )
    parts.append(
        "Include a conftest-style fixture providing an instance of the class with the initial_config."
    )

    return "\n".join(parts)


# ─── Validation ──────────────────────────────────────────────────────────────


def validate_python_code(code: str, filename: str) -> tuple[bool, str]:
    """Check if code compiles without syntax errors."""
    try:
        _ = compile(code, filename, "exec")
        return True, ""
    except SyntaxError as exc:
        return False, f"SyntaxError at line {exc.lineno}: {exc.msg}"
    except (OverflowError, ValueError) as exc:
        return False, str(exc)


def run_tests(
    test_dir: Path, class_key: str, verbose: bool = False
) -> tuple[bool, str]:
    """Run generated tests and return whether pytest succeeded plus output."""
    _ = verbose
    if class_key == "ALL":
        test_path = str(test_dir)
    else:
        test_file = test_dir / f"test_{class_key}.py"
        if not test_file.exists():
            return False, f"Test file not found: {test_file}"
        test_path = str(test_file)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_path,
        "-v",
        "--tb=short",
        "-x",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    output = result.stdout + result.stderr
    success = result.returncode == 0
    return success, output


# ─── Code Extraction Helpers ─────────────────────────────────────────────────


def extract_class_code(text: str, class_name: str) -> str:
    """Extract the class definition from LLM output."""
    code = extract_code_block(text)
    # Verify the class name is present
    if f"class {class_name}" in code:
        return code
    # Try to find it anyway
    return code


def extract_schemas_code(text: str) -> str:
    """Extract schema definitions from LLM output."""
    return extract_code_block(text)


def extract_tests_code(text: str) -> str:
    """Extract test code from LLM output."""
    return extract_code_block(text)


# ─── File Writing ────────────────────────────────────────────────────────────


def write_class_file(
    output_dir: Path,
    class_key: str,
    code: str,
    verbose: bool = False,
) -> None:
    """Write the generated class to its file."""
    filepath = output_dir / f"{class_key}.py"
    # Add a header comment
    header = (
        f'"""Auto-generated {CLASS_KEY_TO_CLASS_NAME[class_key]} implementation."""\n\n'
    )
    if not code.startswith('"""') and not code.startswith("'''"):
        code = header + code
    _ = filepath.write_text(code)
    if verbose:
        print(f"  Wrote {filepath} ({len(code)} bytes)")


def write_schemas_file(
    output_dir: Path,
    all_schemas: dict[str, str],
    verbose: bool = False,
) -> None:
    """Write all accumulated schemas to tools/schemas.py."""
    filepath = output_dir / "schemas.py"
    parts: list[str] = [
        '"""Auto-generated Pydantic input schemas for all BFCL tools."""\n\n'
    ]
    parts.append("from pydantic import BaseModel, Field\n")
    parts.append("from typing import Optional, List, Dict, Any, Literal, Union\n\n")

    for class_key in sorted(all_schemas.keys()):
        schema_code = all_schemas[class_key]
        # Strip any import lines from individual schema blocks (we have them above)
        lines = schema_code.split("\n")
        filtered: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("from pydantic", "import pydantic")):
                continue
            if stripped.startswith(("from typing", "import typing")):
                continue
            filtered.append(line)
        parts.append(f"# ─── {CLASS_KEY_TO_CLASS_NAME[class_key]} ──────────\n\n")
        parts.append("\n".join(filtered))
        parts.append("\n\n")

    _ = filepath.write_text("".join(parts))
    if verbose:
        print(f"  Wrote {filepath}")


def write_test_file(
    test_dir: Path,
    class_key: str,
    code: str,
    verbose: bool = False,
) -> None:
    """Write the generated test file."""
    filepath = test_dir / f"test_{class_key}.py"
    _ = filepath.write_text(code)
    if verbose:
        print(f"  Wrote {filepath} ({len(code)} bytes)")


def write_conftest(test_dir: Path, verbose: bool = False) -> None:
    """Write a basic conftest.py for the test directory."""
    filepath = test_dir / "conftest.py"
    if filepath.exists():
        return
    code = '''"""Shared test fixtures for tools tests."""\n\nimport pytest\nimport json\n'''
    _ = filepath.write_text(code)
    if verbose:
        print(f"  Wrote {filepath}")


# ─── Generation Orchestration ────────────────────────────────────────────────


def generate_class(
    class_key: str,
    client: SimpleLLMClient,
    tools: list[JsonObject],
    func_doc_schemas: dict[str, JsonObject],
    canonical_config: JsonObject,
    examples_by_fn: dict[str, list[JsonObject]],
    config_key: str,
    output_dir: Path,
    test_dir: Path,
    skip_existing: bool,
    verbose: bool,
    max_retries: int = 2,
    skip_tests: bool = False,
) -> tuple[bool, str, str]:
    """Generate class + schemas + tests for one class key.

    Returns (success, class_code, schemas_code).
    """
    class_name = CLASS_KEY_TO_CLASS_NAME[class_key]
    class_file = output_dir / f"{class_key}.py"

    # Check skip-existing
    if skip_existing and class_file.exists():
        print(f"  [SKIP] {class_name} - class file already exists")
        existing_code = class_file.read_text()
        return True, existing_code, ""

    print(f"\n{'=' * 60}")
    print(f"Generating: {class_name} ({len(tools)} tools)")
    print(f"{'=' * 60}")

    # ── Step 1: Generate class code ──
    print("  [1/3] Generating class code...")
    prompt = build_class_prompt(
        class_key, tools, func_doc_schemas, canonical_config, examples_by_fn, config_key
    )

    class_code = ""
    for attempt in range(max_retries + 1):
        response = call_llm(client, SYSTEM_PROMPT, prompt, verbose=verbose)
        class_code = extract_class_code(response, class_name)

        valid, error = validate_python_code(class_code, f"{class_key}.py")
        if valid:
            # Verify all methods are present
            missing: list[str] = []
            for tool in tools:
                method_name = _get_string(tool, "api_name")
                if f"def {method_name}" not in class_code:
                    missing.append(method_name)

            if not missing:
                print(
                    f"  [1/3] ✓ Class code generated and validated ({len(class_code)} chars)"
                )
                break
            else:
                print(f"  [1/3] ⚠ Missing methods: {missing}")
                if attempt < max_retries:
                    prompt += f"\n\n### MISSING METHODS (generate these too):\n{missing}\nPlease include ALL methods."
        else:
            print(f"  [1/3] ✗ Syntax error: {error}")
            if attempt < max_retries:
                prompt += f"\n\n### PREVIOUS OUTPUT HAD SYNTAX ERROR:\n{error}\nPlease fix and regenerate."

    if not class_code:
        print(
            f"  [1/3] ✗ Failed to generate class code after {max_retries + 1} attempts"
        )
        return False, "", ""

    write_class_file(output_dir, class_key, class_code, verbose)

    # Small delay between LLM calls
    time.sleep(1)

    # ── Step 2: Generate Pydantic schemas ──
    print("  [2/3] Generating Pydantic schemas...")
    schema_prompt = build_schemas_prompt(class_key, tools)

    schemas_code = ""
    for attempt in range(max_retries + 1):
        response = call_llm(client, SYSTEM_PROMPT, schema_prompt, verbose=verbose)
        schemas_code = extract_schemas_code(response)

        valid, error = validate_python_code(
            "from pydantic import BaseModel\n" + schemas_code, "schemas.py"
        )
        if valid:
            print("  [2/3] ✓ Schemas generated and validated")
            break
        else:
            print(f"  [2/3] ✗ Schema syntax error: {error}")
            if attempt < max_retries:
                schema_prompt += f"\n\n### PREVIOUS OUTPUT HAD SYNTAX ERROR:\n{error}\nPlease fix and regenerate."

    time.sleep(1)

    # ── Step 3: Generate unit tests ──
    if skip_tests:
        print("  [3/3] Skipping test generation (--skip-tests)")
    else:
        print("  [3/3] Generating unit tests...")
        test_prompt = build_tests_prompt(
            class_key,
            class_code,
            schemas_code,
            canonical_config,
            tools,
            examples_by_fn,
            config_key,
        )

        tests_code = ""
        for attempt in range(max_retries + 1):
            response = call_llm(
                client, SYSTEM_PROMPT, test_prompt, verbose=verbose, max_tokens=8192
            )
            tests_code = extract_tests_code(response)

            valid, error = validate_python_code(tests_code, f"test_{class_key}.py")
            if valid:
                print("  [3/3] ✓ Tests generated and validated")
                break
            else:
                print(f"  [3/3] ✗ Test syntax error: {error}")
                if attempt < max_retries:
                    test_prompt += f"\n\n### PREVIOUS OUTPUT HAD SYNTAX ERROR:\n{error}\nPlease fix and regenerate."

        if tests_code:
            write_test_file(test_dir, class_key, tests_code, verbose)

    return True, class_code, schemas_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM-driven generator for BFCL tool implementations"
    )
    _ = parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated list of class keys to generate (default: all 8)",
    )
    _ = parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "tools"),
        help="Output directory for class modules (default: tools/)",
    )
    _ = parser.add_argument(
        "--test-dir",
        type=str,
        default=str(PROJECT_ROOT / "tests" / "tools"),
        help="Output directory for test files (default: tests/tools/)",
    )
    _ = parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})",
    )
    _ = parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL (default: from OPENAI_API_BASE env var)",
    )
    _ = parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (default: from OPENAI_API_KEY env var)",
    )
    _ = parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip classes that already have generated files",
    )
    _ = parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output including LLM reasoning",
    )
    _ = parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retries per generation step on failure (default: 2)",
    )
    _ = parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip unit test generation (class + schema only)",
    )
    _ = parser.add_argument(
        "--only-tests",
        action="store_true",
        help="Only generate unit tests for existing class files",
    )
    _ = parser.add_argument(
        "--only-schemas",
        action="store_true",
        help="Only generate/update schemas.py from existing class files",
    )
    _ = parser.add_argument(
        "--sequential-tests",
        action="store_true",
        help="Generate sequential stateful API tests for classes with auth gates",
    )
    args = parser.parse_args(namespace=CliArgs())

    # Determine which classes to generate
    all_class_keys = list(CLASS_KEY_TO_CLASS_NAME.keys())
    if args.classes:
        target_classes = [c.strip() for c in args.classes.split(",")]
        invalid = [c for c in target_classes if c not in all_class_keys]
        if invalid:
            print(f"ERROR: Invalid class keys: {invalid}")
            print(f"Available: {all_class_keys}")
            sys.exit(1)
    else:
        target_classes = all_class_keys

    output_dir = Path(args.output_dir)
    test_dir = Path(args.test_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print("BFCL Tool Implementation Generator")
    print(f"  Model: {args.model}")
    print(f"  Output dir: {output_dir}")
    print(f"  Test dir: {test_dir}")
    print(f"  Classes: {target_classes}")
    print(f"  Skip existing: {args.skip_existing}")

    # ── Load data ──
    print("\nLoading data...")
    all_tools = load_tool_definitions()
    print(f"  Loaded {len(all_tools)} tool definitions")

    func_doc_schemas = load_func_doc_schemas()
    print(f"  Loaded {len(func_doc_schemas)} func_doc schemas")

    all_examples = load_invocation_examples()
    print(f"  Loaded {len(all_examples)} invocation examples")

    tools_by_class = group_tools_by_class(all_tools)
    examples_by_fn = group_examples_by_function(all_examples)
    canonical_configs = get_canonical_initial_configs(all_examples)

    # ── Create LLM client ──
    print(f"\nCreating LLM client ({args.model})...")
    client = create_llm_client(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        verbose=args.verbose,
    )

    # ── Write conftest ──
    write_conftest(test_dir, args.verbose)

    # ── Only-tests mode: generate tests for existing class files ──
    if args.only_tests:
        schemas_by_class: dict[str, str] = {}
        schemas_file = output_dir / "schemas.py"
        if schemas_file.exists():
            schemas_src = schemas_file.read_text()
            for ck in target_classes:
                class_name = CLASS_KEY_TO_CLASS_NAME[ck]
                marker = f"# ─── {class_name}"
                if marker in schemas_src:
                    idx = schemas_src.index(marker)
                    next_marker = None
                    for other_ck in target_classes:
                        other_name = CLASS_KEY_TO_CLASS_NAME[other_ck]
                        other_marker = f"# ─── {other_name}"
                        other_idx = schemas_src.find(other_marker, idx + len(marker))
                        if other_idx > idx and (
                            next_marker is None or other_idx < next_marker
                        ):
                            next_marker = other_idx
                    if next_marker:
                        schemas_by_class[ck] = schemas_src[idx:next_marker]
                    else:
                        schemas_by_class[ck] = schemas_src[idx:]

        print(f"\n{'=' * 60}")
        print("Generating unit tests only (class files already exist)")
        print(f"{'=' * 60}")
        test_results: dict[str, bool] = {}
        for class_key in target_classes:
            class_name = CLASS_KEY_TO_CLASS_NAME[class_key]
            class_file = output_dir / f"{class_key}.py"
            test_file = test_dir / f"test_{class_key}.py"
            if not class_file.exists():
                print(f"  [SKIP] {class_name} - class file not found")
                continue
            if test_file.exists() and args.skip_existing:
                print(f"  [SKIP] {class_name} - test file already exists")
                continue

            class_code = class_file.read_text()
            tools = tools_by_class.get(class_key, [])
            config_key = CLASS_KEY_TO_INITIAL_CONFIG_KEY[class_key]
            canonical_config = canonical_configs.get(config_key, {})
            schemas_code = schemas_by_class.get(class_key, "")

            print(f"\n  Generating tests for {class_name} ({len(tools)} tools)...")
            test_prompt = build_tests_prompt(
                class_key,
                class_code,
                schemas_code,
                canonical_config,
                tools,
                examples_by_fn,
                config_key,
            )

            tests_code = ""
            for attempt in range(args.max_retries + 1):
                response = call_llm(
                    client,
                    SYSTEM_PROMPT,
                    test_prompt,
                    verbose=args.verbose,
                    max_tokens=8192,
                )
                tests_code = extract_tests_code(response)
                valid, error = validate_python_code(tests_code, f"test_{class_key}.py")
                if valid:
                    print(f"  ✓ Tests for {class_name} generated and validated")
                    break
                else:
                    print(f"  ✗ Test syntax error: {error}")
                    if attempt < args.max_retries:
                        test_prompt += f"\n\n### PREVIOUS OUTPUT HAD SYNTAX ERROR:\n{error}\nPlease fix and regenerate."

            if tests_code:
                write_test_file(test_dir, class_key, tests_code, args.verbose)
                test_results[class_key] = True
            else:
                test_results[class_key] = False

            time.sleep(1)

        # Summary for test-only mode
        print(f"\n{'=' * 60}")
        print("Test Generation Summary")
        print(f"{'=' * 60}")
        for ck, ok in test_results.items():
            print(f"  {'✓' if ok else '✗'} {CLASS_KEY_TO_CLASS_NAME[ck]}")
        total = len(test_results)
        passed = sum(1 for v in test_results.values() if v)
        print(f"\n Total: {passed}/{total} test files generated")
        return 0 if passed == total else 1

    # ── Sequential-tests mode: generate sequential stateful API tests ──
    if args.sequential_tests:
        print(f"\n{'=' * 60}")
        print("Generating sequential stateful API tests")
        print(f"{'=' * 60}")
        seq_results: dict[str, bool] = {}
        for class_key in target_classes:
            class_name = CLASS_KEY_TO_CLASS_NAME[class_key]
            class_file = output_dir / f"{class_key}.py"
            seq_test_file = test_dir / f"test_sequential_{class_key}.py"
            if not class_file.exists():
                print(f" [SKIP] {class_name} - class file not found")
                continue
            if seq_test_file.exists() and args.skip_existing:
                print(f" [SKIP] {class_name} - sequential test file already exists")
                continue

            class_code = class_file.read_text()
            tools = tools_by_class.get(class_key, [])
            config_key = CLASS_KEY_TO_INITIAL_CONFIG_KEY[class_key]
            canonical_config = canonical_configs.get(config_key, {})

            print(
                f"\n Generating sequential tests for {class_name} ({len(tools)} tools)..."
            )
            seq_prompt = build_sequential_tests_prompt(
                class_key, class_code, canonical_config, tools, config_key
            )

            seq_code = ""
            for attempt in range(args.max_retries + 1):
                response = call_llm(
                    client,
                    SYSTEM_PROMPT,
                    seq_prompt,
                    verbose=args.verbose,
                    max_tokens=8192,
                )
                seq_code = extract_tests_code(response)
                valid, error = validate_python_code(
                    seq_code, f"test_sequential_{class_key}.py"
                )
                if valid:
                    print(
                        f" ✓ Sequential tests for {class_name} generated and validated"
                    )
                    break
                else:
                    print(f" ✗ Syntax error: {error}")
                    if attempt < args.max_retries:
                        seq_prompt += f"\n\n### PREVIOUS OUTPUT HAD SYNTAX ERROR:\n{error}\nPlease fix and regenerate."

            if seq_code:
                test_file_path = test_dir / f"test_sequential_{class_key}.py"
                _ = test_file_path.write_text(seq_code)
                if args.verbose:
                    print(f" Wrote {test_file_path}")
                seq_results[class_key] = True
            else:
                seq_results[class_key] = False

            time.sleep(1)

        # Summary for sequential-test mode
        print(f"\n{'=' * 60}")
        print("Sequential Test Generation Summary")
        print(f"{'=' * 60}")
        for ck, ok in seq_results.items():
            print(f" {'✓' if ok else '✗'} {CLASS_KEY_TO_CLASS_NAME[ck]}")
        total = len(seq_results)
        passed = sum(1 for v in seq_results.values() if v)
        print(f"\n Total: {passed}/{total} sequential test files generated")
        return 0 if passed == total else 1

    # ── Generate each class ──
    all_schemas: dict[str, str] = {}
    results: dict[str, bool] = {}

    for class_key in target_classes:
        if class_key not in tools_by_class:
            print(f"\n  [SKIP] {class_key} - no tools found in extracted data")
            continue

        tools = tools_by_class[class_key]
        config_key = CLASS_KEY_TO_INITIAL_CONFIG_KEY[class_key]
        canonical_config = canonical_configs.get(config_key, {})

        success, class_code, schemas_code = generate_class(
            class_key=class_key,
            client=client,
            tools=tools,
            func_doc_schemas=func_doc_schemas,
            canonical_config=canonical_config,
            examples_by_fn=examples_by_fn,
            config_key=config_key,
            output_dir=output_dir,
            test_dir=test_dir,
            skip_existing=args.skip_existing,
            verbose=args.verbose,
            max_retries=args.max_retries,
            skip_tests=args.skip_tests,
        )

        results[class_key] = success
        if schemas_code:
            all_schemas[class_key] = schemas_code

    # ── Write combined schemas file ──
    if all_schemas:
        write_schemas_file(output_dir, all_schemas, args.verbose)

    # ── Ensure tools/__init__.py exists ──
    init_file = output_dir / "__init__.py"
    if not init_file.exists():
        registry = (
            '"""Auto-generated tool class registry."""\n\n'
            + "from typing import Dict, Any, Optional\n\n"
            + "TOOL_CLASSES: Dict[str, str] = {\n"
        )
        registry += "".join(
            f'    "{class_key}": "tools.{class_key}",\n'
            for class_key in sorted(CLASS_KEY_TO_CLASS_NAME)
        )
        registry += "}\n"
        _ = init_file.write_text(registry)
        if args.verbose:
            print(f"  Wrote {init_file}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("Generation Summary")
    print(f"{'=' * 60}")
    for class_key, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {CLASS_KEY_TO_CLASS_NAME[class_key]} ({class_key})")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  Total: {passed}/{total} classes generated successfully")

    # ── Run all tests ──
    if passed == total and total > 0:
        print("\nRunning all generated tests...")
        success, output = run_tests(test_dir, "ALL", args.verbose)
        if success:
            print("  ✓ All tests passed!")
        else:
            print("  ✗ Some tests failed:")
            for line in output.split("\n")[-30:]:
                print(f"    {line}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
