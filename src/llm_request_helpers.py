"""Shared HTTP request construction and LLM response parsing helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Protocol, TypeIs

import requests
from pydantic import BaseModel

type JsonObject = dict[str, object]
type LLMMessage = dict[str, str]
type RequestTimeout = float | tuple[float, float] | tuple[float, None]


class JsonLoader(Protocol):
    """Typed boundary around the dynamically typed standard JSON loader."""

    def __call__(self, value: str, /) -> object: ...


class ResponseJsonDecoder(Protocol):
    """Typed boundary around ``requests.Response.json``."""

    def __call__(self) -> object: ...


_JSON_LOADS: JsonLoader = json.loads


def is_object_map(value: object) -> TypeIs[JsonObject]:
    """Narrow a JSON-decoded value to its object representation."""
    return isinstance(value, dict)


def is_object_list(value: object) -> TypeIs[list[object]]:
    """Narrow a JSON-decoded value to a list of opaque values."""
    return isinstance(value, list)


def is_object_tuple(value: object) -> TypeIs[tuple[object, ...]]:
    """Narrow an option value to a tuple before inspecting its entries."""
    return isinstance(value, tuple)


def require_object_map(value: object, *, source: str) -> JsonObject:
    """Validate an object-shaped value at an HTTP or JSON boundary."""
    if not is_object_map(value):
        raise TypeError(f"{source} must be a JSON object")
    return value


def decode_json_object(value: str, *, source: str) -> JsonObject:
    """Decode and validate a JSON object without leaking ``Any``."""
    return require_object_map(_JSON_LOADS(value), source=source)


def _call_decoder(decoder: ResponseJsonDecoder) -> object:
    return decoder()


def _decode_response(response: requests.Response) -> JsonObject:
    return require_object_map(
        _call_decoder(response.json),
        source="LLM response",
    )


def request_json(
    url: str,
    headers: dict[str, str],
    payload: JsonObject,
) -> JsonObject:
    """POST a JSON payload and decode its object response body."""
    response = requests.request(
        "POST",
        url=url,
        headers=headers,
        json=payload,
    )
    return _decode_response(response)


def normalize_timeout(value: object) -> RequestTimeout:
    """Validate the timeout forms accepted by requests."""
    if isinstance(value, (int, float)):
        return value
    if is_object_tuple(value) and len(value) == 2:
        connect, read = value
        if isinstance(connect, (int, float)):
            if read is None:
                return float(connect), None
            if isinstance(read, (int, float)):
                return float(connect), float(read)
    raise TypeError("timeout must be a number or a connect/read pair")


def request_json_response(
    url: str,
    headers: dict[str, str],
    payload: JsonObject,
    timeout: RequestTimeout,
) -> tuple[requests.Response, JsonObject]:
    """POST JSON and return both the response and decoded object body."""
    response = requests.request(
        "POST",
        url=url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    return response, _decode_response(response)


def extract_message_text(response: JsonObject, *fields: str) -> str:
    """Extract the first populated text field from the first chat choice."""
    choices = response.get("choices")
    if not is_object_list(choices) or not choices:
        raise ValueError("LLM response has no choices")
    choice = require_object_map(choices[0], source="LLM choice")
    message = require_object_map(choice.get("message"), source="LLM message")
    for field in fields:
        value = message.get(field)
        if value:
            if not isinstance(value, str):
                raise TypeError(f"LLM message field {field!r} must be text")
            return value
    return ""


def extract_completion_text(response: JsonObject) -> str:
    """Extract completion text from the first response choice."""
    choices = response.get("choices")
    if not is_object_list(choices) or not choices:
        raise ValueError("LLM response has no choices")
    choice = require_object_map(choices[0], source="LLM choice")
    text = choice.get("text")
    if not isinstance(text, str):
        raise TypeError("LLM completion text must be a string")
    return text


def parse_reasoning_response(response: str) -> tuple[str, str]:
    """Separate a ``<think>`` block from the visible response text."""
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else ""
    response_wo_think = re.sub(
        r"<think>.*?</think>", "", response, flags=re.DOTALL
    ).strip()
    return response_wo_think, reasoning


def build_json_messages(
    prompt: str,
    system_prompt: str | None = None,
    schema: BaseModel | None = None,
) -> list[LLMMessage]:
    """Build the system and user messages used for structured output."""
    if not system_prompt and schema is not None:
        system_prompt = (
            "Extract the information.\n"
            f"            follow the schema: {schema.model_json_schema()}\n"
            "            "
        )

    if system_prompt is None:
        system_prompt = (
            "You are an information extraction assistant. "
            "Extract the required information from the user's input and respond "
            "ONLY with a valid, minified JSON object. Do not include any "
            "explanations or extra text. "
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def append_reasoning_message(
    messages: list[LLMMessage],
    reasoning: str,
) -> list[LLMMessage]:
    """Append the assistant reasoning prefill used for the final JSON call."""
    return [
        *messages,
        {
            "role": "assistant",
            "content": "<think>\n" + reasoning + "\n</think>\n",
        },
    ]


def build_response_format(
    schema: BaseModel | None = None,
    strict: bool = False,
) -> JsonObject:
    """Build the provider-specific structured-output descriptor."""
    if schema is None:
        return {"type": "json_object"}

    schema_value: object = schema.model_json_schema()
    descriptor: JsonObject = {"schema": schema_value}
    if strict:
        descriptor.update(name="json_schema_response", strict=True)
    return {"type": "json_schema", "json_schema": descriptor}


def parse_json_output(raw_json: object) -> object:
    """Decode a structured response when the provider returned text."""
    return _JSON_LOADS(raw_json) if isinstance(raw_json, str) else raw_json


def retry_delay(
    base_delay: int,
    exponent: int,
    cap: int,
    jitter_cap: int,
    random_uniform: Callable[[float, float], float],
) -> float:
    """Calculate the existing capped exponential delay with jitter."""
    delay = min(base_delay * pow(2.0, min(exponent, 8)), float(cap))
    return delay + random_uniform(0, 1.0) * min(delay, float(jitter_cap))
