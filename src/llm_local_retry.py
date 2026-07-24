"""Retry policy for OpenAI-compatible local chat requests."""

from __future__ import annotations

import json
import random
import time

import requests

if __package__:
    from .llm_request_helpers import (
        extract_message_text,
        normalize_timeout,
        request_json_response,
        retry_delay,
    )
else:
    from llm_request_helpers import (
        extract_message_text,
        normalize_timeout,
        request_json_response,
        retry_delay,
    )


def _integer_option(
    options: dict[str, object],
    name: str,
    default: int,
) -> int:
    value = options.get(name, default)
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def request_chat_with_retries(
    url: str,
    headers: dict[str, str],
    messages: list[dict[str, str]],
    options: dict[str, object],
) -> str:
    """Request a chat response with the established retry behavior."""
    max_retries = _integer_option(options, "max_retries", 3)
    request_timeout = normalize_timeout(options.get("timeout", 900))
    api_options = {
        key: value
        for key, value in options.items()
        if key not in ("max_retries", "timeout")
    }
    payload: dict[str, object] = {"messages": messages, **api_options}
    rate_limit_retries = 0
    server_error_retries = 0
    attempt = 0
    response_text = ""

    while attempt < max_retries:
        try:
            response, response_object = request_json_response(
                f"{url}/chat/completions",
                headers,
                payload,
                request_timeout,
            )
            if "choices" not in response_object:
                if response.status_code == 429:
                    rate_limit_retries += 1
                    delay = retry_delay(
                        2, min(rate_limit_retries, 6), 60, 5, random.uniform
                    )
                    print(
                        f"[LLMClient] Rate limited (429), retrying in {delay:.1f}s... "
                        + f"(rate limit retry #{rate_limit_retries})"
                    )
                    time.sleep(delay)
                    continue
                if response.status_code >= 500:
                    server_error_retries += 1
                    if server_error_retries > 10:
                        raise RuntimeError(
                            "Server error persisted after "
                            + f"{server_error_retries} retries"
                        )
                    delay = retry_delay(
                        2, min(server_error_retries, 6), 60, 3, random.uniform
                    )
                    print(
                        f"[LLMClient] Server error {response.status_code}, "
                        + f"retrying in {delay:.1f}s... (server error retry "
                        + f"#{server_error_retries})"
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Unexpected response from API: {response_object}")

            response_text = extract_message_text(
                response_object, "content", "reasoning_content"
            )
            if response_text.strip() or attempt >= max_retries - 1:
                if not response_text.strip():
                    print(
                        f"[LLMClient] Empty response after {attempt + 1} "
                        + "attempts, proceeding with empty string"
                    )
                break
            delay = retry_delay(2, min(attempt, 6), 128, 3, random.uniform)
            print(
                f"[LLMClient] Empty response (attempt {attempt + 1}), "
                + f"retrying in {delay:.1f}s..."
            )
        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout):
            if attempt >= max_retries - 1:
                print(
                    f"[LLMClient] Request failed after {attempt + 1} "
                    + "attempts due to timeout"
                )
                raise
            delay = retry_delay(2, attempt, 512, 3, random.uniform)
            print(
                f"[LLMClient] Request timeout (attempt {attempt + 1}), "
                + f"retrying in {delay:.1f}s..."
            )
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ) as error:
            if attempt >= max_retries - 1:
                raise
            delay = retry_delay(2, attempt, 512, 3, random.uniform)
            print(
                f"[LLMClient] Connection error (attempt {attempt + 1}): "
                + f"{error}, retrying in {delay:.1f}s..."
            )
        except json.JSONDecodeError as error:
            if attempt >= max_retries - 1:
                raise
            delay = retry_delay(2, attempt, 512, 3, random.uniform)
            print(
                f"[LLMClient] Non-JSON response (attempt {attempt + 1}): "
                + f"{error}, retrying in {delay:.1f}s..."
            )
        time.sleep(delay)
        attempt += 1

    return response_text
