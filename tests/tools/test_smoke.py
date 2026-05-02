"""Smoke tests: verify every tool method can be called and returns the correct type."""

import pytest
import json
import importlib
import math

from scripts.generate_tool_implementations import (
    CLASS_KEY_TO_CLASS_NAME,
    CLASS_KEY_TO_INITIAL_CONFIG_KEY,
    load_tool_definitions,
    group_tools_by_class,
    load_invocation_examples,
    group_examples_by_function,
    get_canonical_initial_configs,
)


@pytest.fixture(scope="module")
def data():
    all_tools = load_tool_definitions()
    all_examples = load_invocation_examples()
    return {
        "tools": all_tools,
        "by_class": group_tools_by_class(all_tools),
        "examples": group_examples_by_function(all_examples),
        "configs": get_canonical_initial_configs(all_examples),
    }


def _make_instance(class_key, configs):
    mod = importlib.import_module(f"tools.{class_key}")
    cls_name = CLASS_KEY_TO_CLASS_NAME[class_key]
    cls = getattr(mod, cls_name)
    config_key = CLASS_KEY_TO_INITIAL_CONFIG_KEY[class_key]
    config = configs.get(config_key, {})
    return cls(initial_config=config)


def _find_real_example(examples, api_name):
    if api_name not in examples:
        return None
    for ex in examples[api_name]:
        func = ex.get("function", {})
        args = func.get("arguments", {})
        if args:
            return args
    return None


CLASS_KEYS = list(CLASS_KEY_TO_CLASS_NAME.keys())


@pytest.mark.parametrize("class_key", CLASS_KEYS)
def test_class_instantiation(class_key, data):
    instance = _make_instance(class_key, data["configs"])
    assert instance is not None


@pytest.mark.parametrize("class_key", CLASS_KEYS)
def test_all_methods_exist(class_key, data):
    instance = _make_instance(class_key, data["configs"])
    tools = data["by_class"].get(class_key, [])
    for tool in tools:
        api_name = tool["api_name"]
        assert hasattr(instance, api_name), f"{class_key} missing method: {api_name}"


@pytest.mark.parametrize("class_key", CLASS_KEYS)
def test_methods_are_callable(class_key, data):
    instance = _make_instance(class_key, data["configs"])
    tools = data["by_class"].get(class_key, [])
    for tool in tools:
        api_name = tool["api_name"]
        method = getattr(instance, api_name)
        assert callable(method), f"{api_name} is not callable"


@pytest.mark.parametrize("class_key", CLASS_KEYS)
def test_methods_return_dict(class_key, data):
    """Every method should return a dict (the standard BFCL return type)."""
    instance = _make_instance(class_key, data["configs"])
    tools = data["by_class"].get(class_key, [])
    examples = data["examples"]
    errors = []
    for tool in tools:
        api_name = tool["api_name"]
        method = getattr(instance, api_name)
        real_args = _find_real_example(examples, api_name)
        if real_args:
            try:
                result = method(**real_args)
                if not isinstance(result, dict):
                    errors.append(f"{api_name} returned {type(result).__name__}, expected dict")
            except Exception as e:
                errors.append(f"{api_name}(**{real_args}) raised {type(e).__name__}: {e}")
        else:
            params = tool.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])
            if not required:
                try:
                    result = method()
                    if not isinstance(result, dict):
                        errors.append(f"{api_name}() returned {type(result).__name__}, expected dict")
                except Exception as e:
                    errors.append(f"{api_name}() raised {type(e).__name__}: {e}")
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("class_key", CLASS_KEYS)
def test_real_invocation_examples(class_key, data):
    """Run all real invocation examples and check they don't crash."""
    instance = _make_instance(class_key, data["configs"])
    examples = data["examples"]
    tools = data["by_class"].get(class_key, [])
    errors = []
    tested = 0
    for tool in tools:
        api_name = tool["api_name"]
        if api_name not in examples:
            continue
        for ex in examples[api_name][:5]:
            func = ex.get("function", {})
            args = func.get("arguments", {})
            if not args:
                continue
            method = getattr(instance, api_name)
            try:
                result = method(**args)
                if not isinstance(result, dict):
                    errors.append(f"{api_name}(**args) returned {type(result).__name__}")
                tested += 1
            except TypeError as e:
                sig_mismatch = f"TypeError in {api_name}(**args): {e}"
                errors.append(sig_mismatch)
            except Exception as e:
                errors.append(f"{api_name}(**args) raised {type(e).__name__}: {e}")
    assert not errors, f"{tested} examples tested, {len(errors)} errors:\n" + "\n".join(errors[:20])
