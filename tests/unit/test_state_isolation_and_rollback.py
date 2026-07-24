"""Prove datapoint isolation and safe rollback without LLM."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import pytest

from tool_manager import ToolManager


class _FakeStep:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class _FakeTC:
    def __init__(self, tool_name: str, arguments: dict):
        self.tool_name = tool_name
        self.arguments = arguments


def _make_manager() -> ToolManager:
    # Load real tools without needing LLM / full BFCL pool file
    tm = ToolManager.__new__(ToolManager)
    tm.tools = []
    tm.tool_map = {}
    tm.use_config_pool = True
    tm._cached_initial_config = None
    # Minimal config with MessageAPI-like structure via real create path if available
    from tools.message_api import MessageAPI
    from tools.math_api import MathAPI

    cfg_a = {
        "message_api": {
            "workspace_id": "WSA",
            "user_count": 1,
            "user_map": {"Alice": "USR001"},
            "messages_sent_map": {},
            "messages_inbox_map": {},
            "message_count": 0,
            "current_user": "",
        }
    }
    cfg_b = {
        "message_api": {
            "workspace_id": "WSB",
            "user_count": 1,
            "user_map": {"Bob": "USR002"},
            "messages_sent_map": {},
            "messages_inbox_map": {},
            "message_count": 0,
            "current_user": "",
        }
    }
    # Monkeypatch generate_random_config to alternate for isolation test
    import tool_manager as tm_mod

    states = {"n": 0, "cfgs": [cfg_a, cfg_b]}

    def _fake_random():
        i = states["n"] % 2
        states["n"] += 1
        return copy.deepcopy(states["cfgs"][i])

    tm_mod.generate_random_config = _fake_random  # type: ignore

    def _reset():
        conf = tm._cached_initial_config or _fake_random()
        tm.python_tool_instances = {
            "message_api": MessageAPI(conf.get("message_api", conf)),
            "math_api": MathAPI({}),
        }

    tm.reset_python_tool_instances = _reset  # type: ignore

    # Wire methods from class that rely on attributes
    tm.initialize_api_state = ToolManager.initialize_api_state.__get__(tm, ToolManager)
    tm.clear_cached_config = ToolManager.clear_cached_config.__get__(tm, ToolManager)
    tm.get_api_state = ToolManager.get_api_state.__get__(tm, ToolManager)
    tm.restore_api_state = ToolManager.restore_api_state.__get__(tm, ToolManager)
    return tm


def test_force_new_between_datapoints_draws_distinct_configs():
    tm = _make_manager()
    tm.initialize_api_state(force_new=True)
    s1 = tm.get_api_state()
    ws1 = s1["message_api"]["workspace_id"]

    tm.initialize_api_state(force_new=True)
    s2 = tm.get_api_state()
    ws2 = s2["message_api"]["workspace_id"]

    assert ws1 != ws2, "force_new=True must pick a new random config between datapoints"


def test_force_new_false_reuses_cached_config():
    tm = _make_manager()
    tm.initialize_api_state(force_new=True)
    s1 = tm.get_api_state()
    tm.python_tool_instances["message_api"].current_user = "USR001"
    tm.initialize_api_state(force_new=False)
    s2 = tm.get_api_state()
    assert s1["message_api"]["workspace_id"] == s2["message_api"]["workspace_id"]
    # reset recreates instances from cached config — current_user cleared
    assert s2["message_api"].get("current_user", "") in ("", None)


def test_restore_api_state_keeps_pre_failure_snapshot():
    tm = _make_manager()
    tm.initialize_api_state(force_new=True)
    baseline = tm.get_api_state()
    # mutate live state
    tm.python_tool_instances["message_api"].current_user = "MUTATED"
    tm.python_tool_instances["message_api"].workspace_id = "MUTATED_WS"
    assert tm.get_api_state()["message_api"]["workspace_id"] == "MUTATED_WS"

    tm.restore_api_state(baseline)
    restored = tm.get_api_state()
    assert restored["message_api"]["workspace_id"] == baseline["message_api"]["workspace_id"]
    assert restored["message_api"].get("current_user", "") == baseline["message_api"].get(
        "current_user", ""
    )


def test_replay_state_uses_baseline_not_new_random():
    """Ship _replay_state: baseline restore must not call force_new path."""
    from apigen_step_by_step import StepByStepGenerator

    tm = _make_manager()
    tm.initialize_api_state(force_new=True)
    baseline = tm.get_api_state()
    ws = baseline["message_api"]["workspace_id"]

    gen = StepByStepGenerator.__new__(StepByStepGenerator)
    gen.tool_manager = tm
    # mutate
    tm.python_tool_instances["message_api"].workspace_id = "HACKED"
    gen._replay_state([], baseline_state=baseline)
    assert tm.get_api_state()["message_api"]["workspace_id"] == ws
