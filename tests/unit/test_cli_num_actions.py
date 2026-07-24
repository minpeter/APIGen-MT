"""CLI must accept --num-actions as advertised by run_10x10 and docs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def test_parse_args_accepts_num_actions(monkeypatch):
    import generate_step_by_step as cli

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_step_by_step.py",
            "--num-datapoints",
            "10",
            "--num-actions",
            "10",
            "--tool-pool",
            "/tmp/pool.jsonl",
            "--invocation-examples",
            "/tmp/ex.jsonl",
        ],
    )
    args = cli.parse_args()
    assert args.num_actions == 10
    assert args.num_datapoints == 10


def test_parse_args_short_flag_a(monkeypatch):
    import generate_step_by_step as cli

    monkeypatch.setattr(sys, "argv", ["generate_step_by_step.py", "-a", "7"])
    args = cli.parse_args()
    assert args.num_actions == 7


def test_help_lists_num_actions():
    import generate_step_by_step as cli
    import argparse

    # Build parser the same way parse_args does by invoking --help
    # Capture via parse_args error path is awkward; call internal construction
    # by re-running main module help
    from io import StringIO
    from contextlib import redirect_stdout, redirect_stderr

    buf = StringIO()
    err = StringIO()
    with pytest.raises(SystemExit) as ei:
        with redirect_stdout(buf), redirect_stderr(err):
            sys.argv = ["generate_step_by_step.py", "--help"]
            cli.parse_args()
    assert ei.value.code == 0
    text = buf.getvalue() + err.getvalue()
    assert "--num-actions" in text
    assert "-a" in text


def test_run_10x10_argv_accepted_by_parser(monkeypatch):
    """The exact argv vector run_10x10.py execs must parse."""
    import generate_step_by_step as cli

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_step_by_step.py",
            "--num-datapoints",
            "10",
            "--num-actions",
            "10",
            "--tool-pool",
            "magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl",
            "--invocation-examples",
            "magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl",
            "--output",
            "data/generated/stateful_10x10_datapoints.jsonl",
        ],
    )
    args = cli.parse_args()
    assert args.num_datapoints == 10
    assert args.num_actions == 10
