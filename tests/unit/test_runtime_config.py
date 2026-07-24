"""No-network tests for the shared runtime configuration and CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from runtime_config import DEFAULT_API_BASE, DEFAULT_MODEL, RuntimeConfig

PROJECT_ROOT = Path(__file__).parents[2]


def test_runtime_config_resolves_shared_defaults_without_network() -> None:
    config = RuntimeConfig.from_environment(
        environ={"OPENAI_API_KEY": "test-key"}
    )

    assert config.api_key == "test-key"
    assert config.api_base == DEFAULT_API_BASE
    assert config.model == DEFAULT_MODEL


def test_cli_help_is_available_without_credentials_or_network() -> None:
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    environment.pop("OPENAI_API_BASE", None)
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert DEFAULT_MODEL in result.stdout
    assert "OPENAI_API_KEY or OPENAI_API_BASE not set" not in result.stdout
