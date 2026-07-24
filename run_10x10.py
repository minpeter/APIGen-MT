#!/usr/bin/env python3
"""Generate 10 datapoints x 10 actions (requires API + local tool pool)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from runtime_config import OPENAI_API_BASE_ENV, OPENAI_API_KEY_ENV, RuntimeConfig


def main() -> int:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print(f"Env: {OPENAI_API_KEY_ENV} (required), {OPENAI_API_BASE_ENV} (optional)")
        print("Paths: magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl")
        return 0

    pool = ROOT / "magnet_tool_extraction" / "bfcl_v3_tools_with_outputs.jsonl"
    examples = ROOT / "magnet_tool_extraction" / "bfcl_v3_invocation_examples.jsonl"

    try:
        RuntimeConfig.from_environment()
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if not pool.is_file():
        print(
            f"ERROR: missing tool pool {pool}\n"
            "Generate/download BFCL outputs (see magnet_tool_extraction/README.md).",
            file=sys.stderr,
        )
        return 2

    # Delegate to primary CLI
    os.execvp(
        sys.executable,
        [
            sys.executable,
            str(ROOT / "src" / "generate_step_by_step.py"),
            "--num-datapoints",
            "10",
            "--num-actions",
            "10",
            "--tool-pool",
            str(pool),
            "--invocation-examples",
            str(examples),
            "--output",
            "data/generated/stateful_10x10_datapoints.jsonl",
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
