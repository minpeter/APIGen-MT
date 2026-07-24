#!/usr/bin/env python3
"""APIGen-MT entrypoints.

Primary generation CLI (help works without API keys)::

    python src/generate_step_by_step.py --help

Optional batch helpers (require OPENAI_API_KEY / OPENAI_API_BASE and a local
tool pool JSONL under magnet_tool_extraction/ — not shipped in git; see
magnet_tool_extraction/README.md)::

    ./run_generation.sh --help
    python run_10x10.py   # exits with clear error if pool/API missing

Legacy Friendli two-phase modules were removed with the BFCL pipeline port.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
