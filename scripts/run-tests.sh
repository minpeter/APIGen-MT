#!/usr/bin/env bash
# Local test runner (not auto-wired as a git hook).
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x .venv/bin/pytest ]]; then
  .venv/bin/pytest "$@"
elif command -v pytest >/dev/null 2>&1; then
  pytest "$@"
else
  python -m pytest "$@"
fi
