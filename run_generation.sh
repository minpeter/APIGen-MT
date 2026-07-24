#!/usr/bin/env bash
# Thin wrapper around the primary generator. Run from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec "$PY" src/generate_step_by_step.py --help
fi

mkdir -p data/generated
exec "$PY" src/generate_step_by_step.py "$@"
