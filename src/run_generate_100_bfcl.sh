#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
else
  PY="$(command -v python3)"
fi
POOL="${TOOL_POOL:-magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl}"
EXAMPLES="${INVOCATION_EXAMPLES:-magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl}"
if [[ ! -f "$POOL" ]]; then
  echo "Missing tool pool: $POOL" >&2
  echo "Download/generate BFCL tool definitions (see magnet_tool_extraction/README.md)." >&2
  exit 2
fi
mkdir -p data/generated
exec "$PY" src/generate_step_by_step.py \
  --num-datapoints 100 \
  --tool-pool "$POOL" \
  --invocation-examples "$EXAMPLES" \
  --output data/generated/apigen_mt_100_datapoints.jsonl \
  "$@"
