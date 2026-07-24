# Final Evidence — APIGen-MT Deslop and Research-Backed Upgrade

## Frozen-tree verification (run on final working tree)

- Full test suite: `uv run --all-extras pytest -q -o addopts=''` → 632 passed, 9 skipped.
- Changed-file lint: `uvx ruff check $(changed_py_files)` → all checks passed.
- Changed-file typecheck: `uvx basedpyright $(changed_py_files)` → 0 errors, 0 warnings, 0 notes.
- Compile: `uv run python -m py_compile $(changed_py_files)` → passed.
- Build: `uv build` → source distribution and wheel built successfully; build directory removed after verification.
- Source size audit: control-flow modules under 250 pure LOC; oversized items are standalone CLIs or declarative registry/data modules.
- Slop markers: zero TODO/FIXME/HACK/XXX/pass-only markers in production Python.
- No commit created; HEAD remains `14b947691da1a9ef91761080cf6f54fc06352776`.

## Real CLI surface evidence

- Isolated base-install help: `uv run --isolated --no-project --with ... python main.py --help` → exit 0, `--verify-trajectory` present, no `transformers` required.
- Direct process-substitution CLI:
  - Valid MessageAPI fixture → status `verified`, exit 0.
  - Tampered output fixture → status `failed`/output mismatch, exit 1.
  - Unavailable `remote_tool` fixture → status `unavailable`/is_valid false, exit 2.
  - Malformed `{}` input → validation error, exit 2.
- Cleanup: process substitutions closed automatically; no persistent files, processes, ports, or directories.

## Research evidence

- Original paper v4, official dataset/xLAM repo SHAs, and implementation-to-code mapping: `.omo/ulw-research/20260724-165021-apigen-mt/original-paper-report.md`, `official-release-report.md`, `paper-implementation-map.md`.
- Follow-up literature inventory with 142 citation edges and direct/adjacent separation: `citation-report.md`.
- Stateful validation, synthetic-data, user-simulation, and skeptical audit briefs: `stateful-validation-report.md`, `synthetic-data-report.md`, `user-simulation-report.md`, `skeptical-audit.md`.
- Evidence-backed decision matrix and selected deterministic replay contract: `SYNTHESIS.md`.

## RED→GREEN behavior proof

- Pre-implementation failing-first tests for tampered output, tampered post-state, replay exception, ignored state verdict, missing multi-turn final verification, unavailable-is-valid, unsafe filesystem replay, expected-tool mismatch, ignored query/capability/state-judge errors, and multi-turn judge wiring: `tests/unit/test_deterministic_verification.py`, `tests/unit/test_query_generation.py`, `tests/unit/test_generate_step_by_step.py`.
- All listed scenarios now pass; CLI integration tests cover valid, tampered, unavailable, and malformed exits.

## Security/design fixes applied

- `ReplayManager` requires `is_replay_safe`; `ToolManager` marks filesystem tools unsafe.
- `trajectory_replay` preflights all calls before restoring untrusted state; unavailable/unsafe tools return `unavailable`/`is_valid=False` without invoking anything.
- Final acceptance gates on both `deterministic_replay.is_valid` and `status == "verified"` semantics (unavailable fails).
- Resumed multi-turn datapoints run final verification and store the result.
- CLI `--judge-model` is passed to `MultiTurnGenerator`.
- `query_retries` is propagated to blueprint generation.
