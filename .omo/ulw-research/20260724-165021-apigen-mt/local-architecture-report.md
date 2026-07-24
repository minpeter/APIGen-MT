<results>
<files>
<!-- PUBLIC SURFACES / ENTRYPOINTS -->
- /home/minpeter/github.com/minpeter/APIGen-MT/src/generate_step_by_step.py - PRIMARY CLI entrypoint (parse_args :89-203, main :359, run_step_by_step :223, run_multi_turn :270, CheckpointManager :54)
- /home/minpeter/github.com/minpeter/APIGen-MT/main.py - Stub entrypoint; only prints docstring (:22-28). Documents that legacy Friendli two-phase modules were removed.
- /home/minpeter/github.com/minpeter/APIGen-MT/run_10x10.py - Thin wrapper; os.execvp delegates to generate_step_by_step.py (:42-56)
- /home/minpeter/github.com/minpeter/APIGen-MT/run_generation.sh - Bash wrapper setting PYTHONPATH, execs generate_step_by_step.py
- /home/minpeter/github.com/minpeter/APIGen-MT/tools/__init__.py - Public tool registry: TOOL_CLASSES + create_tool_instance() factory (:25-44)
- /home/minpeter/github.com/minpeter/APIGen-MT/scripts/generate_tool_implementations.py - Stage-2 codegen CLI (LLM generates tools/*.py + schemas + tests)

<!-- CORE GENERATION / VALIDATION -->
- /home/minpeter/github.com/minpeter/APIGen-MT/src/apigen_step_by_step.py - CORE: StepByStepGenerator (:97), 9 pydantic models (:18-94), 3-stage pipeline, all verify_* methods. ~2189 LOC (largest).
- /home/minpeter/github.com/minpeter/APIGen-MT/src/apigen_multi_turn.py - MultiTurnGenerator(StepByStepGenerator) (:69); adds blueprint + cross-turn validation, reuses parent _stage2/verify_*. ~1473 LOC.
- /home/minpeter/github.com/minpeter/APIGen-MT/src/tool_manager.py - ToolManager (:632): Python-impl invocation + LLM virtual-sim fallback, config pool wiring, state get/restore. ~1704 LOC.
- /home/minpeter/github.com/minpeter/APIGen-MT/src/llm_client.py - LLMClient (:128, raw requests) + LocalOpenAILLMClient (:341, openai SDK, retries, token tracking). ~636 LOC.
- /home/minpeter/github.com/minpeter/APIGen-MT/src/config_pool.py - Diverse initial-state pools + generate_random_config (:772) + generate_query_seed (:762). ~794 LOC (mostly static data :34-760).
- /home/minpeter/github.com/minpeter/APIGen-MT/src/function_schema.py - get_function_schema (:10); used by tool_manager.py:1. LIVE.
- /home/minpeter/github.com/minpeter/APIGen-MT/src/domain_hints.py - DOMAIN_HINTS (:7) + get_domain_hints (:49); used by apigen_multi_turn.py:26. LIVE.
- /home/minpeter/github.com/minpeter/APIGen-MT/src/llm_debug_logger.py - log_llm_call (:10); used by llm_client.py:8. LIVE.

<!-- DEAD / CANDIDATE-DEAD -->
- /home/minpeter/github.com/minpeter/APIGen-MT/src/prompts.py - DEAD: StepByStepPrompts imported at apigen_step_by_step.py:14 but ZERO call sites; prompts are inline. ~150 LOC.
- /home/minpeter/github.com/minpeter/APIGen-MT/src/tool_simulation.py - DEAD: returns {'status':'deprecated'} (:28); no external importers. ~353 LOC.
- /home/minpeter/github.com/minpeter/APIGen-MT/src/q_generator.py - DEAD: standalone legacy query generator (sentence-transformers dedup); only self-references in __main__. ~351 LOC.

<!-- TOOL IMPLEMENTATIONS (Stage-2 output, deterministic) -->
- /home/minpeter/github.com/minpeter/APIGen-MT/tools/math_api.py - 17x broad `except Exception: return {"result":0.0}` (silent wrong-default hazard)
- /home/minpeter/github.com/minpeter/APIGen-MT/tools/gorilla_file_system.py - Stateful FS impl
- /home/minpeter/github.com/minpeter/APIGen-MT/tools/message_api.py - Stateful messaging impl
- /home/minpeter/github.com/minpeter/APIGen-MT/tools/posting_api.py - Stateful social impl
- /home/minpeter/github.com/minpeter/APIGen-MT/tools/ticket_api.py - Stateful ticketing impl
- /home/minpeter/github.com/minpeter/APIGen-MT/tools/trading_bot.py - Stateful trading impl
- /home/minpeter/github.com/minpeter/APIGen-MT/tools/travel_booking.py - Stateful travel impl
- /home/minpeter/github.com/minpeter/APIGen-MT/tools/vehicle_control.py - Stateful vehicle impl
- /home/minpeter/github.com/minpeter/APIGen-MT/tools/schemas.py - Generated pydantic input schemas

<!-- TESTS / DETERMINISTIC SEAMS -->
- /home/minpeter/github.com/minpeter/APIGen-MT/tests/conftest.py - Layer-safe fixtures; optional mock import guarded by try/except ImportError
- /home/minpeter/github.com/minpeter/APIGen-MT/tests/mocks/mock_llm_client.py - MockLLMClient: deterministic generate/chat/json_output/get_token_usage matching LocalOpenAILLMClient surface
- /home/minpeter/github.com/minpeter/APIGen-MT/tests/mocks/mock_llm_responses.py - Canned LLM JSON responses (valid + malformed fixtures)
- /home/minpeter/github.com/minpeter/APIGen-MT/tests/mocks/mock_tool_manager.py - MockToolManager with canned schemas/outputs
- /home/minpeter/github.com/minpeter/APIGen-MT/tests/unit/test_state_isolation_and_rollback.py - Monkeypatches generate_random_config (:67); probes state rollback
- /home/minpeter/github.com/minpeter/APIGen-MT/tests/unit/test_verification.py - Exercises verify_* methods
- /home/minpeter/github.com/minpeter/APIGen-MT/tests/integration/test_integration.py - 12 StepByStepGenerator integration scenarios
- /home/minpeter/github.com/minpeter/APIGen-MT/tests/tools/test_smoke.py - Parametrized cross-class smoke tests

<!-- EXTRACTION TOOLKIT (Stage 1, standalone) -->
- /home/minpeter/github.com/minpeter/APIGen-MT/magnet_tool_extraction/extract_bfcl_with_outputs.py - Stage-1 main extractor
- /home/minpeter/github.com/minpeter/APIGen-MT/magnet_tool_extraction/extract_with_returns.py - DUPLICATE simulate_tool_return (:17) + bare `except:` (:325)
- /home/minpeter/github.com/minpeter/APIGen-MT/magnet_tool_extraction/llm_output_predictor.py - LLM output-type prediction; deprecated client_type warning (:50)

<!-- STALE DOCS / ORPHANED ARTIFACTS -->
- /home/minpeter/github.com/minpeter/APIGen-MT/README.md - STALE: claims sequence-only validation; code already has LLM-judge state validation
- /home/minpeter/github.com/minpeter/APIGen-MT/data/ - 9 orphaned legacy Friendli-pipeline .jsonl datasets (2025-05-02)
- /home/minpeter/github.com/minpeter/APIGen-MT/notebook/ - 4 exploratory .ipynb (likely legacy)
</files>

<answer>

## 1. Public surfaces (behavior contract to preserve)

- **Primary CLI**: `src/generate_step_by_step.py`. Flags (`parse_args` :89-203): `--mode {multi-turn,step-by-step}` (default multi-turn), `-n/--num-datapoints` (100), `-t/--num-turns` (10), `-a/--num-actions` (1), `-o/--output`, `--tool-pool`, `--invocation-examples`, `--category`, `-m/--model` (default `minimax/minimax-m2.7`), `--judge-model/--judge-api-base/--judge-api-key`, `--config-pool/--no-config-pool` (default on), `--checkpoint`, `--resume/--no-resume`.
- **Help is keyless**; generation requires `OPENAI_API_KEY`+`OPENAI_API_BASE` (main :376-377; apigen_step_by_step `__main__` :2152-2156). This is the C005/CA002 "deterministic local seam" gap.
- **Wrappers**: `run_10x10.py` (execvp → CLI with `-n 10 -a 10`), `run_generation.sh` (PYTHONPATH + exec CLI). `main.py` is a docstring-printing stub.
- **Library surface**: `tools/__init__.py` exports `TOOL_CLASSES`, `create_tool_instance`. Pydantic datapoint models in `apigen_step_by_step.py:18-94` (`StepByStepDatapoint`, `ConversationTrajectory`, `TrajectoryStep`, `VerificationResult`, etc.) and `apigen_multi_turn.py:29-66` (`MultiTurnDatapoint`, `MultiTurnConversation`, `Turn`, `DialogBlueprint`) define the on-disk JSONL schema — a public contract.

## 2. Core generation/validation flow

`generate_step_by_step.main` → builds `LocalOpenAILLMClient` + `ToolManager` → picks `MultiTurnGenerator` (default) or `StepByStepGenerator` → loops `generate_datapoint`/`generate_multi_turn_datapoint`, appending verified datapoints to JSONL.

**StepByStepGenerator 3-stage pipeline** (`apigen_step_by_step.py`):
- Stage 1 `_stage1_generate_query` (:862) → `generate_user_query` (:348) + `validate_expected_tools` (:265, LLM judge when num_actions≤5).
- Stage 1.5 `_stage1_5_adjust_initial_state` (:975) + `_ensure_user_identity_coherence` (:930) — mutates initial API state to fit the query.
- Stage 2 `_stage2_generate_tools` (:1474): per tool → `_generate_tool_arguments` (:1375) → `_process_placeholders` (:217) → `_simulate_tool_execution` (:606 → ToolManager.invoke_tool) → `_detect_tool_error` (:693) → `verify_output_consistency` (:1890) → **pre/post state snapshot + `verify_state_transition` (:1954, LLM-as-judge)** → `_replay_state` rollback on failure (:1658).
- Stage 3 `_stage3_finalize` (:1689) → `_generate_final_response` (:1799) → `run_full_verification` (:2084).

**MultiTurnGenerator** adds `_stage0_generate_blueprint` (:845), `_verify_blueprint_capabilities` (:738), `_validate_cross_turn_consistency` (:1326), `_validate_posting_api_entities` (:577), `_validate_vehicle_control_queries` (:678), checkpoint resume (`continue_from_checkpoint` :101). It **reuses** parent `_stage2_generate_tools` (called :208, :393) and inherits all `verify_*` — no override. Inheritance is clean here.

## 3. Largest pure-LOC files (approx; derived from symbol tables + tail reads — no `wc` available)

| File | ~LOC | Logic density |
|---|---|---|
| src/apigen_step_by_step.py | 2189 | HIGH (core generator + 9 models + all verify) |
| src/tool_manager.py | 1704 | HIGH (invocation + virtual sim + config) |
| src/apigen_multi_turn.py | 1473 | HIGH (blueprint + cross-turn validation) |
| src/config_pool.py | 794 | LOW (~600 lines static config/persona/city data :34-760) |
| src/llm_client.py | 636 | MED (two client impls, partly redundant) |
| scripts/generate_tool_implementations.py | 814+ | MED (Stage-2 codegen prompts) |
| src/generate_step_by_step.py | ~494 | MED (CLI + runners + checkpoint) |
| src/tool_simulation.py | ~353 | DEAD |
| src/q_generator.py | ~351 | DEAD |

## 4. Duplication hotspots (deslop targets)

- **JSON code-fence extraction duplicated 11×**: the `if "```json" in response_text: split...; start=find("{"); end=rfind("}")+1; json.loads` block appears at apigen_step_by_step.py :297-301, :451-457, :586-592, :674-680, :1057-1063, :1453-1459, :2058-2064 and apigen_multi_turn.py :823-829, :1010-1015. Extract to one `_extract_json` helper. Note `LocalOpenAILLMClient.json_output` (llm_client.py:537) already does robust extraction, but these sites use raw `_safe_llm_generate` + hand parsing.
- **`simulate_tool_return` exists twice**: src/tool_simulation.py:10 (dead) AND magnet_tool_extraction/extract_with_returns.py:17 (standalone copy).
- **Two LLM clients** in llm_client.py: `LLMClient` (raw `requests`, :128) and `LocalOpenAILLMClient` (openai SDK, :341). Pipeline only uses `LocalOpenAILLMClient`; `LLMClient` may be legacy (verify before removing — q_generator.py:304 references it in a dead `__main__`).

## 5. Broad exception boundaries

- **tools/math_api.py**: 17× `except Exception: return {"result": 0.0}` (:24,33,44,101,114,125,136,147,156,167,176,185,255,268,283,292,303). Silently returns a **wrong numeric default** on any error — a correctness hazard disguised as the "never raise" codegen rule. Highest-value narrowing target, but it is LLM-generated Stage-2 output, so fix the generator prompt (`scripts/generate_tool_implementations.py:333-347`) not just the file.
- **magnet_tool_extraction/extract_with_returns.py:325**: bare `except:`.
- **scripts/generate_tool_implementations.py:236, :814**: `except Exception:` (codegen retry paths — likely intentional).
- **src/apigen_multi_turn.py:572** and **verify_state_transition** (apigen_step_by_step.py:2073): `except Exception` → "assume valid" fallback. The state-judge failure mode silently passes trajectories — relevant to C004 fidelity.
- **tests/unit/test_generate_step_by_step.py:243**: `except Exception:` in test (inspect for nondeterminism-masking).

## 6. Dead abstractions (confirmed by reference tracing)

- **src/prompts.py** — `StepBy
...[elided 6680 chars]...
py`, `generate_top5_table.py`, `sample_extraction.py`, `usage_examples.py`) to decide Stage-1 pruning scope separately from `src/` deslop.
4. **Run the test suite once** (`python -m pytest tests/ -q`) to establish the green baseline (C001) before any deletion — requires shell, which this read-only explore pass did not exercise.
5. **Inspect `data/diverse_queries_with_scores_v4.json`** and the 4 notebooks to confirm they are orphaned vs. referenced by any script.
6. **Verify the `[ml]` pyproject extra** (sentence-transformers/transformers) is needed only by dead `q_generator.py` — if so it can be dropped with the module.

Ready to proceed — the map above is sufficient to plan deslop deletions (prompts.py, tool_simulation.py, q_generator.py, JSON-helper extraction, README correction) and to scope the research-backed upgrade (deterministic fail-closed state/outcome validation + fixture-backed CLI seam) with exact blast-radius data.
</next_steps>
</results>
