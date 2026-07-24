I read this as a concise hostile decision brief request - I verified exact local evidence and will give a narrow go/no-go with a minimal contract.

# Hostile decision brief

Decision: **Do not expand APIGen-MT wholesale.** The repo already has simplified APIGen-like machinery; the weakest link is acceptance quality, not missing paper-shaped architecture. Implement only deterministic state/outcome invariants, then make adversarial tests prove they catch what current LLM/heuristic checks accept.

## 5 strongest risks

1. **Fail-open state verification can bless bad data.**  
   Local evidence: `src/apigen_step_by_step.py:1955` defines `verify_state_transition` as LLM-as-judge; `src/apigen_step_by_step.py:2038` explicitly tells the judge to “assume the state change is valid” when uncertain; `src/apigen_step_by_step.py:2063-2073` defaults JSON parsing/judge failures to `is_valid=True`. This is reward-hackable acceptance, not verification.

2. **Tool/order verification is mostly placebo.**  
   Local evidence: `src/apigen_step_by_step.py:1837` marks relevance true on `relevance_score > 0.1 or name_overlap > 0`; `src/apigen_step_by_step.py:1847-1849` returns order correct for empty/non-empty trajectories with “Order appears logical.” `src/apigen_step_by_step.py:2113` aggregates these weak checks into overall pass. This cannot support correctness claims.

3. **Blueprints are thinner than APIGen-MT ground truth.**  
   Local evidence: `src/apigen_multi_turn.py:61-66` `DialogBlueprint` only stores `overall_task`, `num_turns`, and free-form `turns`; APIGen-MT’s project page describes task configs as `q`, ground-truth actions `a_gt`, and expected outputs `o_gt`. Local capability validation is another judge: `src/apigen_multi_turn.py:738-747`, with truncated state context at `src/apigen_multi_turn.py:787`.

4. **Diversity evidence is semantic-near-duplicate filtering, not trajectory diversity.**  
   Local evidence: `src/q_generator.py:7` imports `SentenceTransformer`; `src/q_generator.py:24` instantiates it; `src/q_generator.py:107` uses `similarity_threshold`; `src/q_generator.py:211` rejects only if max cosine similarity exceeds threshold. That says little about tool, policy, state-delta, turn-depth, or edge-case coverage.

5. **End-to-end confidence is structurally weak.**  
   Local evidence: full datapoint success tests are skipped at `tests/unit/test_datapoint_generation.py:43`, `:69`, `:89`, `:130`, `:229`; integration E2E/retry tests are skipped at `tests/integration/test_integration.py:73`, `:90`, `:147`, `:178`. Positive evidence exists for isolation/rollback (`tests/unit/test_state_isolation_and_rollback.py:1`, `:86-96`, `:111-120`, `:128-134`), so the gap is not state snapshotting; it is semantic acceptance.

## 5 primary citations

1. **APIGen-MT arXiv v4** — https://arxiv.org/abs/2504.03601v4  
   Claims two-phase generation, blueprints, LLM reviewer committees, simulated interplay, and benchmark wins. Treat as method/result source, not causal proof for this repo.

2. **APIGen-MT project page** — https://apigen-mt.github.io/  
   Describes `q`, `a_gt`, `o_gt`, validation/refinement, state/output-based trajectory acceptance, tau-bench case study, and xLAM benchmark claims.

3. **tau-bench paper** — https://arxiv.org/abs/2406.12045  
   Primary benchmark source: simulated users, domain APIs/policies, final database-state evaluation, and `pass^k`. Supports outcome-state validation; also warns simulator success is not real-world proof.

4. **BFCL leaderboard** — https://gorilla.cs.berkeley.edu/leaderboard.html  
   Primary leaderboard page: evaluates LLM ability to call tools accurately; overall accuracy is benchmark category aggregation, not proof of local data quality.

5. **BFCL methodology blog** — https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html  
   Primary methodology source for AST matching, executable function evaluation, relevance detection, exact/structural matching, and benchmark-specific scoring.

## Minimal implementation contract

Implement **one deterministic state invariant layer** before adding any more APIGen-MT machinery.

Contract:

```python
validate_state_transition(
    tool_name: str,
    arguments: dict,
    tool_output: Any,
    pre_state: dict[str, dict],
    post_state: dict[str, dict],
) -> {
    "valid": bool,
    "issues": list[str],
    "checked_invariants": list[str],
}
```

Rules:

- Deterministic invalid result **must reject** the step/datapoint.
- LLM judge may only run when no deterministic invariant applies.
- LLM failure must not convert a checked deterministic result into pass.
- Start with existing real Python tool domains already used in tests:
  - message creation/deletion/auth invariants;
  - vehicle fuel/capacity/state-change invariants;
  - entity-existence-before-mutation invariant.
- Reuse existing snapshots from `ToolManager.get_api_state` / `restore_api_state`; do not duplicate rollback infrastructure.

Adversarial tests required:

1. **Judge-fail-open trap:** corrupt judge raises/returns garbage; deterministic invalid message deletion still rejects. This directly guards `src/apigen_step_by_step.py:2071-2073`.
2. **No-op mutation trap:** mutating tool reports success but post-state is unchanged; validator rejects instead of accepting “No state changes detected” from `src/apigen_step_by_step.py:1984-1986`.
3. **Wrong-entity trap:** tool mutates or references an entity ID absent from pre-state; validator rejects even if query/tool names overlap.
4. **Cross-domain collateral trap:** message tool changes vehicle state or auth fields unrelated to the call; validator rejects.
5. **Positive control:** legitimate send/delete/vehicle fuel transition passes using real Python tool instances, following `tests/unit/test_state_isolation_and_rollback.py` style.

EXPAND status: **CONDITIONAL GO for deterministic invariant layer only; NO-GO for broad APIGen-MT expansion, reviewer committees, graph samplers, or BoN simulator work until adversarial invariant tests pass.**
