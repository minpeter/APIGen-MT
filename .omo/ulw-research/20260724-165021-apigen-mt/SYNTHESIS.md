# APIGen-MT Research Synthesis and Upgrade Decision

Date: 2026-07-24

## Research coverage

Eight independent lanes covered the original paper, official releases, forward/backward citations, synthetic tool-data methods, stateful validation, simulated users, hostile counter-search, and this repository's architecture. The citation lane checked 142 Semantic Scholar citing records and converged after two expansion waves; the complete categorized inventory is in `citation-report.md`. Primary-source and SHA-pinned details are preserved in:

- `original-paper-report.md`
- `official-release-report.md`
- `citation-report.md`
- `synthetic-data-report.md`
- `stateful-validation-report.md`
- `user-simulation-report.md`
- `skeptical-audit.md`
- `local-architecture-report.md`
- `paper-implementation-map.md`

The only material retrieval gap is Papers with Code's oversized API response; canonical project, arXiv, OpenReview, Semantic Scholar, Hugging Face, and official repository sources cover the same method and artifact claims.

## What the original paper establishes

APIGen-MT generates a structured blueprint `{instruction, a_gt, o_gt}` from sampled APIs, policies, domain data, and personas; validates actions, outputs, and holistic task quality with iterative feedback; then simulates agent-human interplay and verifies the resulting trajectory against blueprint ground truth. The 5k released trajectories train xLAM-2-fc-r models that improve tau-bench and BFCL performance.

The paper does not isolate every generator component causally. In particular, it does not establish that dependency-graph random walks, reverse recombination, Best-of-N simulation, or a reviewer committee independently cause the reported model gains. The official project releases the dataset and trained-model materials but not the APIGen-MT generator pipeline:

- Paper: https://arxiv.org/abs/2504.03601
- Project: https://apigen-mt.github.io/
- Dataset: https://huggingface.co/datasets/Salesforce/APIGen-MT-5k
- xLAM repository: https://github.com/SalesforceAIResearch/xLAM/tree/a88aa3aeddbc2d7d6aa7a87687d1a085c34e2aec
- Dataset revision: https://huggingface.co/datasets/Salesforce/APIGen-MT-5k/tree/abc4a517d67c541f85f6470cbd8fd3186b36830e

## What follow-up work changes

### Direct synthesis lineage

The direct successors consistently strengthen executable verification, data efficiency, or environment coverage:

- ToolACE-MT replaces costly autoregressive simulation with skeleton generation, mask filling, and offline verification: https://arxiv.org/abs/2508.12685
- FunReason-MT combines environment/API graphs, tool-query synthesis, and iterative reasoning: https://arxiv.org/abs/2510.24645
- TOUCAN scales rule-plus-model-validated MCP trajectories: https://arxiv.org/abs/2510.01179
- ToolWeave tracks tool dependencies and parameter provenance: https://arxiv.org/abs/2605.12521
- WRIT varies write/read complexity and evaluates on tau2-bench: https://arxiv.org/abs/2606.02908
- RODS closes the online reward/data loop: https://arxiv.org/abs/2606.19047
- EigenData couples self-evolving synthesis with executable checkers: https://arxiv.org/abs/2601.22607 and https://arxiv.org/abs/2603.05553
- Trajectory2Task generates ambiguous, changing, and infeasible intents under closed-loop verification: https://arxiv.org/abs/2601.20144
- ToolMind argues for turn-level rather than trajectory-only filtering: https://arxiv.org/abs/2511.15718
- LoopTool uses a probe, verify, and expand loop: https://arxiv.org/abs/2511.09148
- PARL-MT adds progress-aware reinforcement learning after data generation: https://arxiv.org/abs/2509.23206
- User-oriented multi-turn synthesis emphasizes user goals over tool lists: https://arxiv.org/abs/2601.08225
- Controllable and verifiable synthesis makes validation an explicit generation control: https://arxiv.org/abs/2604.09813
- Hard-sample mining turns observed failure modes into new data: https://arxiv.org/abs/2601.01498

Environment-scaling successors such as AgentScaler, EnvScaler, ScaleEnv, Agent-World, AutoForge, Simia, and SynthTools matter when building many new executable worlds, but this repository already has local Python domains and currently fails to validate them deterministically:

- https://arxiv.org/abs/2509.13311
- https://arxiv.org/abs/2601.05808
- https://arxiv.org/abs/2602.06820
- https://arxiv.org/abs/2604.18292
- https://arxiv.org/abs/2512.22857
- https://arxiv.org/abs/2511.01824
- https://arxiv.org/abs/2511.09572

### Stateful evaluation consensus

The strongest cross-paper consensus is to validate outcomes in executable environments rather than trust an LLM's narration or require one exact trajectory:

- tau-bench compares the final database state and communicated information: https://arxiv.org/abs/2406.12045
- tau2-bench adds dual-control user/agent interaction while retaining outcome checks: https://arxiv.org/abs/2506.07982
- AppWorld evaluates task-specific state and output assertions in executable apps: https://arxiv.org/abs/2407.18901
- ToolSandbox validates stateful tool interactions and exact milestones: https://arxiv.org/abs/2408.04682
- AgentDojo demonstrates executable utility/security evaluation under adversarial instructions: https://arxiv.org/abs/2406.13352
- SABER/tau2-verified finds invalid ground truth and policy violations before attributing failures to agents: https://arxiv.org/abs/2512.07850

This consensus does not make exact action sequences useless. Sequence and per-turn checks are valuable data-integrity diagnostics, but semantic task success should be grounded in final outcomes and communicated values.

### Simulated-user evidence

tau-bench and tau2-bench show that user simulators need explicit policy, hidden-goal, refusal, termination, and information-release constraints. UserBench and UserRL extend the user-centric direction:

- https://arxiv.org/abs/2507.22034
- https://arxiv.org/abs/2509.19736

The research does not justify grounded self-play, user-side tools, or simulator reinforcement learning in this repository before it has a reliable executable acceptance layer. Prompt-only realism improvements remain secondary because better-looking conversations can still encode invalid state transitions.

## Local evidence

The current repository already has iterative generation, placeholder resolution, Python tool execution, initial/pre/post state snapshots, and an optional state judge. Its final acceptance layer is weaker than its stored evidence:

1. `verify_state_transition` instructs one LLM judge to assume validity under uncertainty.
2. Judge exceptions return `is_valid=True`.
3. `run_full_verification` omits stored state-verification verdicts.
4. Multi-turn datapoints are assembled with `verification_result=None`.
5. The invocation-order verifier returns true without checking order.
6. Recorded tool outputs and post-states are not replayed or reconciled against the executable environment.
7. Unknown-error detection relies partly on human-language substrings.

These defects can accept tampered or irreproducible training examples even when local Python implementations could prove them invalid.

## Decision matrix

| Candidate | Evidence | Local value | Complexity/risk | Decision |
|---|---|---|---|---|
| Deterministic replay and outcome integrity | Strong across original paper, tau/AppWorld/ToolSandbox lineage, and direct successors | Closes an observable fail-open path using existing snapshots/tools | Moderate; state must always be restored | Implement now |
| Final-state/outcome verifier result in step-by-step and multi-turn data | Strong | Makes stored verification truthful and filterable | Moderate | Implement now |
| Exact expected-tool count/metadata contracts | Local RED tests and paper blueprint structure | Prevents malformed blueprints | Low | Already repaired |
| Lazy optional tokenizer dependency | Executable CLI RED | Restores base CLI surface | Low | Already repaired |
| Multi-judge committee | Original paper description; no released generator or isolated ablation | May reduce one-judge variance | High cost and correlated-judge bias | Defer |
| API dependency-graph random walks | Original and several successors | More coherent sampling | High design surface; validity is currently weaker bottleneck | Defer |
| Reverse recombination | Original paper | More diversity | Can amplify invalid trajectories | Defer |
| Best-of-N user simulation | Original/user-simulator literature | Potential realism | Judge bias/cost; no grounded acceptance | Defer |
| Non-autoregressive synthesis | ToolACE-MT | Lower generation cost | Replaces the current generation architecture | Defer until quality gate is measured |
| Environment scaling/self-play/RL | 2025-2026 successors | Higher scale | Material new domains/training stack | Out of current scope |

## Selected implementation contract

Create one shared deterministic verifier below the generators:

1. Accept an initial API snapshot and a flattened sequence of recorded tool calls with pre-state, output, and post-state evidence.
2. Snapshot the caller's live tool state before verification.
3. Restore the recorded initial state.
4. For each locally implemented tool:
   - compare live relevant pre-state with the recorded pre-state;
   - resolve and invoke the recorded call;
   - compare normalized structured output with the recorded output;
   - compare live relevant post-state with the recorded post-state.
5. Record checked calls, unavailable calls, mismatches, and a stable final-state digest.
6. Restore the caller's original live state in `finally`, including when replay throws.
7. Reject replayed trajectories on any mismatch or execution error.
8. Keep non-replayable tools explicitly marked unavailable; never report them as deterministically verified.
9. Include the deterministic result in step-by-step final acceptance.
10. Flatten all multi-turn steps, run the same verifier, populate `verification_result`, and reject deterministic mismatches.
11. Expose a local CLI verification mode for JSON/JSONL trajectories so valid and tampered fixtures can be tested without an external LLM write.

Artifact integrity and semantic task outcome are kept distinct: replay proves the stored call/output/state record is reproducible; task-specific final-state assertions remain the stronger semantic layer when a domain supplies them.

## Adversarial acceptance scenarios

1. Tampered recorded tool output is rejected.
2. Tampered recorded post-state is rejected.
3. Incorrect pre-state ordering is rejected.
4. Replay exception is rejected and the caller's live state is restored.
5. A legitimate mutating tool transition passes.
6. A legitimate read-only tool with no state change passes.
7. A non-replayable tool is marked unavailable, not verified.
8. A multi-turn trajectory with one tampered turn is rejected.

## Research stop

The expansion graph has converged on executable validity as the immediate bottleneck. Remaining leads concern environment scaling, simulator RL, and downstream model training; none changes this repository's next implementation decision. Research saturation is therefore reached for the requested upgrade.
