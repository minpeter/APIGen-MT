# Original APIGen-MT vs. This Repository

Primary sources:

- Paper v4: https://arxiv.org/abs/2504.03601
- Paper HTML: https://arxiv.org/html/2504.03601v4
- Project: https://apigen-mt.github.io/
- Released dataset: https://huggingface.co/datasets/Salesforce/APIGen-MT-5k
- xLAM repository: https://github.com/SalesforceAIResearch/xLAM

## Mechanism map

| Paper mechanism | Empirical status in paper | Local implementation | Fidelity | Upgrade decision |
|---|---|---|---|---|
| API dependency graph and random-walk context sampling | Described as Phase 1 configuration sampling; no isolated ablation establishing the graph sampler's contribution. | `src/config_pool.py` samples domain configurations and `src/config_pool.py:712` builds query seeds, but there is no API argument/output dependency graph or graph walk. | Partial | Do not add before deterministic acceptance; evidence does not isolate its value. |
| Structured task configuration `{instruction, actions, outputs}` | Core Phase 1 artifact and source of Phase 2 ground truth. | `src/apigen_step_by_step.py:18` records tool name/arguments/output and `src/apigen_step_by_step.py:43` records a trajectory, but there is no independent expected-outcome artifact: the actual simulated output doubles as truth. | Partial with circularity risk | Add deterministic replay comparison so recorded actuals cannot validate themselves. |
| Three-stage action, outcome, and holistic validation | Core data-quality gate; paper uses generated Python execution plus model verifiers and iterative feedback. | `src/apigen_step_by_step.py:1890` checks declared output type; `src/apigen_step_by_step.py:1954` asks one judge about state diffs; `src/apigen_step_by_step.py:2084` omits state-verification results from final acceptance. | Material gap | Implement replayable output/pre-state/post-state invariants and include them in final acceptance. |
| Multi-agent verifier committee | Described mechanism; no public generator or isolated committee ablation. | Constructors set one `judge_client` or reuse the generator model (`src/apigen_step_by_step.py:100`); no independent committee or majority policy exists. | Absent | Defer: weak causal evidence and increased correlated-judge cost. |
| Iterative feedback/refinement | Described in Phase 1; accepted configurations are refined or regenerated on verifier feedback. | Query and tool generation collect validation feedback across bounded retries (`src/apigen_step_by_step.py:342`, `src/apigen_step_by_step.py:738`). | Substantial | Preserve, but replace fail-open malformed boundaries with deterministic guards. |
| Simulated human-agent interplay | Core Phase 2 mechanism; behavior directives induce multi-turn interaction. | `src/apigen_multi_turn.py:69` creates a multi-turn blueprint and generated user queries/assistant turns, but a single generator performs the simulation and the assembled datapoint has no final verification result (`src/apigen_multi_turn.py:440-505`). | Partial with acceptance gap | Reuse the deterministic verifier across flattened turns and populate `verification_result`. |
| Expected action and outcome agreement at trajectory completion | Paper verifier compares the trajectory against task-configuration ground truth. | `expected_tools` is generated and used to constrain step selection, but `run_full_verification` checks relevance, a constant-true order function, output shape, and placeholders only (`src/apigen_step_by_step.py:2084-2146`). | Material gap | Enforce exact expected-tool sequence and deterministic replay at the final gate. |
| Policy compliance | Paper conditions tasks on domain policy and includes policy validation. | Prompts mention policy and local tools implement auth/state rules; no structured policy artifact or final policy evaluator is stored. | Partial | Preserve real-tool policy enforcement; do not invent a generic LLM policy judge. |
| Reverse task recombination | Paper describes reverse operation pairing to increase task diversity. | No equivalent mechanism found. | Absent | Defer until accepted-trajectory validity is trustworthy and a diversity metric is selected. |
| Best-of-N user simulation | Paper reports selecting simulation candidates. | No candidate scoring/ranking exists. | Absent | Defer: follow-up counter-search questions simulator-judge reliability and cost. |

## Empirical bounds

- The paper reports model-level gains from training on the completed APIGen-MT dataset, not isolated causal gains for every generator component.
- The official project releases the 5k dataset and trained models, but not the generation pipeline. Reproducing paper-shaped internals therefore requires inference rather than source fidelity.
- The strongest implementable fact shared by paper, stateful-agent benchmarks, and local adversarial inspection is that outcome validity must be grounded in executable environment effects.
- The repository's current state judge is explicitly instructed to assume validity under uncertainty and returns valid on exceptions; its verdict is then omitted from the final `overall_verification_passed` calculation. This is an observable acceptance defect independent of paper benchmark claims.

## Selected paper-fidelity boundary

Implement a deterministic replay verifier for tools with local Python implementations:

1. Restore the recorded initial API state.
2. Before each call, compare live relevant state with the recorded pre-state.
3. Invoke the same tool with the recorded resolved arguments.
4. Compare normalized actual output and relevant post-state with the recorded artifacts.
5. Restore the original live state in `finally`.
6. Reject any mismatch or replay exception; report non-replayable tools explicitly rather than claiming deterministic verification.
7. Include the result in both step-by-step and multi-turn final acceptance.

This upgrades the exact missing acceptance layer without pretending to reproduce unreleased committee, graph-sampler, or Best-of-N code.
