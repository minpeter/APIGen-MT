# Claim Graph

## Verified claims digest

No claims verified yet.

## Claims

| claim_id | statement | type | risk | scope | intent ids | supporting observations | contradicting observations | independent groups | convergence | counter-search | primary source | dependencies | status | synthesis location |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C001 | Repository-wide cleanup can preserve shipped behavior under characterization tests. | code | high | repository | I001 | O001 | none | local-baseline | open | pending | local source/tests | baseline | unresolved | pending |
| C002 | APIGen-MT's original pipeline contains concrete mechanisms absent or simplified here. | research | high | method fidelity | I002 | O002, O004, O005 | none | README, paper-core, official-code | converging | official release counter-search complete | https://arxiv.org/abs/2504.03601 | paper/code inspection | verified | synthesis pending |
| C003 | Follow-up research identifies a best-supported subset worth implementing now. | research | high | literature | I002 | O006, O007, O009 | broad-expansion evidence is weak | citation, skeptic, stateful | saturated | skeptic counter-search complete | `citation-report.md`, `stateful-validation-report.md` | saturation waves | verified | `SYNTHESIS.md` |
| C004 | Outcome/state validation is a higher-value upgrade than surface action-sequence matching alone. | causal | high | validation | I003 | O004, O006, O009, O010 | committee mechanism may add recall but lacks released implementation | paper-core, skeptic, stateful, local RED | strong | adversarial RED complete | https://arxiv.org/abs/2504.03601; https://arxiv.org/abs/2406.12045; https://arxiv.org/abs/2407.18901; https://arxiv.org/abs/2408.04682 | C002, C003 | verified | `SYNTHESIS.md` |
| C005 | A deterministic local generation mode is necessary for faithful CLI QA. | design | medium | CLI | I004 | O003, O008 | none | local source, executable surface | strong | base-install counter-scenario complete | local source | architecture map | partial | implementation pending |
| C006 | Final-state and communicated-outcome evaluation is semantically stronger than exact trajectory matching. | research | high | acceptance semantics | I002, I003 | O009 | APIGen-MT still uses expected action sequences for data construction | stateful-validation, paper-core | strong | counter-position retained for artifact integrity only | `stateful-validation-report.md` | C002, C003 | verified | `SYNTHESIS.md` |
| C007 | Deterministic replay is the smallest strongest upgrade for this repository now. | design | high | implementation decision | I002, I003, I004 | O004, O005, O006, O007, O009 | committee/sampler/user-simulation candidates | paper, citation, stateful, skeptic, local | strong | explicit alternative matrix complete | `.omo/ulw-research/20260724-165021-apigen-mt/SYNTHESIS.md` | C002-C006 | verified | `SYNTHESIS.md` |
