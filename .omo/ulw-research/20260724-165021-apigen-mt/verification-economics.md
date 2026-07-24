# Verification Economics

| claim | risk | error cost | verification cost/time | chosen path | decision | outcome | residual risk |
|---|---|---|---|---|---|---|---|
| C001 behavior preservation | high | Silent dataset regression | Moderate | Baseline characterization, focused/full suite, real CLI | verify | pending | pending |
| C002 original method fidelity | high | Wrong upgrade target | Moderate | Original paper + official code + independent summaries | verify | pending | pending |
| C003 follow-up best practice | high | Overfit to one paper | High | Saturation swarm + citation expansion + counter-search | verify | pending | pending |
| C004 state/outcome validation value | high | False-positive trajectories | Moderate | Paper evidence + code execution on adversarial fixture | verify | pending | pending |
| C005 deterministic QA seam | medium | Unverifiable CLI claim | Moderate | Inspect tests/adapters; execute fixture-backed CLI | verify | pending | pending |
| C007 deterministic replay upgrade | high | Corrupt training trajectories accepted | Moderate | Shared replay verifier + adversarial fixtures + CLI mode | verify | selected | Non-replayable tools remain explicitly unavailable until domain implementations exist |
