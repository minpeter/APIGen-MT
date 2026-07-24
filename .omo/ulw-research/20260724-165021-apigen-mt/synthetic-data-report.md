# Decision Brief: Mechanism Selection for the Local APIGen-MT Pipeline

Scope: rank mechanisms by empirical support, implementation cost, and fit to the existing code (`apigen_step_by_step.py`, `prompts.py`, `tool_manager.py`, `config_pool.py`); separate direct APIGen-MT follow-ups from adjacent work; one minimal upgrade; explicit rejects; material leads only.

## 1. Top 7 mechanisms, ranked

Scoring: Evidence = strength/isolation of published empirical support. Cost = incremental work in *this* codebase. Fit = how directly it attaches to existing code. D = direct APIGen-MT mechanism; A = adjacent work.

| # | Mechanism | Evidence | Cost | Fit | Type |
|---|---|---|---|---|---|
| 1 | Committee review (N judges, majority, 4-axis rubric) feeding the existing feedback loop | Strong, isolated ablation: blueprint success 70% with agentic feedback vs 28% without (2504.03601) | Low: N× judge calls at blueprint stage only; reuses `accumulated_feedback` | Exact: `prompts.py` already has single-judge validation + feedback loop | D |
| 2 | State-diff entailment gate (verify final-response claims against pre/post `get_api_state()` diff) | Strong: LLM judges cap at AUROC 0.65 for false success; 45–48% of single-control τ² failures are false successes (2606.09863); APIGen-MT's own `diff_patch` alignment stage | Near-zero: snapshots already exist in `tool_manager.py` | Exact: `get_api_state()`/`restore_api_state()` present | D+A (diff_patch is APIGen-MT; false-success framing is 2606.09863) |
| 3 | Best-of-N user stabilization (N=4 candidates + self-critique) + pass^k recording | Strong, isolated: τ-bench eval variance 11.1→2.6 (gpt-4o), 9.7→4.0 (xLAM-70b) (2504.03601, Table 3); pass^k from 2406.12045 (gpt-4o pass^8 <25% retail) | Low-moderate: 4× user-turn generation + scoring pass per turn | High: user-simulation phase exists; prompt structure in place | D |
| 4 | Failure-trajectory archive with τ-bench fault taxonomy, enabling STILL-2/Agentic-DPO-style reuse | Strong but downstream: Agentic-DPO lifts τ-bench retail 21.7%→41.4% (9B) offline, matching online GRPO (2607.10601); STILL-2 large margins via prefix/suffix conditioning (2402.11651); AgentRefine held-out gains (2501.01702); BranPO contrastive-branch gains (2602.03719) | Near-zero to archive + label (data already flows at failure time); training-side cost is downstream | High: pipeline currently discards failures; APIGen-MT explicitly names this as future work | A (invited by D) |
| 5 | Reverse task recombination (merge validated same-persona trajectories, re-validate from alignment stage) | Moderate: part of the validated APIGen-MT pipeline that produced the 5K dataset (67% Phase-2 success, avg 7 tool calls, max 29 turns); no isolated ablation published | Low: reuses validated components + existing rollback/restore | High: `config_pool.py` personas give the join key | D |
| 6 | Milestone-DAG trajectory validation (snapshot/addition/removal/update/tool_trace_dependent/guardrail; geometric mean; topological matching) | Moderate: ToolSandbox shows SOTA models fail State Dependency / Canonicalization / Insufficient Information categories; metric is a design contribution, not a training-gain number (2408.04682) | Moderate: new validator module | High: pre/post snapshots already supply most similarity signals | A |
| 7 | Difficulty-aware config-pool curriculum (per-variation success tracking, oversample near 50% boundary; tag API dependency depth, read/write ratio) | Strong for the RL-adjacent form: RODS matches a 17K offline pipeline with 400 seeds + ~800 active pool, ~20× fewer trajectories (2606.19047); Tool-Star easy→hard classification across 10+ benchmarks (2505.16410); OrchDAG graph rewards effective with GRPO (2510.24663) | Moderate: instrumentation + scheduling over `generate_random_config()` | Medium-high: variation axis exists; full boundary detection presupposes an RL loop this project lacks, so only the data-side version applies | A |

Ranking rationale, where the table understates it:

- **#1 over #2 for the top slot** only on *isolated* evidence size (2.5× yield is the single cleanest ablation in the whole corpus). In practice #2 is near-zero cost and should ship alongside; see recommendation.
- **#4 is ranked below cheaper items** because its payoff is realized at training time, outside this repo's current scope. The archiving itself is nearly free and preserves optionality; the training recipes (STILL-2 conditioning, Agentic-DPO state-conditioned negatives, BranPO prefix branches) are downstream consumers.
- **#5's evidence is real but unisolated** — no paper ablation separates recombination from the rest of Phase 1. Ranked on cost/fit, not evidence.
- **#7's strong numbers (RODS) are RL-loop numbers.** The transferable part for a data-generation repo is mundane but sound: track success rate per config variation, oversample the boundary, record structural-difficulty features. Do not claim RODS-scale gains from the data-side version.

## 2. Primary sources (verified this session)

arXiv IDs (all confirmed via export.arxiv.org API):

| Work | arXiv | URL |
|---|---|---|
| APIGen-MT (anchor) | 2504.03601 | https://arxiv.org/abs/2504.03601 |
| APIGen (single-turn) | 2406.18518 | https://arxiv.org/abs/2406.18518 |
| xLAM | 2409.03215 | https://arxiv.org/abs/2409.03215 |
| ToolACE | 2409.00920 | https://arxiv.org/abs/2409.00920 |
| ToolACE-R (AAAI'26) | 2504.01400 | https://arxiv.org/abs/2504.01400 |
| ToolACE-MT (ICLR'26) | 2508.12685 | https://arxiv.org/abs/2508.12685 |
| ToolSandbox | 2408.04682 | https://arxiv.org/abs/2408.04682 |
| τ-bench | 2406.12045 | https://arxiv.org/abs/2406.12045 |
| τ²-bench | 2506.07982 | https://arxiv.org/abs/2506.07982 |
| ToolGen (ICLR'25) | 2410.03439 | https://arxiv.org/abs/2410.03439 |
| Evol-Instruct/WizardLM | 2304.12244 | https://arxiv.org/abs/2304.12244 |
| AgentInstruct | 2407.03502 | https://arxiv.org/abs/2407.03502 |
| Magpie | 2406.08464 | https://arxiv.org/abs/2406.08464 |
| STILL-2 / Learning From Failure | 2402.11651 | https://arxiv.org/abs/2402.11651 |
| AgentRefine (ICLR'25) | 2501.01702 | https://arxiv.org/abs/2501.01702 |
| Agentic-DPO | 2607.10601 | https://arxiv.org/abs/2607.10601 |
| BranPO | 2602.03719 | https://arxiv.org/abs/2602.03719 |
| RODS | 2606.19047 | https://arxiv.org/abs/2606.19047 |
| OrchDAG | 2510.24663 | https://arxiv.org/abs/2510.24663 |
| Tool-Star | 2505.16410 | https://arxiv.org/abs/2505.16410 |
| ToRL | 2503.23383 | https://arxiv.org/abs/2503.23383 |
| ToolRL | 2504.13958 | https://arxiv.org/abs/2504.13958 |
| LoopTool | 2511.09148 | https://arxiv.org/abs/2511.09148 |
| Trajectory2Task | 2601.20144 | https://arxiv.org/abs/2601.20144 |
| User-Oriented MT Dialogue Gen | 2601.08225 | https://arxiv.org/abs/2601.08225 |
| False Success characterization | 2606.09863 | https://arxiv.org/abs/2606.09863 |

Official repos with HEAD SHAs (fetched read-only via `gh api`, 2026-07-24):

| Repo | SHA |
|---|---|
| https://github.com/sierra-research/tau-bench | `59a200c6d575d595120f1cb70fea53cef0632f6b` |
| https://github.com/apple/ToolSandbox | `165848b9a78cead7ca7fe7c89c688b58e6501219` |
| https://github.com/SalesforceAIResearch/xLAM | `a88aa3aeddbc2d7d6aa7a87687d1a085c34e2aec` |
| https://github.com/Reason-Wang/ToolGen | `6839374a255810efe69deea4056eec5c55e25802` |
| https://github.com/Schuture/Agentic-DPO | `1c6a39ad4b68ab1fd9b963f39bb41fdb69d9aa76` |
| https://github.com/Rednote-DeepExperience/LoopTool | `51603ae457689e6a43a72b4d166d18d5a4e4059c` |
| https://github.com/YubaoZhao/BranPO | `7863ebcf7837e8b61722d6db5e5e697fb3f863a1` |
| https://github.com/dongguanting/Tool-Star | `df08f67a89b27feda425306cfe892d65f6569f9a` |

Note: the xLAM repo at this SHA hosts models + the 5K dataset, **not** the APIGen-MT generation pipeline (`gh search code blueprint` returns empty) — no pipeline implementation claims are made from it. τ-bench README at this SHA warns tasks are outdated and points to τ²/τ³.

## 3. Direct APIGen-MT follow-ups vs adjacent work

**Direct follow-ups** (mechanisms inside 2504.03601 that the local pipeline partially implements or omits):

- Committee review with majority vote + Correctness/Completeness/Satisfaction/Creativity rubric (local: single judge).
- BoN=4 user stabilization with self-critique (local: absent).
- `diff_patch`-based alignment validation of state change vs stated intent (local: has pre/post verification, no claim-entailment check).
- Reverse task recombination of validated same-persona tasks (local: absent; personas exist).
- API dependency-graph random walk with read/write-biased API sampler, policy sampler, example sampler, randomized sampling frequency (local: `domain_hints.py` + `config_pool.py` cover a subset heuristically; only Vehicle Control hints populated).
- pass^k consistency reporting (local: absent).

**Adjacent work** (independent methods compatible with the pipeline):

- Failure-trajectory reuse: STILL-2 (2402.11651), AgentRefine (2501.01702), Agentic-DPO (2607.10601), BranPO (2602.03719).
- Milestone-DAG evaluation + tool augmentations (distractors, name scrambling, arg renaming, description/type-hint removal): ToolSandbox (2408.04682).
- Fault taxonomy (user/agent/environment × four fault types): τ-bench repo.
- Curriculum: RODS (2606.19047), Tool-Star (2505.16410), OrchDAG (2510.24663).
- Intent perturbation (ambiguous/changing/infeasible): Trajectory2Task (2601.20144).
- False-success detection rationale: 2606.09863.
- Dual-control user tool use: τ²-bench (2506.07982).
- Alternative generator architectures: ToolACE-MT non-autoregressive mask-and-fill (2508.12685); ToolACE API-pool self-evolution (2409.00920).
- Model-aware data loops: LoopTool (2511.09148), ToolACE-R (2504.01400).

## 4. Recommendation

**Ship #1 and #2 together as the minimal upgrade.** If forced to exactly one: **committee review** (#1). It has the largest isolated ablation in the corpus (28%→70% blueprint yield), attacks the pipeline's binding constraint (blueprint acceptance rate), and drops into the existing `accumulated_feedback` loop in `prompts.py` with no structural change: N=3 judge calls, majority vote, structured disagreement reasons as feedback. The state-diff entailment gate (#2) is the near-zero-cost companion — a mechanical check that final-response claims are supported by the `get_api_state()` pre/post diff, justified by judges capping at 0.65 AUROC on false success (2606.09863). Both are direct APIGen-MT mechanisms, not borrowings.

**Explicit rejects:**

- **ToolGen (2410.03439)** — tools as vocabulary tokens require a fixed 47K-tool training vocabulary; incompatible with this project's dynamic per-datapoint tool pools. No applicable piece.
- **τ² dual-control extension (2506.07982)** — new domain semantics (user-side tools over shared state, Dec-POMDP validation); largest scope item, no evidence the current pipeline is bottlenecked on it. Revisit only after items #1–#5.
- **ToolACE-MT non-autoregressive rewrite (2508.12685)** — replaces the simulation architecture rather than augmenting it; quality claims are self-reported and not independently reproduced. Wrong shape for a minimal upgrade; keep as a research lead.
- **LoopTool/ToolACE-R model-aware loops (2511.09148, 2504.01400)** — presuppose a trained student model to probe; this repo generates data, it does not train. Premature.
- **RL reward engineering (ToolRL 2504.13958, ToRL 2503.23383, BranPO training side)** — downstream of this project's scope; the data-side artifacts (difficulty tags, failure archive) are what matter here.
- **Magpie/Evol-Instruct instruction-side tricks (2406.08464, 2304.12244)** — blueprint generation with execution validation already dominates prompt-level instruction evolution; marginal value too low to justify surface area.
- **DiGiT-TC** — could not be re-located on arXiv this session; dropped entirely. No claims rest on it.
- **RFT-specific citation** — ID unresolved (2408.05109 is a Text-to-SQL survey, not the expected paper); failure-trajectory coverage is complete via STILL-2/AgentRefine/Agentic-DPO/BranPO, so nothing is lost.

## EXPAND

Material leads only — each has a primary source, a concrete attachment point, and no duplicate above:

- **Failure archive schema.** Persist `{blueprint, trajectory, pre_state, post_state, error_stage, fault_label}` JSONL at every validation failure, fault-labeled per τ-bench taxonomy (repo @ `59a200c`). This single artifact unlocks three independent downstream recipes at zero generation-time cost: STILL-2 prefix/suffix conditioning (2402.11651), Agentic-DPO state-conditioned one-step negatives using `get_api_state()` states + `invoke_python_tool()` wrong-action sampling, no online rollouts (2607.10601, repo @ `1c6a39a`), and BranPO prefix-truncated contrastive branches (2602.03719, repo @ `7863ebc`).
- **ToolSandbox robustness augmentations as a transform layer** over `get_tools_with_descriptions()`: distractor injection, tool-name scrambling, argument renaming, description/type-hint removal, per-datapoint randomized and recorded in metadata (paper 2408.04682; reference impl apple/ToolSandbox @ `165848b`). Pure data-diversity gain; no validator changes.
- **Intent-perturbation variants from validated trajectories** (Trajectory2Task, 2601.20144): emit ambiguous / changing / infeasible-intent derivatives with expected clarify/adapt/refuse behavior, validated against the same stateful tools. Covers interaction patterns instruction-first generation systematically misses; their finding that SOTA models fail frequently on these three conditions is the demand signal.
- **Structural-difficulty metadata on every blueprint**: API dependency depth, read/write ratio, DAG topology features (OrchDAG 2510.24663; RODS 2606.19047; APIGen-MT Phase-1 samplers). Enables difficulty-stratified sampling (Tool-Star easy→hard, 2505.16410, repo @ `df08f67`) and, if an RL consumer ever exists, RODS-style boundary-matched resampling without re-deriving features.
- **User-simulator behavior contract audit**: 2601.08225 documents task-oriented simulators collapsing to minimal-interaction trajectories. Add a turn-count/incremental-disclosure check to the simulation phase (the τ-bench "lazy user, one piece at a time, ≥5 tool uses" pattern, repo @ `59a200c`) and log per-blueprint simulator variance (pass^k of the generator itself) — cheap early-warning for the exact defect BoN stabilization fixes.
- **Milestone-DAG validator as a second acceptance channel** (ToolSandbox @ `165848b`, 2408.04682): keep exact-sequence checks, add milestone matching (`snapshot/addition/removal/update/tool_trace_dependent/guardrail`, geometric mean, topologically constrained assignment). Accepts valid trajectories that exact matching currently rejects, directly raising Phase-2 yield; the snapshot inputs already exist.
- **Recombination join index**: maintain an index of validated trajectories keyed by `config_pool.py` persona + domain, so reverse recombination (APIGen-MT) becomes a batch job rather than a pipeline stage; re-validate merged trajectories from the alignment stage onward, skipping action validation since components are already verified.
