# APIGen-MT Implementation Decision Brief

## Phase 1: Task Configuration & Groundtruth Generation

**Input**: API definitions (Python functions), domain policies, domain data, personas (PersonaHub).

**Algorithm**:
1. **Context Sampling** — API dependency graph (directed; edge A→B if B's args depend on A's output and co-occurrence is policy-permitted). Random walks sample related API subsets. Separate samplers for: write-APIs (core of a_gt), policies, domain data+metadata, personas, few-shot examples. Sampling frequency randomized per iteration.
2. **LLM Generation** — gpt-4o or DeepSeek V3 generates `{thought, instruction(q), actions(a_gt), outputs(o_gt)}` as structured JSON.
3. **3-Stage Validation**:
   - *Stage 1 (Action Validation)*: Format check (JSON structure, valid tool names/arg types) → Execution check (run a_gt in τ-bench sandbox, capture `diff_patch` state delta) → Policy compliance (domain policies compiled as Python unit tests run against execution trace).
   - *Stage 2 (Alignment Validation)*: Committee of diverse LLM judges scores on 4 binary rubric criteria: Correctness, Completeness, Satisfaction, Creativity. Majority vote determines pass/fail.
   - *Stage 3 (Semantic Review & Refinement)*: If avg score ≥ threshold → accept. Otherwise, a Feedback Generator LLM consolidates committee critiques into actionable summary → fed back to generator for next iteration.
4. **Feedback Loop**: Max 3 iterations (retail) / 5 iterations (airline).
5. **Reverse Task Recombination**: Select 2+ validated tasks sharing a persona → concatenate a_gt sequences → re-run policy compliance on combined actions → LLM synthesizes unified q_combined → re-validate from Stage 2 onward (Stage 1 skipped since constituents already passed).

**Acceptance/Rejection**:
- Reject if: format invalid, execution fails, policy unit test fails, committee majority votes fail on any rubric criterion after max iterations.
- Accept if: all Stage 1 checks pass AND committee majority approves AND score ≥ threshold.

## Phase 2: Trajectory Collection via Simulated Interplay

**Algorithm**:
1. **User Simulation** — LLM (gpt-4o) given persona + blueprint intent q. Rules: reveal info incrementally, don't hallucinate, signal `###STOP###` when goal met.
2. **Agent Simulation** — gpt-4o in function-calling mode. Has access to tools + policies but NOT the blueprint.
3. **Turn Loop** — Alternate user/agent turns. Agent executes tool calls in real τ-bench environment.
4. **Stabilization** — Best-of-N (N=4) sampling for user utterances + self-critique scorer (0-10 scale judging factual correctness and conversation flow).
5. **Trajectory Validation** — Compare final environment state against a_gt (state-based) AND agent's final responses against o_gt (output-based). Only r=1 trajectories retained.
6. **Rejection Sampling** — Each task attempted up to 3 times; union of unique successful trajectories collected.

**Acceptance/Rejection**:
- Accept if: environment state matches ground-truth actions AND output matches expected outputs.
- Reject (discard) if: state mismatch, incomplete task, or max turns exceeded.

## Empirically Supported vs. Merely Described

| Mechanism | Status | Evidence |
|-----------|--------|----------|
| Agentic feedback loop | **EMPIRICAL** | 70% vs 28% task config success rate (Fig 4) — 2.5x boost |
| 3-stage validation pipeline | **EMPIRICAL** | Underlies the 70% acceptance; described with implementation detail |
| Reverse Task Recombination | **DESCRIBED** | Algorithm specified (§4.1.3) but no isolated ablation number |
| BoN user stabilization (N=4) | **EMPIRICAL** | Table 3: variance drops 11.1→2.6 (gpt-4o), 9.7→4.0 (xLAM-2-70b); avg SR increases |
| Policy-as-unit-tests | **EMPIRICAL** | Implementation confirmed; catches inter-action conflicts |
| LLM review committee (majority vote) | **EMPIRICAL** | Used in pipeline; rubric scores shown in prompts (Fig 9) |
| State-based + output-based trajectory validation | **EMPIRICAL** | 67% trajectory success rate reported |
| Contrastive use of failed trajectories | **MERELY MENTIONED** | Listed as future work in §6; not implemented |
| Embedding-based deduplication | **NOT IN PAPER** | Absent from v4 text; was inferred in prior summary |
| Persona sampling from PersonaHub | **EMPIRICAL** | Cited [17], used in context preparation |

## Key Numbers

**Pipeline Statistics** (τ-bench, both domains):
- Phase 1 success rate: **70%** (with feedback) / **28%** (without)
- Phase 2 trajectory success rate: **67%**
- Avg tool calls/trajectory: **7**; Avg user turns: **6**; Range: 1–29 turns
- APIs: 15 read + 13 write across retail+airline

**τ-bench** (pass@1, avg ≥5 trials):

| Model | Retail | Airline | Overall |
|-------|--------|---------|---------|
| xLAM-2-70b-fc-r | 67.1 | 45.2 | **56.2** |
| xLAM-2-32b-fc-r | 64.3 | 45.0 | 54.6 |
| xLAM-2-8b-fc-r | 58.2 | 35.2 | 46.7 |
| GPT-4o | 62.8 | 43.0 | 52.9 |
| Claude 3.5 Sonnet (new) | 71.5 | 48.8 | 60.1 |
| Claude 3.7 Sonnet + prompt opt. | 81.2 | 58.4 | **69.8** |

**BFCL v3** (overall / multi-turn accuracy):

| Model | Overall | Multi-Turn |
|-------|---------|------------|
| xLAM-2-70b-fc-r | **78.19** (Rank 1) | **75.12** |
| xLAM-2-8b-fc-r | 72.83 (Rank 4) | 69.25 |
| GPT-4o (FC) | 69.58 (Rank 7) | 41.00 |
| o1 (Prompt) | 67.87 (Rank 11) | 36.00 |

**No per-component ablation table exists** in the paper. The only ablation is the feedback-loop on/off comparison (70% vs 28%).

## Released Code & Artifacts

| Artifact | Available? | URL |
|----------|-----------|-----|
| Pipeline source code | **NO** | GitHub org `apigen-mt` hosts only project page HTML |
| Dataset (5k trajectories) | YES (gated, CC-BY-NC-4.0) | HuggingFace `Salesforce/APIGen-MT-5k` |
| Trained models (xLAM-2-fc-r, 1B–70B) | YES | HuggingFace `Salesforce/xLAM-2-*` |
| Prompts (all stages) | YES (in paper Appendix B) | Figures 8–12 in arXiv HTML |
| τ-bench environment | YES (third-party) | `sierra-research/tau-bench` |

## Limitations

**Stated**: (1) User simulation stochasticity persists despite BoN; (2) Failed trajectories discarded, not leveraged; (3) Multi-stage validation is computationally expensive; (4) Domain-specific sandbox required per new environment.

**Structural**: (5) Training and evaluation domains overlap (τ-bench retail/airline used for both); generalization to unseen domains untested. (6) No human evaluation of trajectory naturalness. (7) GPT-4o as both generator and agent creates circular quality dependency. (8) No official code means reproduction requires re-implementation from prompts alone.

## Primary URLs

1. https://arxiv.org/abs/2504.03601 — Paper abstract/metadata
2. https://arxiv.org/html/2504.03601v4 — Full HTML (v4, all sections + prompts)
3. https://apigen-mt.github.io/ — Project page
4. https://huggingface.co/datasets/Salesforce/APIGen-MT-5k — Dataset
5. https://huggingface.co/collections/Salesforce/xlam-2-67ef5be12949d8dcdae354c4 — Model collection
6. https://github.com/apigen-mt/apigen-mt.github.io — "Code" repo (project page only)
7. https://github.com/sierra-research/tau-bench — τ-bench environment
8. https://gorilla.cs.berkeley.edu/leaderboard.html — BFCL leaderboard
9. https://arxiv.org/abs/2406.12045 — τ-bench paper
10. https://arxiv.org/abs/2409.03215 — xLAM (v1) paper
11. https://arxiv.org/abs/2406.20094 — PersonaHub (persona source)
12. https://arxiv.org/abs/2310.11441 — Reflexion (feedback loop basis)

## EXPAND

- **Dataset schema inspection**: Download `apigen-mt_5k.json` (gated) to verify exact field names, turn distributions, and whether `diff_patch` is included per trajectory.
- **Salesforce AI Research blog**: Potential additional implementation color not in the paper.
- **xLAM-2 model card training data mix**: Exact ratio of APIGen-MT data vs. APIGen v1 data vs. ActionStudio data during joint training.
- **τ-bench pass^k curves at k>5**: Paper shows pass^5; longer-horizon consistency data may exist in supplementary.
- **Citing papers**: Check Semantic Scholar for reproductions or critiques since April 2025.
- **DeepSeek V3 role**: Paper says both gpt-4o and DeepSeek V3 used in "task generation, validation and agent-human interplay stages" — exact role split per stage is unclear.
