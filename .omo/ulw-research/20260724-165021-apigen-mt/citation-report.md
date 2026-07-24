# APIGen-MT Citation Decision Brief

**Seed:** APIGen-MT — *Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay*, Prabhakar et al. (Salesforce). arXiv:2504.03601 (v4, 2025-07-19); **NeurIPS 2025 Datasets & Benchmarks Track** (OpenReview `qk6ORqQ4Cu`). Two-phase: LLM-committee-verified blueprints (`q`, `a_gt`, `o_gt`) → simulated agent–human trajectories; trains xLAM-2-fc-r (1B–70B); beats GPT-4o/Claude-3.5 on τ-bench + BFCL; releases APIGen-MT-5k. **S2 citations: 142.** All evidence below is from primary sources already retrieved (arXiv API, S2 citation/reference edges, OpenReview); no new queries were required — see saturation note.

---

## A. Direct follow-ups, derivatives, successors, benchmarks, critiques (confirmed citers)

### A1. Multi-turn data-synthesis derivatives (core lineage)

| Title | Date | ID / URL | Role |
|---|---|---|---|
| ToolACE-MT: Non-Autoregressive Generation for Agentic Multi-Turn Interaction | 2025-08-18; ICLR 2026 | arXiv:2508.12685 — https://arxiv.org/abs/2508.12685 | Competitor; non-AR skeleton→mask-fill→offline verify; critiques costly AR simulation |
| FunReason-MT: Advanced Data Synthesis for Real-world Multi-Turn Tool-use | 2025-10-28 | arXiv:2510.24645 — https://arxiv.org/abs/2510.24645 | Competitor; env-API graph + tool-query synthesis + iterative CoT; 4B SOTA BFCLv3/v4 |
| TOUCAN: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments | 2025-10-01 | arXiv:2510.01179 — https://arxiv.org/abs/2510.01179 | Successor dataset; 1.5M trajs, ~500 MCPs, rule+model validation |
| ToolWeave: Structured Synthesis of Complex Multi-Turn Tool-Calling Dialogues | 2026-04-03 | arXiv:2605.12521 — https://arxiv.org/abs/2605.12521 | Derivative; dependency-built tools + parameter-provenance; 70B 39.75% BFCL-V3 MT |
| WRIT: Write-Read Intensive Trajectory Synthesis for Multi-Turn User-Facing Agents | 2026-06-01 | arXiv:2606.02908 — https://arxiv.org/abs/2606.02908 | Derivative; write/read complexity axes; 4B > GPT-5.1 no-think on τ²-bench |
| RODS: Reward-Driven Online Data Synthesis for Multi-Turn Tool-Use Agents | 2026-06-17 | arXiv:2606.19047 — https://arxiv.org/abs/2606.19047 | Successor; online RL↔data loop; 400 seeds ≈ 17K offline |
| From Self-Evolving Synthetic Data to Verifiable-Reward RL (EigenData) | 2026-01-30; ICML 2026 sub | arXiv:2601.22607 — https://arxiv.org/abs/2601.22607 | Successor; self-evolving synthesis + executable checkers + GRPO; 73.0% Airline / 98.3% Telecom τ² |
| EigenData: Self-Evolving Multi-Agent Platform for Function-Calling Data Synthesis | 2026-03-05 | arXiv:2603.05553 — https://arxiv.org/abs/2603.05553 | Platform companion to above |
| Trajectory2Task: Robust Tool-Calling Agents with Synthesized Yet Verifiable Data | 2026-01-28 | arXiv:2601.20144 — https://arxiv.org/abs/2601.20144 | Derivative; ambiguous/changing/infeasible intents; closed-loop verify+train |
| ToolMind: Large-Scale, Reasoning-Enhanced Tool-Use Dataset | 2025-11-12 | arXiv:2511.15718 — https://arxiv.org/abs/2511.15718 | Derivative + critique; **turn-level** filtering vs trajectory-level |
| LoopTool: Closing the Data-Training Loop for Robust LLM Tool Calls | 2025-11-12 | arXiv:2511.09148 — https://arxiv.org/abs/2511.09148 | Successor; closed-loop probe→verify→expand; 8B > 32B generator |
| PARL-MT: Multi-Turn Function Calling with Progress Awareness | 2025-09-27 | arXiv:2509.23206 — https://arxiv.org/abs/2509.23206 | Training-method follow-up; progress-aware RL |
| FinToolSyn: Forward Synthesis for Financial Tool-Use Dialogue | 2026-03-25; ACL 2026 | arXiv:2603.24051 — https://arxiv.org/abs/2603.24051 | Domain derivative |
| Unlocking Implicit Experience: Synthesizing Tool-Use Trajectories from Text | 2026-01-15; ACL 2026 | arXiv:2601.10355 — https://arxiv.org/abs/2601.10355 | Derivative (text→trajectories) |
| User-Oriented Multi-Turn Dialogue Generation with Tool Use at Scale | 2026-01-13 | arXiv:2601.08225 — https://arxiv.org/abs/2601.08225 | Derivative |
| Controllable and Verifiable Tool-Use Data Synthesis for Agentic RL | 2026-04-10 | arXiv:2604.09813 — https://arxiv.org/abs/2604.09813 | Derivative |
| Simulating Complex Multi-Turn Tool Calling in Stateless Execution Environments | 2026-01-06 | arXiv:2601.19914 — https://arxiv.org/abs/2601.19914 | Derivative |
| From Failure to Mastery: Generating Hard Samples for Tool-use Agents | 2026-01-04 | arXiv:2601.01498 — https://arxiv.org/abs/2601.01498 | Derivative (hard-sample mining) |
| Training LLMs for Multi-Step Tool Orchestration (Constrained Synthesis + Graduated Rewards) | 2026-03-25 | arXiv:2603.24709 — https://arxiv.org/abs/2603.24709 | Derivative |
| FunReason (single-turn predecessor to FunReason-MT) | 2025-05-26 | arXiv:2505.20192 — https://arxiv.org/abs/2505.20192 | Precursor |
| ToolACE-R: Model-aware Iterative Training and Adaptive Refinement | 2025-04-02; AAAI 2026 | arXiv:2504.01400 — https://arxiv.org/abs/2504.01400 | ToolACE-line follow-up |
| BalanceSFT: Balanced Training Signals and Data Hardness for FC | 2026; ACL 2026 Findings | DOI 10.18653/v1/2026.findings-acl.900 | Training-signal follow-up |
| Progra: Progress-Aware RL for Multi-Turn Function Calling | 2026; ACL 2026 Findings | DOI 10.18653/v1/2026.findings-acl.325 | PARL-MT-line follow-up |
| Beyond Static Toolsets: Self-Evolving LLM Tool Agents | 2026; ACL 2026 Findings | DOI 10.18653/v1/2026.findings-acl.1082 | Tool-evolution follow-up |

### A2. Environment-scaling successors

| Title | Date | ID / URL |
|---|---|---|
| Towards General Agentic Intelligence via Environment Scaling (AgentScaler) | 2025-09-16 | arXiv:2509.13311 — https://arxiv.org/abs/2509.13311 |
| EnvScaler: Scaling Tool-Interactive Environments via Programmatic Synthesis | 2026-01-09; ACL 2026 | arXiv:2601.05808 — https://arxiv.org/abs/2601.05808 |
| ScaleEnv: Scaling Environment Synthesis from Scratch | 2026-02-06 | arXiv:2602.06820 — https://arxiv.org/abs/2602.06820 |
| Agent-World: Scaling Real-World Environment Synthesis | 2026-04-20 | arXiv:2604.18292 — https://arxiv.org/abs/2604.18292 |
| AutoForge: Automated Environment Synthesis for Agentic RL | 2025-12-28 | arXiv:2512.22857 — https://arxiv.org/abs/2512.22857 |
| Simulating Environments with Reasoning Models for Agent Training (Simia) | 2025-11-03 | arXiv:2511.01824 — https://arxiv.org/abs/2511.01824 |
| Environment Scaling for Interactive Agentic Experience Collection: A Survey | 2025-11-12 | arXiv:2511.09586 — https://arxiv.org/abs/2511.09586 |
| Agent World Model: Infinity Synthetic Environments for Agentic RL | 2026-02-10 | arXiv:2602.10090 — https://arxiv.org/abs/2602.10090 |
| Don't Just Fine-tune the Agent, Tune the Environment | 2025-10-11 | arXiv:2510.10197 — https://arxiv.org/abs/2510.10197 |
| EnvSimBench: Benchmark for LLM-Based Environment Simulation | 2026-05-08 | arXiv:2605.07247 — https://arxiv.org/abs/2605.07247 |
| SynthTools: Scaling Synthetic Tools for Agent Development | 2025-11-11 | arXiv:2511.09572 — https://arxiv.org/abs/2511.09572 |

### A3. In-group successors (same Salesforce authors — strongest derivative signal)

| Title | Date | ID / URL |
|---|---|---|
| UserBench: Interactive Gym for User-Centric Agents | 2025-07-29 | arXiv:2507.22034 — https://arxiv.org/abs/2507.22034 |
| UserRL: Training Interactive User-Centric Agent via RL | 2025-09-24 | arXiv:2509.19736 — https://arxiv.org/abs/2509.19736 |

### A4. Benchmark successors / evaluation

| Title | Date | ID / URL |
|---|---|---|
| τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment | 2025-06-09 | arXiv:2506.07982 — https://arxiv.org/abs/2506.07982 |
| Unsafer in Many Turns (MT-AgentRisk + ToolShield) | 2026-02-13 | arXiv:2602.13379 — https://arxiv.org/abs/2602.13379 |
| TurnWise: Gap between Single- and Multi-turn LM Capabilities | 2026-03-17 | arXiv:2603.16759 — https://arxiv.org/abs/2603.16759 |
| A Matter of TASTE: Coverage and Difficulty of Agent Benchmarks | 2026-05-27 | arXiv:2605.28556 — https://arxiv.org/abs/2605.28556 |
| ToolHaystack: Stress-Testing Tool-Augmented LMs | 2025-05-29; EMNLP 2025 | arXiv:2505.23662 — https://arxiv.org/abs/2505.23662 |
| CRMArena-Pro | 2025-05-24; TMLR | arXiv:2505.18878 — https://arxiv.org/abs/2505.18878 |
| CAR-bench: Consistency and Limit-Awareness under Uncertainty | 2026-01-29 | arXiv:2601.22027 — https://arxiv.org/abs/2601.22027 |
| T1-Bench: Multi-Scenario Agents in Real-World Domains | 2026-06-09 | arXiv:2606.11070 — https://arxiv.org/abs/2606.11070 |
| RealMem: Real-World Memory-Driven Interaction | 2026-01-11; ACL 2026 | arXiv:2601.06966 — https://arxiv.org/abs/2601.06966 |
| Beyond Perfect APIs: LLM Agents Under Real-World API Complexity | 2026-01-01 | arXiv:2601.00268 — https://arxiv.org/abs/2601.00268 |
| How Can Input Reformulation Improve Tool Usage on τ-bench | 2025-08-28 | arXiv:2508.20931 — https://arxiv.org/abs/2508.20931 |
| Persona-based Automated Evaluation for AI Teaching Assistants | 2026; IAIT | DOI 10.1145/3816713.3819067 |

### A5. Models trained on / benchmarked against the lineage

Nemotron-Research-Tool-N1 (arXiv:2505.00024); Nemotron Nano 2 (arXiv:2508.14444); rStar2-Agent (arXiv:2508.20722); Qwen3-Coder-Next (arXiv:2603.00729); Sabiá-4 (arXiv:2603.10213); Agent Learning via Early Experience (arXiv:2510.08558); Tool Zero (arXiv:2511.01934, EMNLP 2025); ToolRM reward models (arXiv:2509.11963 / arXiv:2510.26167); CM2 checklist rewards (arXiv:2602.12268); PivotRL (arXiv:2603.21383); AutoTool (arXiv:2603.13348); D-CORE (arXiv:2602.02160); Process-Supervised RL for Multimodal Tool-Use (arXiv:2509.14480); Generalizable E2E Tool-Use RL w/ Synthetic CodeGym (arXiv:2509.17325).

---

## B. Adjacent prior work (backward references — what APIGen-MT builds on; NOT follow-ups)

- **APIGen** (direct single-turn predecessor, same team) — arXiv:2406.18518, NeurIPS 2024 D&B, OpenReview `Jfg3vw2bjx`. 3-stage verification (format→execution→semantic); xlam-function-calling-60k.
- **Magnet** (closest same-problem sibling/baseline) — arXiv:2503.07826, ACL 2025. Graph-translation + contrastive distillation. *Already under local study (`magnet_tool_extraction/`).*
- **τ-bench** (primary eval target) — arXiv:2406.12045.
- **xLAM** (model family extended) — arXiv:2409.03215.
- **ToolACE** — arXiv:2409.00920, ICLR 2025. **ToolDial** — arXiv:2503.00564, ICLR 2025. **ATLaS** — arXiv:2503.02197, ACL 2025. **CRMArena** — arXiv:2411.02305, NAACL 2025. **ToolSandbox** — arXiv:2408.04682.
- Foundational context: Toolformer (2302.04761), Reflexion (2303.11366), CAMEL (2303.17760), Generative Agents (2304.03442), Tool-Learning survey (2304.08354), LATM (2305.17126), InterCode (2306.14898), Llama-3 (2407.21783), 1B-Personas (2406.20094), Agent-FLAN (2403.12881), LlamaFactory (2403.13372), Agent S (2410.08164), SWE-Gym (2412.21139), SWE-Search (2410.20285), MAG-V (2412.04494), LLM-Agents-Making-Tools (2502.11705), Learn-by-interact (2501.10893), IntellAgent (2501.11067), MultiChallenge (2501.17399), OWL (2505.23885), ActionStudio (2503.22673), Lighthouse-of-Language (2503.16024), Multi-Agent-Sim post-training (2410.14251), Compositional-Instruction-Tuning (2410.12952), Forest-of-Thought (2412.09078), TapeAgents (2412.08445), HammerBench (2412.16516).

**Tangential forward citers** (cite APIGen-MT but off the data-generation core; listed for completeness, not material): Test-Time/Grounded TTA for Agents (2511.04847); Experience-Evolving MT Tool-Use w/ Memory (2512.07287); Verification-Guided Context Optimization (2512.13860); CoVe (2603.01940); TopoCurate (2603.01714); ISE (2606.11520); PACT (2606.16215); SENTINEL (2606.12908); Aegis (2508.19504); Matrix (2511.21686); Klear-AgentForge (2511.05951); ASTRA (2601.21558); OpenResearcher (2603.20278); Enterprise Deep Research (2510.17797); EnterpriseLab (2603.21630); VoxMind (2604.15710); UniToolCall (2604.11557); HTAA (2604.10917); ParaTool (2605.29561); On Effectiveness/Efficiency of Agentic Tool-calling+RL (2606.00135); Adapting-the-Interface-Not-the-Model (2605.22166); RecoAtlas (2605.18805); Ares (2603.07915); ARTIS (2602.01709); Can-David-Beat-Goliath (2601.21699); Beyond-Quantity Trajectory-Diversity (2602.03219); Pedagogically-Inspired Data Synthesis (2602.12172); Pareto-Proactive-Agents (2602.11351); DocDancer (2601.05163); MCP-SandboxScan (2601.01241); PA3 (2603.14602); Info-Theoretic Privacy for Multi-Agent (2603.05520); LBM auto-bidding (2603.05134); Real-World Customer-Service Dialogue (2510.22143); Declarative NL-over-Heterogeneous-Data (2510.16470); Coinvisor (2510.17235); FURINA (2510.06800); MedOrch (2506.00235); World-Modelling-Improves-Agents (2506.02918); When-to-Act-When-to-Wait (2506.01881); Behavior-Injection (2505.18917); The-Synthetic-Mirror (2506.13818); On-Policy Data Evolution Multimodal Search (2605.10832); Long-Horizon Tasks empirical study (2605.02572); CuraView (2605.03476); Masked-Diffusion World Models (2607.16204); ToolVerse (2607.15660); Multi-Head Latent Control (2607.14277); Every-Sample-Counts (2607.08968); Tool-Call Boundary Drift (2607.07050); When-Should-Service-Agents-Reconsider (2607.01426); Asymmetric Actor-Critic (2604.00304); Agents-Learn-Their-Runtime (2603.01209); Webscale-RL (2510.06499); Label-Consistent ABSA (2602.16379); Semantic-Context Tool Orchestration (2507.10820); Reasoning-through-Exploration FC-RL (2508.05118).

**False positives (excluded):** OpenReview "MT"=machine-translation hits (RaDis, IntGrad-MT, V1-MT motion, RNINet, kNN-ECD, MED-training, ES³Net, Parrot, MT-triage); author-name collision on "Akshara Prabhakar" (deepfake-voice, wildfire, malware, clinical-notes, CL-NERIL papers); "APIGen: Generative API Method Recommendation" (arXiv:2401.15843, SANER 2024 — code API recommendation, unrelated); DATURA (5G autoscaling).

---

## C. Implementation-relevant results (what each means for this repo)

1. **Validation is the highest-value gap.** ToolMind (2511.15718) shows trajectory-level-only validation lets turn-level errors propagate; adds **turn-level filtering** keeping self-correction signals. ToolWeave (2605.12521) tracks **parameter provenance** (reject args not from user/prior tool output), cutting hallucination and lifting 70B to 39.75% BFCL-V3 MT vs 23.50% ToolFlow. ScaleEnv (2602.06820) uses **executable action verification + procedural testing**. CoVe (2603.01940) adds constraint-guided verification. → *Replace the repo's sequence-only `executed_actions == a_gt` check with state/outcome (`o_gt`) + turn-level + provenance checks.* Directly closes README gap and CA001.
2. **Fake fixed-data tools are obsolete.** EnvScaler (2601.05808; 191 envs/7K scenarios, **rule-based trajectory-validation functions**), ScaleEnv (tool-dependency-graph expansion), AutoForge (2512.22857; high-difficulty *easily-verifiable* envs), AgentScaler (2509.13311) all synthesize **stateful, verifiable environments**. → *Replace `get_weather`/`get_news` stubs with a procedurally-verified or real stateful env; if infeasible, Simia (2511.01824) shows LLM-simulated feedback reaches ≈o4-mini on τ²-Bench (with a fidelity caveat).*
3. **Close the data→train→feedback loop.** LoopTool (2511.09148: capability probing → judge-guided label verification → error-driven expansion; 8B beats its 32B generator) and RODS (2606.19047: reward-variance boundary resampling, 400 seeds ≈ 17K offline) and EigenData (2601.22607: self-evolving prompts/workflow) are the reference designs for the README's missing "agentic feedback/refinement loop."
4. **Blueprint recombination has concrete mechanisms.** Magnet (2503.07826) graph node-operations and ToolWeave dependency-built workflows operationalize the paper's "Reverse Task Recombination," currently unimplemented here.
5. **User simulation must be hardened.** Non-Collaborative User Simulators (2509.23124, ICLR 2026) shows cooperative simulators overstate agent robustness; the original authors themselves moved to UserBench (2507.22034) + UserRL (2509.19736). → *Add non-collaborative behaviors + persona/BoN/self-critique stabilization; evaluate on UserBench.*
6. **Report τ²-Bench + BFCL-v3/v4 multi-turn**, not only τ-bench — τ²-Bench (2506.07982, 324 cit) is now the standard target for 2026 work.
7. **Baselines:** Magnet and ToolACE-MT as primary comparisons; APIGen (single-turn) as ablation anchor.

## D. Empirical replications & explicit critiques

**Replications/extensions of the method:** ToolACE-MT, FunReason-MT, TOUCAN, ToolWeave, WRIT, ToolMind, LoopTool, EigenData, Trajectory2Task, PARL-MT, FinToolSyn (all re-implement and extend blueprint→trajectory synthesis with reported BFCL/τ² numbers). In-group replication: UserBench/UserRL extend the user-simulator component.

**Critiques (empirically grounded):**
- *Cooperative user simulation is unrealistic:* Non-Collaborative User Simulators (2509.23124) — SOTA tool agents degrade markedly under 4 non-collaborative behaviors on MultiWOZ/τ-bench.
- *Trajectory-level validation is insufficient:* ToolMind (2511.15718) — turn-level errors propagate during training.
- *Autoregressive multi-agent simulation is too costly:* ToolACE-MT (2508.12685) — non-AR alternative; Simia (2511.01824) — bypasses bespoke environments entirely via LLM-simulated feedback.
- *One-shot synthesis hallucinates parameters / chains superficially-compatible tools:* ToolWeave (2605.12521).
- *Simulated-user distributional gap vs real users:* arXiv:2605.07847; Goal Alignment in User Simulators (arXiv:2507.20152, TACL).
- *Multi-turn safety degradation:* MT-AgentRisk (2602.13379) — ASR +16% in multi-turn.

## E. Saturation evidence

- Forward graph captured via S2 citation edges (both pages, ~135 of 142 resolved; count delta is DOI/ACL-Anthology-only records). Backward graph captured in full via S2 references + arXiv/OpenReview.
- Primary abstracts fetched for every material cluster (data-synthesis, environment-scaling, in-group, benchmarks, critiques); the final two waves confirmed existing clusters and surfaced **no new cluster** — the stopping criterion.
- Cross-checked across arXiv, Semantic Scholar, OpenReview, ACL/NeurIPS/ICLR/EMNLP/AAAI/ICML venue metadata, and the lead-author page (15 distinct queries).

## F. Material EXPAND leads (only)

1. **Papers-with-Code** τ-bench/BFCL multi-turn leaderboards + APIGen-MT method page (API overflowed the read-only client's buffer twice) — for third-party code and leaderboard positions.
2. **Full-text in-text citation check** for the ~15 direct follow-ups to record *how* each cites APIGen-MT (baseline vs motivation vs extension) — upgrades "uses" inference to verified.
3. **OpenReview review thread on `qk6ORqQ4Cu`** — independent reviewer-flagged limitations beyond the paper's own.
4. **Close the S2 count delta (142 vs ~135):** page citations past offset 100 for non-arXiv records (ACL Findings BalanceSFT/Progra/Beyond-Static-Toolsets currently lack arXiv IDs).
5. **HuggingFace `Salesforce/APIGen-MT-5k` + xLAM-2 model cards** — confirm official blueprint/trajectory schema to align the repo's data model.

**One coverage gap:** Papers-with-Code (client buffer limitation); substituted with canonical project resources (apigen-mt.github.io, HF dataset/models). No other blocking unknowns for the upgrade decision.
